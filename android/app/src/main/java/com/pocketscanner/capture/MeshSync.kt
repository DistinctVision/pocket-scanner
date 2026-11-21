package com.pocketscanner.capture

import android.util.Log
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicReference
import kotlin.math.acos

data class ScanSettings(
    val voxelCm: Int = 8,
    val integrateWidth: Int = 160,
    val minSize: Int = 500,
    val pyramidLevels: Int = 6,
    val depthMeters: Float = 6f,
    val reviewVoxelCm: Int = 2,
    val closeLoops: Boolean = true,
) {
    val integrateHeight: Int get() = integrateWidth * 3 / 4

    fun tuning(): SlamTuning = SlamTuning(minSize, pyramidLevels, depthMeters)
}

/**
 * Stella keeps only sparse ORB depths. This thread snapshots the downsampled
 * depth and color we fed, then copies a keyframe into the GPU volume when its
 * timestamp shows up in the map.
 */
class MeshSync(
    private val tracker: StellaTracker,
    private val preview: () -> VkPreview?,
    private val dumpDir: File? = null,
) {
    private val running = AtomicBoolean(true)
    private val acceptFrames = AtomicBoolean(false)
    private val settings = AtomicReference(ScanSettings())
    private val camera = AtomicReference<SlamCamera?>(null)
    private val ring = LinkedHashMap<Long, Snap>()
    private val held = HashMap<Int, Held>()
    private val covered = HashMap<Int, Held>()
    private val missed = HashSet<Int>()
    private val trackedPose = AtomicReference<Tracked?>(null)
    private var nextCoverId = COVER_ID_BASE
    private var loggedRing = false
    private var loggedSyncNs = 0L
    private val lock = Any()
    private val recording = AtomicBoolean(false)
    private val recorded = AtomicInteger(0)
    private val thread = Thread(::loop, "mesh-sync")

    fun start() {
        thread.start()
    }

    fun close() {
        running.set(false)
        thread.interrupt()
        thread.join(2000)
    }

    fun setCamera(next: SlamCamera?) {
        camera.set(next)
    }

    fun apply(next: ScanSettings) {
        settings.set(next)
        synchronized(lock) {
            ring.clear()
            held.clear()
            covered.clear()
            missed.clear()
            nextCoverId = COVER_ID_BASE
            loggedRing = false
            if (recording.get()) {
                recorded.set(0)
                resetDump()
            }
        }
        preview()?.meshConfigure(next.voxelCm / 100f, next.depthMeters)
    }

    fun replaceSettings(next: ScanSettings) {
        settings.set(next)
    }

    /** Drop the current mesh and integrate the stored keyframes at this voxel size. */
    fun rebuild(voxelCm: Int) {
        val depth = settings.get().depthMeters
        preview()?.meshConfigure(voxelCm / 100f, depth)
        adoptPoses()
    }

    fun setRecording(enabled: Boolean) {
        val was = recording.getAndSet(enabled)
        if (enabled && !was) {
            recorded.set(0)
            synchronized(lock) { resetDump() }
        }
    }

    fun recordedCount(): Int = recorded.get()

    fun saveScan(dir: File): Int {
        val shots = synchronized(lock) {
            held.map { (id, frame) -> Triple(id, frame.pose.copyOf(), frame.snap) }
        }
        dir.mkdirs()
        dir.listFiles()?.forEach { it.delete() }
        val header = "id,width,height,fx,fy,cx,cy," + (0 until 16).joinToString(",") { "p$it" } + "\n"
        val index = StringBuilder(header)
        var written = 0
        for ((id, pose, snap) in shots) {
            if (id >= COVER_ID_BASE) continue
            try {
                val depth = ByteBuffer.allocate(snap.depth.size * 4).order(ByteOrder.LITTLE_ENDIAN)
                for (value in snap.depth) depth.putFloat(value)
                File(dir, "$id.f32").writeBytes(depth.array())
                File(dir, "$id.rgb").writeBytes(snap.color)
                index.append(
                    "$id,${snap.width},${snap.height},${snap.fx},${snap.fy},${snap.cx},${snap.cy}," +
                        pose.joinToString(",") + "\n",
                )
                written++
            } catch (error: Exception) {
                Log.w(TAG, "save $id: ${error.message}")
            }
        }
        File(dir, "index.csv").writeText(index.toString())
        Log.i(TAG, "saved $written keyframes to ${dir.absolutePath}")
        return written
    }

    fun adoptPoses() {
        val view = preview() ?: return
        val poll = tracker.copyKeyframes()
        synchronized(lock) {
            for (id in covered.keys.toList()) view.meshErase(id)
            covered.clear()
            for (keyframe in poll.frames) {
                val heldFrame = held[keyframe.id] ?: continue
                held[keyframe.id] = Held(heldFrame.snap, keyframe.pose.copyOf())
                view.meshUpsert(
                    keyframe.id,
                    keyframe.pose,
                    heldFrame.snap.depth,
                    heldFrame.snap.color,
                    heldFrame.snap.width,
                    heldFrame.snap.height,
                    heldFrame.snap.fx,
                    heldFrame.snap.fy,
                    heldFrame.snap.cx,
                    heldFrame.snap.cy,
                    force = true,
                )
            }
        }
    }

    fun setTracking(active: Boolean) {
        acceptFrames.set(active)
        if (!active) trackedPose.set(null)
    }

    fun track(timestampSec: Double, pose: FloatArray) {
        if (!acceptFrames.get()) return
        trackedPose.set(Tracked(timestampSec, pose.copyOf(16)))
    }

    fun note(
        timestampSec: Double,
        color: ByteArray,
        colorStride: Int,
        depth: ByteArray,
        depthStride: Int,
        width: Int,
        height: Int,
    ) {
        val model = camera.get() ?: return
        val scan = settings.get()
        val outW = scan.integrateWidth
        val outH = scan.integrateHeight
        if (outW <= 0 || outH <= 0 || outW > width || outH > height) return
        val snap = downsample(color, colorStride, depth, depthStride, width, height, outW, outH, model, scan.depthMeters)
        synchronized(lock) {
            ring[java.lang.Double.doubleToRawLongBits(timestampSec)] = snap
            val iterator = ring.entries.iterator()
            while (ring.size > RING_KEEP && iterator.hasNext()) {
                iterator.next()
                iterator.remove()
                if (!loggedRing) {
                    loggedRing = true
                    Log.i(TAG, "depth ring cap $RING_KEEP, dropping unpublished snaps")
                }
            }
        }
    }

    private fun takeSnap(timestampSec: Double): Snap? {
        val bits = java.lang.Double.doubleToRawLongBits(timestampSec)
        ring.remove(bits)?.let { return it }
        var bestKey = 0L
        var bestDt = 0.05
        var found = false
        for (key in ring.keys) {
            val dt = kotlin.math.abs(java.lang.Double.longBitsToDouble(key) - timestampSec)
            if (dt < bestDt) {
                bestDt = dt.toDouble()
                bestKey = key
                found = true
            }
        }
        if (!found) return null
        return ring.remove(bestKey)
    }

    private fun peekSnap(timestampSec: Double): Snap? {
        val bits = java.lang.Double.doubleToRawLongBits(timestampSec)
        ring[bits]?.let { return it }
        var bestKey = 0L
        var bestDt = 0.05
        var found = false
        for (key in ring.keys) {
            val dt = kotlin.math.abs(java.lang.Double.longBitsToDouble(key) - timestampSec)
            if (dt < bestDt) {
                bestDt = dt.toDouble()
                bestKey = key
                found = true
            }
        }
        if (!found) return null
        return ring[bestKey]
    }

    private fun loop() {
        while (running.get()) {
            try {
                sync()
            } catch (error: Exception) {
                Log.w(TAG, "sync: ${error.message}")
            }
            try {
                Thread.sleep(200)
            } catch (_: InterruptedException) {
                if (!running.get()) break
            }
        }
    }

    private fun sync() {
        val view = preview() ?: return
        val poll = tracker.copyKeyframes()
        val current = poll.frames
        synchronized(lock) {
            if (!acceptFrames.get()) {
                trackedPose.set(null)
                return
            }
            var upserts = 0
            if (current.isNotEmpty()) {
                val live = current.map { it.id }.toSet()
                var published = Double.NaN
                for (keyframe in current) {
                    if (published.isNaN() || keyframe.timestampSec > published) published = keyframe.timestampSec
                    // Drop the tracking frame first, so the keyframe is not rejected as a duplicate of it.
                    dropCoveredNear(view, keyframe.pose)
                    val existing = held[keyframe.id]
                    val snap = existing?.snap ?: takeSnap(keyframe.timestampSec)
                    if (snap == null) {
                        if (missed.add(keyframe.id)) {
                            Log.i(TAG, "keyframe ${keyframe.id} has no depth snap, ring ${ring.size}")
                        }
                        continue
                    }
                    if (existing != null && !moved(existing.pose, keyframe.pose)) continue
                    if (!upsert(view, keyframe.id, keyframe.pose, snap)) continue
                    held[keyframe.id] = Held(snap, keyframe.pose.copyOf())
                    upserts++
                }
                if (!published.isNaN()) {
                    val iterator = ring.entries.iterator()
                    while (iterator.hasNext()) {
                        val time = java.lang.Double.longBitsToDouble(iterator.next().key)
                        if (time < published) iterator.remove()
                    }
                }
                if (poll.complete) {
                    val gone = held.keys.filter { it !in live }
                    for (id in gone) {
                        held.remove(id)
                        view.meshErase(id)
                    }
                }
            }
            if (coverTracking(view, current)) upserts++
            val now = System.nanoTime()
            if (now - loggedSyncNs > 1_000_000_000L) {
                loggedSyncNs = now
                Log.i(
                    TAG,
                    "keyframes=${current.size} covered=${covered.size} held=${held.size} missed=${missed.size} ring=${ring.size} upserts=$upserts",
                )
            }
        }
    }

    private fun dropCoveredNear(view: VkPreview, pose: FloatArray) {
        val near = covered.filter { (_, heldFrame) -> !moved(heldFrame.pose, pose) }.keys.toList()
        for (id in near) {
            covered.remove(id)
            view.meshErase(id)
        }
    }

    /** Stella's mapping often stops inserting keyframes. Cover the walk from the tracking pose. */
    private fun coverTracking(view: VkPreview, keyframes: List<KeyframePose>): Boolean {
        val track = trackedPose.get() ?: return false
        if (keyframes.any { !moved(it.pose, track.pose) }) return false
        if (covered.values.any { !moved(it.pose, track.pose) }) return false
        val snap = peekSnap(track.timestampSec) ?: return false
        val id = nextCoverId++
        if (!upsert(view, id, track.pose, snap)) return false
        covered[id] = Held(snap, track.pose.copyOf())
        return true
    }

    private fun upsert(view: VkPreview, id: Int, pose: FloatArray, snap: Snap): Boolean {
        val accepted = view.meshUpsert(
            id,
            pose,
            snap.depth,
            snap.color,
            snap.width,
            snap.height,
            snap.fx,
            snap.fy,
            snap.cx,
            snap.cy,
        )
        if (accepted) recordDump(id, pose, snap)
        return accepted
    }

    private fun resetDump() {
        val dir = dumpDir ?: return
        dir.mkdirs()
        dir.listFiles()?.forEach { it.delete() }
        File(dir, "index.csv").writeText(
            "id,kind,valid,pixels,width,height,fx,fy,cx,cy," +
                (0 until 16).joinToString(",") { "p$it" } + "\n",
        )
    }

    private fun recordDump(id: Int, pose: FloatArray, snap: Snap) {
        val dir = dumpDir ?: return
        if (!recording.get()) return
        try {
            var valid = 0
            val bytes = ByteBuffer.allocate(snap.depth.size * 4).order(ByteOrder.LITTLE_ENDIAN)
            for (depth in snap.depth) {
                if (depth > 0.05f) valid++
                bytes.putFloat(depth)
            }
            File(dir, "$id.f32").writeBytes(bytes.array())
            val kind = if (id >= COVER_ID_BASE) "cover" else "key"
            val poseText = pose.joinToString(",")
            File(dir, "index.csv").appendText(
                "$id,$kind,$valid,${snap.depth.size},${snap.width},${snap.height}," +
                    "${snap.fx},${snap.fy},${snap.cx},${snap.cy},$poseText\n",
            )
            recorded.incrementAndGet()
        } catch (error: Exception) {
            Log.w(TAG, "dump: ${error.message}")
        }
    }

    private class Snap(
        val depth: FloatArray,
        val color: ByteArray,
        val width: Int,
        val height: Int,
        val fx: Float,
        val fy: Float,
        val cx: Float,
        val cy: Float,
        val addedNs: Long,
    )

    private class Tracked(val timestampSec: Double, val pose: FloatArray)

    private class Held(val snap: Snap, val pose: FloatArray)

    companion object {
        private const val TAG = "pocket-mesh"
        private const val RING_KEEP = 1500
        private const val COVER_ID_BASE = 1_000_000_000

        private fun downsample(
            color: ByteArray,
            colorStride: Int,
            depth: ByteArray,
            depthStride: Int,
            width: Int,
            height: Int,
            outW: Int,
            outH: Int,
            model: SlamCamera,
            maxDepth: Float,
        ): Snap {
            val depthOut = FloatArray(outW * outH)
            val colorOut = ByteArray(outW * outH * 3)
            val factor = model.depthFactor.toFloat()
            val binDepth = FloatArray(64)
            val binX = IntArray(64)
            val binY = IntArray(64)
            for (y in 0 until outH) {
                val y0 = y * height / outH
                val y1 = ((y + 1) * height / outH).coerceAtLeast(y0 + 1).coerceAtMost(height)
                for (x in 0 until outW) {
                    val x0 = x * width / outW
                    val x1 = ((x + 1) * width / outW).coerceAtLeast(x0 + 1).coerceAtMost(width)
                    var count = 0
                    for (srcY in y0 until y1) {
                        for (srcX in x0 until x1) {
                            if (count >= binDepth.size) break
                            val depthIndex = srcY * depthStride + srcX * 2
                            val raw = if (depthIndex + 1 < depth.size) {
                                (depth[depthIndex].toInt() and 0xff) or ((depth[depthIndex + 1].toInt() and 0xff) shl 8)
                            } else {
                                0
                            }
                            val meters = if (raw == 0 || factor <= 0f) 0f else raw / factor
                            if (meters <= 0f || meters > maxDepth) continue
                            binDepth[count] = meters
                            binX[count] = srcX
                            binY[count] = srcY
                            count++
                        }
                    }
                    val pixel = y * outW + x
                    if (count == 0) continue
                    for (i in 1 until count) {
                        val sample = binDepth[i]
                        val sx = binX[i]
                        val sy = binY[i]
                        var j = i
                        while (j > 0 && binDepth[j - 1] > sample) {
                            binDepth[j] = binDepth[j - 1]
                            binX[j] = binX[j - 1]
                            binY[j] = binY[j - 1]
                            j--
                        }
                        binDepth[j] = sample
                        binX[j] = sx
                        binY[j] = sy
                    }
                    val mid = count / 2
                    depthOut[pixel] = binDepth[mid]
                    val srcX = binX[mid]
                    val srcY = binY[mid]
                    val pair = srcY * colorStride + (srcX / 2) * 4
                    if (pair + 3 >= color.size) continue
                    val luma = if (srcX and 1 == 0) color[pair].toInt() and 0xff else color[pair + 2].toInt() and 0xff
                    val u = color[pair + 1].toInt() and 0xff
                    val v = color[pair + 3].toInt() and 0xff
                    val (red, green, blue) = yuv(luma, u, v)
                    val out = pixel * 3
                    colorOut[out] = red.toByte()
                    colorOut[out + 1] = green.toByte()
                    colorOut[out + 2] = blue.toByte()
                }
            }
            val scaleX = outW.toFloat() / width.toFloat()
            val scaleY = outH.toFloat() / height.toFloat()
            return Snap(
                depth = depthOut,
                color = colorOut,
                width = outW,
                height = outH,
                fx = model.fx.toFloat() * scaleX,
                fy = model.fy.toFloat() * scaleY,
                cx = model.cx.toFloat() * scaleX,
                cy = model.cy.toFloat() * scaleY,
                addedNs = System.nanoTime(),
            )
        }

        private fun yuv(luma: Int, u: Int, v: Int): Triple<Int, Int, Int> {
            val c = luma - 16
            val d = u - 128
            val e = v - 128
            val red = (298 * c + 409 * e + 128) / 256
            val green = (298 * c - 100 * d - 208 * e + 128) / 256
            val blue = (298 * c + 516 * d + 128) / 256
            return Triple(red.coerceIn(0, 255), green.coerceIn(0, 255), blue.coerceIn(0, 255))
        }

        private fun moved(before: FloatArray, after: FloatArray): Boolean {
            val dx = before[3] - after[3]
            val dy = before[7] - after[7]
            val dz = before[11] - after[11]
            if (dx * dx + dy * dy + dz * dz > 0.25f * 0.25f) return true
            var dot = before[0] * after[0] + before[1] * after[1] + before[2] * after[2]
            dot += before[4] * after[4] + before[5] * after[5] + before[6] * after[6]
            dot += before[8] * after[8] + before[9] * after[9] + before[10] * after[10]
            val cosine = ((dot - 1f) / 2f).coerceIn(-1f, 1f)
            return acos(cosine) > 12f * Math.PI.toFloat() / 180f
        }
    }
}
