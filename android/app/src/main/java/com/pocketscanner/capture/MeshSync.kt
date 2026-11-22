package com.pocketscanner.capture

import android.util.Log
import java.io.DataInputStream
import java.io.DataOutputStream
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.ArrayBlockingQueue
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicReference
import kotlin.math.acos

data class TrainFrame(
    val id: Int,
    val pose: FloatArray,
    val width: Int,
    val height: Int,
    val fx: Float,
    val fy: Float,
    val cx: Float,
    val cy: Float,
    val rgb: File,
)

data class ScanSettings(
    val voxelCm: Int = 8,
    val integrateWidth: Int = 160,
    val minSize: Int = 500,
    val pyramidLevels: Int = 6,
    val depthMeters: Float = 6f,
    val reviewVoxelCm: Int = 8,
    val closeLoops: Boolean = true,
    val epochs: Int = 100,
    val validation: Boolean = true,
    val validationPercent: Int = 5,
    val validationEvery: Int = 5,
    val gaussBudget: Int = 3_000_000,
    val videoSeconds: Int = 10,
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
    private val fullDir: File? = null,
) {
    private val running = AtomicBoolean(true)
    private val acceptFrames = AtomicBoolean(false)
    private val settings = AtomicReference(ScanSettings())
    private val camera = AtomicReference<SlamCamera?>(null)
    private val ring = LinkedHashMap<Long, Snap>()
    private val fullFiles = LinkedHashMap<Long, Stored>()
    private val held = HashMap<Int, Held>()
    private var lens: ViewLens? = null
    private val covered = HashMap<Int, Held>()
    private val missed = HashSet<Int>()
    private val bare = HashSet<Int>()
    private val trackedPose = AtomicReference<Tracked?>(null)
    private var nextCoverId = COVER_ID_BASE
    private var loggedRing = false
    private var loggedSyncNs = 0L
    private val lock = Any()
    private val recording = AtomicBoolean(false)
    private val recorded = AtomicInteger(0)
    private val thread = Thread(::loop, "mesh-sync")
    private val saveQueue = ArrayBlockingQueue<SaveJob>(2)
    private val saving = ConcurrentHashMap.newKeySet<Long>()
    private val saveEpoch = AtomicInteger(0)
    private val publishedTime = AtomicReference(Double.NaN)
    private val saver = Thread(::saveLoop, "frame-save")
    private var loggedDrop = false

    fun start() {
        fullDir?.mkdirs()
        fullDir?.listFiles()?.forEach { it.delete() }
        thread.start()
        saver.start()
    }

    fun close() {
        running.set(false)
        thread.interrupt()
        saver.interrupt()
        thread.join(2000)
        saver.join(2000)
    }

    fun setCamera(next: SlamCamera?) {
        camera.set(next)
    }

    fun apply(next: ScanSettings) {
        settings.set(next)
        saveEpoch.incrementAndGet()
        val doomed = synchronized(lock) {
            val files = collectFullFiles()
            ring.clear()
            fullFiles.clear()
            held.clear()
            covered.clear()
            missed.clear()
            bare.clear()
            lens = null
            nextCoverId = COVER_ID_BASE
            loggedRing = false
            publishedTime.set(Double.NaN)
            if (recording.get()) {
                recorded.set(0)
                resetDump()
            }
            files
        }
        doomed.forEach { it.delete() }
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

    fun storedLens(): ViewLens? = synchronized(lock) { lens }

    fun exportTraining(dest: File): List<TrainFrame> {
        dest.mkdirs()
        val shots = synchronized(lock) {
            ring.clear()
            held.entries
                .filter { it.key < COVER_ID_BASE && it.value.full != null }
                .sortedBy { it.key }
                .map { it.key to it.value }
        }
        val written = ArrayList<TrainFrame>(shots.size)
        for ((id, frame) in shots) {
            val full = frame.full ?: continue
            try {
                val raw = readStored(full.file)
                val rgb = if (raw.colorStride >= full.width * 3 && raw.color.size >= full.width * full.height * 3) {
                    raw.color
                } else {
                    yuyvToRgb(raw.color, full.width, full.height)
                }
                val file = File(dest, "$id.rgb")
                file.writeBytes(rgb)
                written.add(
                    TrainFrame(
                        id,
                        frame.pose.copyOf(),
                        full.width,
                        full.height,
                        full.fx,
                        full.fy,
                        full.cx,
                        full.cy,
                        file,
                    ),
                )
            } catch (error: Exception) {
                Log.w(TAG, "export $id: ${error.message}")
            }
        }
        return written
    }

    /** Drops the working frames after a training run has its own copies. */
    fun releaseTrainingSources() {
        acceptFrames.set(false)
        saveEpoch.incrementAndGet()
        val doomed = synchronized(lock) {
            val files = collectFullFiles()
            ring.clear()
            fullFiles.clear()
            held.clear()
            covered.clear()
            missed.clear()
            bare.clear()
            files
        }
        doomed.forEach { it.delete() }
    }

    private fun yuyvToRgb(yuyv: ByteArray, width: Int, height: Int): ByteArray {
        val out = ByteArray(width * height * 3)
        var i = 0
        var o = 0
        val pairs = minOf(width * height / 2, yuyv.size / 4)
        repeat(pairs) {
            val y0 = yuyv[i].toInt() and 255
            val u = yuyv[i + 1].toInt() and 255
            val y1 = yuyv[i + 2].toInt() and 255
            val v = yuyv[i + 3].toInt() and 255
            i += 4
            val first = yuv(y0, u, v)
            val second = yuv(y1, u, v)
            out[o++] = first.first.toByte()
            out[o++] = first.second.toByte()
            out[o++] = first.third.toByte()
            out[o++] = second.first.toByte()
            out[o++] = second.second.toByte()
            out[o++] = second.third.toByte()
        }
        return out
    }

    fun oldestPose(): FloatArray? = synchronized(lock) {
        held.entries.filter { it.key < COVER_ID_BASE }.minByOrNull { it.key }?.value?.pose?.copyOf()
    }

    fun writePackage(dest: java.io.OutputStream, scan: ScanSettings, mesh: File?): Int {
        val shots = synchronized(lock) {
            held.entries
                .filter { it.key < COVER_ID_BASE && it.value.full != null }
                .sortedBy { it.key }
                .map { (id, frame) ->
                    val full = frame.full!!
                    ShotMeta(
                        id,
                        frame.pose.copyOf(),
                        full.width,
                        full.height,
                        full.fx,
                        full.fy,
                        full.cx,
                        full.cy,
                        full.depthScale,
                        full.file,
                    )
                }
        }
        if (shots.isEmpty()) return 0
        ScanPackage.write(dest, scan, shots, mesh) { meta ->
            val raw = readStored(meta.file)
            PackageShot(
                meta.id,
                meta.pose,
                meta.width,
                meta.height,
                meta.fx,
                meta.fy,
                meta.cx,
                meta.cy,
                meta.depthScale,
                raw.color,
                raw.colorStride,
                raw.depth,
                raw.depthStride,
            )
        }
        return shots.size
    }

    internal fun restore(rebuildMesh: Boolean, feed: ((PackageFrame) -> Unit) -> Unit): Int {
        saveEpoch.incrementAndGet()
        val built = ArrayList<Pair<Int, Held>>()
        var eye: ViewLens? = null
        feed { frame ->
            val scan = settings.get()
            if (frame.id >= COVER_ID_BASE) return@feed
            val yuyv = rgbToYuyv(frame.rgb, frame.width, frame.height)
            val model = SlamCamera(
                w = frame.width,
                h = frame.height,
                fps = 30,
                fx = frame.fx.toDouble(),
                fy = frame.fy.toDouble(),
                cx = frame.cx.toDouble(),
                cy = frame.cy.toDouble(),
                k1 = 0.0,
                k2 = 0.0,
                p1 = 0.0,
                p2 = 0.0,
                k3 = 0.0,
                depthFactor = frame.depthScale.toDouble(),
                note = "package",
            )
            val snap = downsample(
                yuyv,
                frame.width * 2,
                frame.depth,
                frame.width * 2,
                frame.width,
                frame.height,
                scan.integrateWidth,
                scan.integrateHeight,
                model,
                scan.depthMeters,
            )
            val stored = storeFrame(
                frame.id.toLong(),
                yuyv,
                frame.width * 2,
                frame.depth,
                frame.width * 2,
                frame.width,
                frame.height,
                frame.fx,
                frame.fy,
                frame.cx,
                frame.cy,
                frame.depthScale,
            )
            built.add(frame.id to Held(snap, frame.pose.copyOf(), stored))
            if (eye == null) {
                eye = ViewLens(frame.fx, frame.fy, frame.cx, frame.cy, frame.width, frame.height)
            }
        }
        val scan = settings.get()
        val doomed = synchronized(lock) {
            val files = collectFullFiles()
            ring.clear()
            fullFiles.clear()
            covered.clear()
            missed.clear()
            bare.clear()
            held.clear()
            for ((id, frame) in built) held[id] = frame
            lens = eye
            files
        }
        doomed.forEach { it.delete() }
        if (!rebuildMesh) return built.size
        val view = preview()
        if (view != null) {
            view.meshConfigure(scan.reviewVoxelCm / 100f, scan.depthMeters)
            synchronized(lock) {
                for ((id, frame) in held) {
                    view.meshUpsert(
                        id,
                        frame.pose,
                        frame.snap.depth,
                        frame.snap.color,
                        frame.snap.width,
                        frame.snap.height,
                        frame.snap.fx,
                        frame.snap.fy,
                        frame.snap.cx,
                        frame.snap.cy,
                        force = true,
                    )
                }
            }
        }
        return built.size
    }

    fun adoptPoses() {
        val view = preview() ?: return
        val poll = tracker.copyKeyframes()
        synchronized(lock) {
            for (id in covered.keys.toList()) view.meshErase(id)
            covered.clear()
            val live = HashSet<Int>()
            for (keyframe in poll.frames) {
                live.add(keyframe.id)
                val heldFrame = held[keyframe.id] ?: continue
                held[keyframe.id] = Held(heldFrame.snap, keyframe.pose.copyOf(), heldFrame.full)
                upsertHeld(view, keyframe.id, held[keyframe.id]!!)
            }
            for ((id, frame) in held) {
                if (id in live) continue
                upsertHeld(view, id, frame)
            }
        }
    }

    private fun upsertHeld(view: VkPreview, id: Int, frame: Held) {
        view.meshUpsert(
            id,
            frame.pose,
            frame.snap.depth,
            frame.snap.color,
            frame.snap.width,
            frame.snap.height,
            frame.snap.fx,
            frame.snap.fy,
            frame.snap.cx,
            frame.snap.cy,
            force = true,
        )
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
        val bits = java.lang.Double.doubleToRawLongBits(timestampSec)
        if (acceptFrames.get() && fullDir != null) {
            val job = SaveJob(
                saveEpoch.get(),
                bits,
                color,
                colorStride,
                depth,
                depthStride,
                width,
                height,
                model.fx.toFloat(),
                model.fy.toFloat(),
                model.cx.toFloat(),
                model.cy.toFloat(),
                model.depthFactor.toFloat(),
            )
            saving.add(bits)
            if (!saveQueue.offer(job)) {
                saving.remove(bits)
                if (!loggedDrop) {
                    loggedDrop = true
                    Log.i(TAG, "frame save queue full, dropping a full frame")
                }
            }
        }
        synchronized(lock) {
            ring[bits] = snap
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

    private fun takeGrab(timestampSec: Double): Take? {
        val key = matchKey(timestampSec) ?: return null
        if (!fullFiles.containsKey(key) && saving.contains(key)) return Take.Later
        val snap = ring.remove(key) ?: return null
        return Take.Ready(Grab(snap, fullFiles.remove(key)))
    }

    private fun dropOlderThan(published: Double, doomed: MutableList<File>) {
        publishedTime.set(published)
        val snaps = ring.entries.iterator()
        while (snaps.hasNext()) {
            val time = java.lang.Double.longBitsToDouble(snaps.next().key)
            if (time < published) snaps.remove()
        }
        val full = fullFiles.entries.iterator()
        while (full.hasNext()) {
            val entry = full.next()
            val time = java.lang.Double.longBitsToDouble(entry.key)
            if (time < published) {
                doomed.add(entry.value.file)
                full.remove()
            }
        }
    }

    private fun matchKey(timestampSec: Double): Long? {
        val bits = java.lang.Double.doubleToRawLongBits(timestampSec)
        if (ring.containsKey(bits)) return bits
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
        return if (found) bestKey else null
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
        val doomed = ArrayList<File>()
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
                    val taken = if (existing == null) takeGrab(keyframe.timestampSec) else null
                    if (taken is Take.Later) continue
                    val grab = (taken as? Take.Ready)?.grab
                    val snap = existing?.snap ?: grab?.snap
                    if (snap == null) {
                        if (missed.add(keyframe.id)) {
                            Log.i(TAG, "keyframe ${keyframe.id} has no depth snap, ring ${ring.size}")
                        }
                        continue
                    }
                    if (existing != null && !moved(existing.pose, keyframe.pose)) continue
                    if (!upsert(view, keyframe.id, keyframe.pose, snap)) continue
                    val full = existing?.full ?: grab?.full
                    if (full == null && bare.add(keyframe.id)) {
                        Log.i(TAG, "keyframe ${keyframe.id} has no full frame")
                    }
                    held[keyframe.id] = Held(snap, keyframe.pose.copyOf(), full)
                    upserts++
                }
                if (!published.isNaN()) dropOlderThan(published, doomed)
                if (poll.complete) {
                    val gone = held.keys.filter { it !in live }
                    for (id in gone) {
                        held.remove(id)?.full?.file?.let { doomed.add(it) }
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
        doomed.forEach { it.delete() }
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

    private class Grab(val snap: Snap, val full: Stored?)

    private class Stored(
        val file: File,
        val width: Int,
        val height: Int,
        val colorStride: Int,
        val depthStride: Int,
        val fx: Float,
        val fy: Float,
        val cx: Float,
        val cy: Float,
        val depthScale: Float,
    )

    private class SaveJob(
        val epoch: Int,
        val bits: Long,
        val color: ByteArray,
        val colorStride: Int,
        val depth: ByteArray,
        val depthStride: Int,
        val width: Int,
        val height: Int,
        val fx: Float,
        val fy: Float,
        val cx: Float,
        val cy: Float,
        val depthScale: Float,
    )

    private sealed class Take {
        class Ready(val grab: Grab) : Take()
        data object Later : Take()
    }

    private class Held(val snap: Snap, val pose: FloatArray, val full: Stored? = null)

    private fun collectFullFiles(): List<File> {
        val files = ArrayList<File>()
        fullFiles.values.forEach { files.add(it.file) }
        held.values.forEach { it.full?.file?.let(files::add) }
        return files
    }

    private fun saveLoop() {
        while (running.get()) {
            val job = try {
                saveQueue.poll(200, TimeUnit.MILLISECONDS)
            } catch (_: InterruptedException) {
                null
            } ?: continue
            try {
                saveJob(job)
            } catch (error: Exception) {
                Log.w(TAG, "save frame: ${error.message}")
            } finally {
                saving.remove(job.bits)
            }
        }
    }

    private fun saveJob(job: SaveJob) {
        val dir = fullDir ?: return
        if (job.epoch != saveEpoch.get()) return
        val time = java.lang.Double.longBitsToDouble(job.bits)
        val published = publishedTime.get()
        if (!published.isNaN() && time < published) return
        dir.mkdirs()
        val dest = File(dir, "${job.bits}.bin")
        val tmp = File(dir, "${job.bits}.tmp")
        writeStored(
            tmp,
            job.color,
            job.colorStride,
            job.depth,
            job.depthStride,
            job.width,
            job.height,
            job.fx,
            job.fy,
            job.cx,
            job.cy,
            job.depthScale,
        )
        if (!tmp.renameTo(dest)) {
            tmp.copyTo(dest, overwrite = true)
            tmp.delete()
        }
        val overflow = ArrayList<File>()
        synchronized(lock) {
            val publishedNow = publishedTime.get()
            if (job.epoch != saveEpoch.get() || (!publishedNow.isNaN() && time < publishedNow)) {
                dest.delete()
                return
            }
            fullFiles[job.bits] = Stored(
                dest,
                job.width,
                job.height,
                job.colorStride,
                job.depthStride,
                job.fx,
                job.fy,
                job.cx,
                job.cy,
                job.depthScale,
            )
            while (fullFiles.size > FULL_DISK_KEEP) {
                val oldest = fullFiles.entries.iterator()
                if (!oldest.hasNext()) break
                overflow.add(oldest.next().value.file)
                oldest.remove()
            }
        }
        overflow.forEach { it.delete() }
    }

    private fun storeFrame(
        id: Long,
        color: ByteArray,
        colorStride: Int,
        depth: ByteArray,
        depthStride: Int,
        width: Int,
        height: Int,
        fx: Float,
        fy: Float,
        cx: Float,
        cy: Float,
        depthScale: Float,
    ): Stored {
        val dir = fullDir ?: error("No frame directory")
        dir.mkdirs()
        val dest = File(dir, "kept-$id.bin")
        writeStored(dest, color, colorStride, depth, depthStride, width, height, fx, fy, cx, cy, depthScale)
        return Stored(dest, width, height, colorStride, depthStride, fx, fy, cx, cy, depthScale)
    }

    companion object {
        private const val TAG = "pocket-mesh"
        private const val RING_KEEP = 1500
        private const val FULL_DISK_KEEP = 160
        private const val COVER_ID_BASE = 1_000_000_000
        private const val FRAME_MAGIC = 0x50534631

        private class RawFrame(
            val color: ByteArray,
            val colorStride: Int,
            val depth: ByteArray,
            val depthStride: Int,
        )

        private fun writeStored(
            dest: File,
            color: ByteArray,
            colorStride: Int,
            depth: ByteArray,
            depthStride: Int,
            width: Int,
            height: Int,
            fx: Float,
            fy: Float,
            cx: Float,
            cy: Float,
            depthScale: Float,
        ) {
            DataOutputStream(FileOutputStream(dest).buffered()).use { out ->
                out.writeInt(FRAME_MAGIC)
                out.writeInt(width)
                out.writeInt(height)
                out.writeInt(colorStride)
                out.writeInt(depthStride)
                out.writeFloat(fx)
                out.writeFloat(fy)
                out.writeFloat(cx)
                out.writeFloat(cy)
                out.writeFloat(depthScale)
                out.writeInt(color.size)
                out.write(color)
                out.writeInt(depth.size)
                out.write(depth)
            }
        }

        private fun readStored(file: File): RawFrame {
            DataInputStream(FileInputStream(file).buffered()).use { input ->
                val magic = input.readInt()
                if (magic != FRAME_MAGIC) error("Bad frame file")
                input.readInt()
                input.readInt()
                val colorStride = input.readInt()
                val depthStride = input.readInt()
                input.readFloat()
                input.readFloat()
                input.readFloat()
                input.readFloat()
                input.readFloat()
                val color = ByteArray(input.readInt())
                input.readFully(color)
                val depth = ByteArray(input.readInt())
                input.readFully(depth)
                return RawFrame(color, colorStride, depth, depthStride)
            }
        }

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

        private fun rgbToYuyv(rgb: ByteArray, width: Int, height: Int): ByteArray {
            val out = ByteArray(width * height * 2)
            var i = 0
            var o = 0
            repeat(height) {
                var x = 0
                while (x < width) {
                    val r0 = rgb[i].toInt() and 255
                    val g0 = rgb[i + 1].toInt() and 255
                    val b0 = rgb[i + 2].toInt() and 255
                    val r1: Int
                    val g1: Int
                    val b1: Int
                    if (x + 1 < width) {
                        r1 = rgb[i + 3].toInt() and 255
                        g1 = rgb[i + 4].toInt() and 255
                        b1 = rgb[i + 5].toInt() and 255
                        i += 6
                    } else {
                        r1 = r0
                        g1 = g0
                        b1 = b0
                        i += 3
                    }
                    out[o++] = yuvLuma(r0, g0, b0)
                    out[o++] = yuvChroma(r0, g0, b0, r1, g1, b1, chroma = true)
                    out[o++] = yuvLuma(r1, g1, b1)
                    out[o++] = yuvChroma(r0, g0, b0, r1, g1, b1, chroma = false)
                    x += 2
                }
            }
            return out
        }

        private fun yuvLuma(red: Int, green: Int, blue: Int): Byte {
            val value = ((66 * red + 129 * green + 25 * blue + 128) shr 8) + 16
            return value.coerceIn(0, 255).toByte()
        }

        private fun yuvChroma(
            r0: Int,
            g0: Int,
            b0: Int,
            r1: Int,
            g1: Int,
            b1: Int,
            chroma: Boolean,
        ): Byte {
            val red = (r0 + r1) / 2
            val green = (g0 + g1) / 2
            val blue = (b0 + b1) / 2
            val value = if (chroma) {
                ((-38 * red - 74 * green + 112 * blue + 128) shr 8) + 128
            } else {
                ((112 * red - 94 * green - 18 * blue + 128) shr 8) + 128
            }
            return value.coerceIn(0, 255).toByte()
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
