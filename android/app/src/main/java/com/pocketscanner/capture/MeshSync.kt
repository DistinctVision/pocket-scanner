package com.pocketscanner.capture

import android.util.Log
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicReference
import kotlin.math.acos

data class ScanSettings(
    val voxelCm: Int = 4,
    val integrateWidth: Int = 160,
    val minSize: Int = 500,
    val pyramidLevels: Int = 6,
    val depthMeters: Float = 6f,
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
) {
    private val running = AtomicBoolean(true)
    private val settings = AtomicReference(ScanSettings())
    private val camera = AtomicReference<SlamCamera?>(null)
    private val ring = LinkedHashMap<Long, Snap>()
    private val held = HashMap<Int, Held>()
    private val lock = Any()
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
        }
        preview()?.meshConfigure(next.voxelCm / 100f, next.depthMeters)
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
            val cutoff = System.nanoTime() - 8_000_000_000L
            val iterator = ring.entries.iterator()
            while (iterator.hasNext() && ring.size > 24) {
                val entry = iterator.next()
                if (entry.value.addedNs >= cutoff) break
                iterator.remove()
            }
        }
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
        val voxel = settings.get().voxelCm / 100f
        val current = tracker.copyKeyframes()
        val live = current.map { it.id }.toSet()
        synchronized(lock) {
            for (keyframe in current) {
                val bits = java.lang.Double.doubleToRawLongBits(keyframe.timestampSec)
                val existing = held[keyframe.id]
                val snap = existing?.snap ?: ring.remove(bits) ?: continue
                if (existing != null && !moved(existing.pose, keyframe.pose, voxel)) continue
                if (!view.meshUpsert(
                        keyframe.id,
                        keyframe.pose,
                        snap.depth,
                        snap.color,
                        snap.width,
                        snap.height,
                        snap.fx,
                        snap.fy,
                        snap.cx,
                        snap.cy,
                    )
                ) {
                    continue
                }
                held[keyframe.id] = Held(snap, keyframe.pose.copyOf())
            }
            val gone = held.keys.filter { it !in live }
            for (id in gone) {
                held.remove(id)
                view.meshErase(id)
            }
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

    private class Held(val snap: Snap, val pose: FloatArray)

    companion object {
        private const val TAG = "pocket-mesh"

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
            for (y in 0 until outH) {
                val srcY = y * height / outH
                for (x in 0 until outW) {
                    val srcX = x * width / outW
                    val depthIndex = srcY * depthStride + srcX * 2
                    val raw = if (depthIndex + 1 < depth.size) {
                        (depth[depthIndex].toInt() and 0xff) or ((depth[depthIndex + 1].toInt() and 0xff) shl 8)
                    } else {
                        0
                    }
                    val meters = if (raw == 0 || factor <= 0f) 0f else raw / factor
                    depthOut[y * outW + x] = if (meters <= 0f || meters > maxDepth) 0f else meters
                    val pair = srcY * colorStride + (srcX / 2) * 4
                    if (pair + 3 >= color.size) continue
                    val luma = if (srcX and 1 == 0) color[pair].toInt() and 0xff else color[pair + 2].toInt() and 0xff
                    val u = color[pair + 1].toInt() and 0xff
                    val v = color[pair + 3].toInt() and 0xff
                    val (red, green, blue) = yuv(luma, u, v)
                    val out = (y * outW + x) * 3
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

        private fun moved(before: FloatArray, after: FloatArray, voxel: Float): Boolean {
            val dx = before[3] - after[3]
            val dy = before[7] - after[7]
            val dz = before[11] - after[11]
            val limit = voxel * 0.5f
            if (dx * dx + dy * dy + dz * dz > limit * limit) return true
            var dot = before[0] * after[0] + before[1] * after[1] + before[2] * after[2]
            dot += before[4] * after[4] + before[5] * after[5] + before[6] * after[6]
            dot += before[8] * after[8] + before[9] * after[9] + before[10] * after[10]
            val cosine = ((dot - 1f) / 2f).coerceIn(-1f, 1f)
            return acos(cosine) > 2f * Math.PI.toFloat() / 180f
        }
    }
}
