package com.pocketscanner.capture

import android.content.Context
import android.system.ErrnoException
import android.system.Os
import android.util.Log
import java.io.File
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicReference

data class SlamCamera(
    val w: Int,
    val h: Int,
    val fps: Int,
    val fx: Double,
    val fy: Double,
    val cx: Double,
    val cy: Double,
    val k1: Double,
    val k2: Double,
    val p1: Double,
    val p2: Double,
    val k3: Double,
    val depthFactor: Double,
    val note: String,
)

private class SlamFrame(
    val color: ByteArray,
    val width: Int,
    val height: Int,
    val stride: Int,
    val depth: ByteArray,
    val depthW: Int,
    val depthH: Int,
    val depthStride: Int,
    val timestampSec: Double,
)

data class SlamTuning(
    val minSize: Int = 500,
    val pyramidLevels: Int = 6,
    val depthMeters: Float = 6f,
)

data class KeyframePose(
    val id: Int,
    val timestampSec: Double,
    val pose: FloatArray,
)

fun interface RefineCount {
    fun onCount(done: Int, total: Int)
}

class KeyframePoll(
    val frames: List<KeyframePose>,
    val complete: Boolean,
)

class TrackSample(
    val state: String,
    val pose: FloatArray?,
    val fx: Float,
    val fy: Float,
    val cx: Float,
    val cy: Float,
    val width: Int,
    val height: Int,
    val fps: Float,
    val frameMs: Int,
    val timestampSec: Double = 0.0,
)

/**
 * Owns the stella_vslam system. The camera thread only drops a frame into the slot.
 * feed_RGBD_frame runs here, so the USB loop is not blocked by ORB or the pose graph.
 */
class StellaTracker(
    private val context: Context,
    private val onSample: (TrackSample) -> Unit,
) {
    private val nativeLock = Any()
    private val pending = AtomicReference<SlamFrame?>(null)
    private val camera = AtomicReference<SlamCamera?>(null)
    private val generation = AtomicInteger(0)
    private val running = AtomicBoolean(true)
    private val ready = AtomicBoolean(false)
    private val tuning = AtomicReference(SlamTuning())

    @Volatile
    private var handle = 0L
    private val thread = Thread(::loop, "stella-track")

    init {
        thread.start()
    }

    fun wantsFrame(): Boolean = ready.get() && pending.get() == null

    fun updateTuning(next: SlamTuning) {
        tuning.set(next)
        generation.incrementAndGet()
        ready.set(false)
        pending.set(null)
    }

    fun copyKeyframes(): KeyframePoll {
        val native = handle
        if (native == 0L) return KeyframePoll(emptyList(), true)
        val ids = IntArray(2048)
        val times = DoubleArray(2048)
        val poses = FloatArray(2048 * 16)
        val count = synchronized(nativeLock) {
            if (handle == 0L) 0 else nativeCopyKeyframes(handle, ids, times, poses)
        }
        val frames = List(count) { index ->
            KeyframePose(
                id = ids[index],
                timestampSec = times[index],
                pose = poses.copyOfRange(index * 16, index * 16 + 16),
            )
        }.filter { it.id >= 0 }
        return KeyframePoll(frames, count < ids.size)
    }

    fun configure(next: SlamCamera?) {
        camera.set(next)
        generation.incrementAndGet()
        ready.set(false)
        pending.set(null)
        if (next == null) {
            report("off")
        }
    }

    fun offer(
        color: ByteArray,
        width: Int,
        height: Int,
        stride: Int,
        depth: ByteArray,
        depthW: Int,
        depthH: Int,
        depthStride: Int,
        timestampSec: Double,
    ) {
        pending.set(
            SlamFrame(color, width, height, stride, depth, depthW, depthH, depthStride, timestampSec),
        )
    }

    fun close() {
        running.set(false)
        ready.set(false)
        thread.join(5000)
    }

    private fun loop() {
        val loaded = try {
            configureOpenMp()
            System.loadLibrary("pocket_slam")
            true
        } catch (error: UnsatisfiedLinkError) {
            Log.e(TAG, "load pocket_slam", error)
            report("Tracker failed to load")
            false
        }
        var seen = generation.get()
        while (running.get()) {
            val gen = generation.get()
            val model = camera.get()
            if (!loaded || model == null) {
                if (gen != seen) {
                    stopNative()
                    seen = gen
                }
                sleep()
                continue
            }
            if (gen != seen || handle == 0L) {
                stopNative()
                if (!startNative(model)) {
                    seen = gen
                    try {
                        Thread.sleep(2000)
                    } catch (_: InterruptedException) {
                        Thread.currentThread().interrupt()
                    }
                    continue
                }
                seen = generation.get()
                if (seen != gen) {
                    continue
                }
                ready.set(true)
            }
            val frame = pending.getAndSet(null)
            if (frame == null) {
                sleep()
                continue
            }
            feed(frame)
        }
        stopNative()
    }

    private fun startNative(model: SlamCamera): Boolean {
        report("Loading vocabulary")
        return try {
            val vocab = vocabFile()
            val config = writeConfig(model)
            handle = nativeStart(vocab.absolutePath, config.absolutePath)
            trackFrames = 0
            trackWindowStart = System.nanoTime()
            shownFps = 0f
            Log.i(TAG, "started ${model.w}x${model.h} factor=${model.depthFactor} ${model.note}")
            report("Move the camera to start")
            handle != 0L
        } catch (error: Exception) {
            Log.e(TAG, "start", error)
            handle = 0L
            report(error.message?.lineSequence()?.firstOrNull() ?: "Tracker failed to start")
            false
        }
    }

    private val poseBuf = FloatArray(17)
    private var shownFps = 0f
    private var trackFrames = 0
    private var trackWindowStart = System.nanoTime()

    fun freezeMapping() {
        synchronized(nativeLock) {
            if (handle != 0L) nativeFreeze(handle)
        }
    }

    fun localBundleAdjust(tick: RefineCount) {
        synchronized(nativeLock) {
            if (handle != 0L) nativeLocalBa(handle, tick)
        }
    }

    fun searchLoops(tick: RefineCount): Int {
        return synchronized(nativeLock) {
            if (handle == 0L) 0 else nativeSearchLoops(handle, tick)
        }
    }

    fun resumeMapping() {
        synchronized(nativeLock) {
            if (handle != 0L) nativeResume(handle)
        }
    }

    private fun feed(frame: SlamFrame) {
        val started = System.nanoTime()
        val state = try {
            synchronized(nativeLock) {
            nativeFeed(
                handle,
                frame.color,
                frame.width,
                frame.height,
                frame.stride,
                frame.depth,
                frame.depthW,
                frame.depthH,
                frame.depthStride,
                frame.timestampSec,
                poseBuf,
            )
            }
        } catch (error: Exception) {
            Log.e(TAG, "feed", error)
            "Frame error"
        }
        val elapsedMs = (System.nanoTime() - started) / 1_000_000.0
        trackFrames++
        val now = System.nanoTime()
        val window = now - trackWindowStart
        if (window >= 1_000_000_000L) {
            shownFps = (trackFrames * 1e9 / window).toFloat()
            trackFrames = 0
            trackWindowStart = now
            Log.i(TAG, "$state ${"%.1f".format(shownFps)} fps ${elapsedMs.toInt()} ms")
        }
        val pose = if (poseBuf[0] == 1f) poseBuf.copyOfRange(1, 17) else null
        report(state, pose, elapsedMs.toInt(), frame.timestampSec)
    }

    private fun report(state: String, pose: FloatArray? = null, frameMs: Int = 0, timestampSec: Double = 0.0) {
        val model = camera.get()
        onSample(
            TrackSample(
                state = state,
                pose = pose,
                fx = model?.fx?.toFloat() ?: 0f,
                fy = model?.fy?.toFloat() ?: 0f,
                cx = model?.cx?.toFloat() ?: 0f,
                cy = model?.cy?.toFloat() ?: 0f,
                width = model?.w ?: 0,
                height = model?.h ?: 0,
                fps = shownFps,
                frameMs = frameMs,
                timestampSec = timestampSec,
            ),
        )
    }

    private fun stopNative() {
        ready.set(false)
        synchronized(nativeLock) {
            val current = handle
            handle = 0L
            if (current != 0L) nativeStop(current)
        }
    }

    private fun vocabFile(): File {
        val out = File(context.filesDir, "orb_vocab.fbow")
        val stamp = File(context.filesDir, "orb_vocab.stamp")
        // The first launch copies the asset into filesDir. A later install must
        // replace it: the broken file differs by two bytes, so a size check keeps it.
        if (stamp.takeIf { it.isFile }?.readText()?.trim() == VOCAB_STAMP && out.length() > 1_000_000L) {
            return out
        }
        val tmp = File(context.filesDir, "orb_vocab.fbow.tmp")
        context.assets.open("orb_vocab.fbow").use { input ->
            tmp.outputStream().use { input.copyTo(it) }
        }
        if (!tmp.renameTo(out)) {
            tmp.copyTo(out, overwrite = true)
            tmp.delete()
        }
        stamp.writeText(VOCAB_STAMP)
        Log.i(TAG, "vocab replaced (${out.length()} bytes)")
        return out
    }

    private fun writeConfig(model: SlamCamera): File {
        val chosen = tuning.get()
        val file = File(context.filesDir, "d435.yaml")
        file.writeText(
            """
            Camera:
              name: "D435"
              setup: "RGBD"
              model: "perspective"
              fx: ${model.fx}
              fy: ${model.fy}
              cx: ${model.cx}
              cy: ${model.cy}
              k1: ${model.k1}
              k2: ${model.k2}
              p1: ${model.p1}
              p2: ${model.p2}
              k3: ${model.k3}
              fps: ${model.fps}.0
              cols: ${model.w}
              rows: ${model.h}
              focal_x_baseline: ${model.fx * 0.05}
              depth_threshold: ${chosen.depthMeters}
              color_order: "Gray"
            Tracking:
              margin_local_map_projection: 10.0
            Preprocessing:
              min_size: ${chosen.minSize}
              depthmap_factor: ${model.depthFactor}
            Feature:
              name: "default ORB feature extraction setting"
              scale_factor: 1.2
              num_levels: ${chosen.pyramidLevels}
              ini_fast_threshold: 20
              min_fast_threshold: 7
            Mapping:
              baseline_dist_thr: 0.05
              redundant_obs_ratio_thr: 0.9
            """.trimIndent() + "\n",
        )
        return file
    }

    private fun configureOpenMp() {
        try {
            Os.setenv("OMP_NUM_THREADS", "4", true)
            Os.setenv("OMP_WAIT_POLICY", "PASSIVE", true)
            Os.setenv("OMP_DYNAMIC", "FALSE", true)
            Os.setenv("KMP_BLOCKTIME", "0", true)
        } catch (error: ErrnoException) {
            Log.w(TAG, "openmp env: ${error.message}")
        }
    }

    private fun sleep() {
        try {
            Thread.sleep(10)
        } catch (_: InterruptedException) {
            Thread.currentThread().interrupt()
        }
    }

    private external fun nativeStart(vocab: String, config: String): Long

    private external fun nativeFeed(
        handle: Long,
        color: ByteArray,
        width: Int,
        height: Int,
        stride: Int,
        depth: ByteArray,
        depthW: Int,
        depthH: Int,
        depthStride: Int,
        timestampSec: Double,
        poseOut: FloatArray,
    ): String

    private external fun nativeFreeze(handle: Long)

    private external fun nativeLocalBa(handle: Long, tick: RefineCount)

    private external fun nativeSearchLoops(handle: Long, tick: RefineCount): Int

    private external fun nativeResume(handle: Long)

    private external fun nativeStop(handle: Long)

    private external fun nativeCopyKeyframes(handle: Long, ids: IntArray, times: DoubleArray, poses: FloatArray): Int

    companion object {
        private const val TAG = "pocket-slam"
        private const val VOCAB_STAMP = "2"
    }
}
