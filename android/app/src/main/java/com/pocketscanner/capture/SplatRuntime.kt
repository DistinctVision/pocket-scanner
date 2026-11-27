package com.pocketscanner.capture

import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.graphics.Bitmap
import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaFormat
import android.media.MediaMuxer
import android.os.Build
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.os.PowerManager
import android.util.Log
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.core.app.NotificationCompat
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.util.Locale
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.acos
import kotlin.math.ceil
import kotlin.math.cos
import kotlin.math.roundToInt
import kotlin.math.sin
import kotlin.math.sqrt

data class TrainMetric(val epoch: Int, val l1: Float, val ssim: Float, val psnr: Float, val gaussians: Int = 0, val depth: Float = 0f)

data class TrainUi(
    val status: String = "Preparing",
    val epoch: Int = 0,
    val epochs: Int = 100,
    val fraction: Float = 0f,
    val log: List<String> = emptyList(),
    val train: List<TrainMetric> = emptyList(),
    val validation: List<TrainMetric> = emptyList(),
    val image: Bitmap? = null,
    val running: Boolean = false,
    val finished: Boolean = false,
    val runDir: String = "",
)

internal object SplatRuntime {
    var ui by mutableStateOf(TrainUi())
        private set
    private val main = Handler(Looper.getMainLooper())
    private val stop = AtomicBoolean(false)
    private var worker: Thread? = null

    fun start(context: Context) {
        if (worker?.isAlive == true) return
        stop.set(false)
        ui = TrainUi(running = true, status = "Preparing")
        val app = context.applicationContext
        worker = Thread({ run(app) }, "splat-train").also { it.start() }
    }

    fun requestStop() {
        stop.set(true)
    }

    fun running(): Boolean = worker != null && !ui.finished

    private fun run(context: Context) {
        val session = CaptureRuntime.session
        val preview = VkPreview.current()
        if (session == null || preview == null) {
            fail(context, "Training could not start")
            return
        }
        val stamp = java.text.SimpleDateFormat("yyyyMMdd-HHmmss", Locale.US).format(java.util.Date())
        val dir = File(context.getExternalFilesDir(null), "splat-$stamp")
        dir.mkdirs()
        val settings = session.scanSettings()
        publish { it.copy(status = "Reading the mesh", epochs = settings.epochs, running = true, runDir = dir.absolutePath) }
        val prepared = try {
            session.beginSplatting(File(dir, "views"), settings)
        } catch (error: Throwable) {
            android.util.Log.e("pocket-splat", "start", error)
            session.resumeAfterSplat()
            fail(context, error.message?.lineSequence()?.firstOrNull() ?: "Training could not start")
            return
        }
        val frames = prepared.frames
        val width = frames.first().width
        val height = frames.first().height
        val usable = frames.filter { it.width == width && it.height == height }
        val step = (100.0 / settings.validationPercent).roundToInt().coerceAtLeast(1)
        val validation = if (settings.validation) {
            usable.filterIndexed { index, _ -> index % step == 0 }
        } else {
            emptyList()
        }
        val validationIds = validation.mapTo(HashSet()) { it.id }
        val sourceTrain = usable.filter { it.id !in validationIds }
        val selection = if (settings.filterRedundantFrames && sourceTrain.isNotEmpty()) {
            selectFrames(sourceTrain, prepared.coverage)
        } else {
            null
        }
        val train = selection?.train ?: sourceTrain
        if (train.isEmpty() && validation.isEmpty()) {
            session.resumeAfterSplat()
            fail(context, "No frames to train")
            return
        }
        val every = settings.validationEvery.coerceAtLeast(1)
        val valPasses = if (validation.isEmpty()) 0 else settings.epochs / every
        val meanSteps = (settings.epochs * train.size).coerceAtLeast(1)
        val gridSteps = (meanSteps + valPasses * validation.size).coerceAtLeast(1)
        writeRun(dir, settings, meanSteps, gridSteps, sourceTrain.size, train, validation, selection)
        val droppedIds = selection?.droppedIds.orEmpty().toHashSet()
        val viewIndex = usable.filter { it.id !in droppedIds }.mapIndexed { index, frame -> frame.id to index }.toMap()
        preview.splatDensity(settings.densifyInterval, settings.gradThreshold)
        val seeded = preview.splatInitFromSeed(viewIndex.size, width, height, settings.gaussBudget, meanSteps, gridSteps)
        if (seeded) writeHyper(dir, settings, preview.splatHyperJson())
        if (!seeded) {
            session.resumeAfterSplat()
            fail(context, "The GPU trainer did not start")
            return
        }
        session.releaseTrainingSources()
        TrainService.start(context)
        val metrics = File(dir, "metrics.jsonl")
        val valDir = File(dir, "val")
        valDir.mkdirs()
        val trainSeries = ArrayList<TrainMetric>()
        val valSeries = ArrayList<TrainMetric>()
        val video = File(dir, "video.mp4")
        var finishedEpochs = 0
        var failed = false
        for (epoch in 1..settings.epochs) {
            if (stop.get()) break
            if (train.isNotEmpty()) {
                publish { it.copy(fraction = 0f, status = "Epoch $epoch / ${settings.epochs}, train") }
                val trainMean = pass(context, preview, train, viewIndex, width, height, epoch, settings.epochs, "train", updateGauss = true, metricsAfter = false, pngDir = null)
                if (trainMean == null) {
                    failed = !stop.get()
                    break
                }
                val counted = trainMean.copy(gaussians = preview.splatGaussianCount())
                trainSeries.add(counted)
                append(metrics, epoch, "train", counted)
                note("Epoch $epoch train  L1 ${fmt(trainMean.l1)}  SSIM ${fmt(trainMean.ssim)}  PSNR ${fmt(trainMean.psnr)}  depth ${fmt(trainMean.depth)}")
                publish { it.copy(epoch = epoch, train = trainSeries.toList(), fraction = 1f) }
            }
            val validateNow = validation.isNotEmpty() && epoch % every == 0
            if (validateNow) {
                if (stop.get()) break
                publish { it.copy(fraction = 0f, status = "Epoch $epoch / ${settings.epochs}, validation") }
                val valMean = pass(context, preview, validation, viewIndex, width, height, epoch, settings.epochs, "validation", updateGauss = false, metricsAfter = true, pngDir = valDir)
                if (valMean == null) {
                    failed = !stop.get()
                    break
                }
                valSeries.add(valMean)
                append(metrics, epoch, "val", valMean)
                note("Epoch $epoch val  L1 ${fmt(valMean.l1)}  SSIM ${fmt(valMean.ssim)}  PSNR ${fmt(valMean.psnr)}  depth ${fmt(valMean.depth)}")
                publish { it.copy(validation = valSeries.toList(), image = latestImage(valDir), fraction = 0f, status = "Rendering video") }
                if (!stop.get()) encodeVideo(context, preview, usable, width, height, video, epoch, settings.epochs, settings.videoSeconds)
            }
            finishedEpochs = epoch
            TrainService.update(context, epoch, settings.epochs, 1f)
        }
        if (!failed && !settings.validation && finishedEpochs > 0 && !stop.get()) {
            publish { it.copy(fraction = 0f, status = "Rendering video") }
            encodeVideo(context, preview, usable, width, height, video, finishedEpochs, settings.epochs, settings.videoSeconds)
        }
        if (failed) {
            TrainService.stop(context)
            session.resumeAfterSplat()
            publish { it.copy(running = false, finished = false) }
            worker = null
            return
        }
        val early = stop.get() || finishedEpochs < settings.epochs
        publish { it.copy(status = "Writing the scene") }
        preview.splatWritePly(File(dir, "scene.ply").absolutePath)
        TrainService.stop(context)
        session.resumeAfterSplat()
        publish {
            it.copy(
                status = if (early) "Stopped" else "Done",
                running = false,
                finished = true,
                fraction = 1f,
            )
        }
        worker = null
    }

    private fun selectFrames(train: List<TrainFrame>, coverage: FloatArray?): SplatFrameSelector.Selection? {
        if (stop.get()) return null
        if (coverage == null || coverage.isEmpty()) {
            note("Mesh coverage unavailable. Keeping all ${train.size} train views.")
            return null
        }
        publish { it.copy(status = "Checking frame coverage") }
        val selection = try {
            SplatFrameSelector.select(train, coverage, stop::get)
        } catch (error: IOException) {
            Log.w("pocket-splat", "Frame selection could not read training data", error)
            null
        } catch (error: OutOfMemoryError) {
            Log.w("pocket-splat", "Frame selection exceeded available memory", error)
            null
        }
        if (selection != null) {
            note("Kept ${selection.train.size} of ${train.size} train views")
        } else if (!stop.get()) {
            note("Frame selection unavailable. Keeping all ${train.size} train views.")
        }
        return selection
    }

    private fun pass(
        context: Context,
        preview: VkPreview,
        frames: List<TrainFrame>,
        viewIndex: Map<Int, Int>,
        width: Int,
        height: Int,
        epoch: Int,
        epochs: Int,
        passName: String,
        updateGauss: Boolean,
        metricsAfter: Boolean,
        pngDir: File?,
    ): TrainMetric? {
        if (frames.isEmpty()) return null
        var l1 = 0.0
        var ssim = 0.0
        var psnr = 0.0
        var depthLoss = 0.0
        val rgb = FloatArray(width * height * 3)
        val depth = FloatArray(width * height)
        frames.forEachIndexed { index, frame ->
            if (stop.get()) return null
            val bytes = frame.rgb.readBytes()
            val n = minOf(rgb.size, bytes.size)
            for (i in 0 until n) rgb[i] = (bytes[i].toInt() and 255) / 255f
            depth.fill(0f)
            if (frame.depth.isFile) {
                val raw = frame.depth.readBytes()
                val view = java.nio.ByteBuffer.wrap(raw).order(java.nio.ByteOrder.LITTLE_ENDIAN)
                val samples = minOf(depth.size, raw.size / 4)
                for (i in 0 until samples) depth[i] = view.float
            }
            val metrics = preview.splatStep(
                rgb, depth, frame.pose, frame.fx, frame.fy, frame.cx, frame.cy,
                viewIndex[frame.id] ?: 0, updateGauss, true, metricsAfter,
            )
            if (metrics == null) {
                if (!stop.get()) {
                    val reason = preview.splatError().ifBlank { "The rasterizer stopped" }
                    publish { it.copy(status = reason) }
                    note(reason)
                }
                return null
            }
            l1 += metrics[0]
            ssim += metrics[1]
            psnr += metrics[2]
            depthLoss += metrics[3]
            if (pngDir != null) {
                val rendered = preview.splatRender(frame.pose, frame.fx, frame.fy, frame.cx, frame.cy, corrected = true)
                val image = rendered?.let { bitmapOf(it, width, height) }
                if (image != null) {
                    FileOutputStream(File(pngDir, "${frame.id}.png")).use { image.compress(Bitmap.CompressFormat.PNG, 100, it) }
                }
            }
            val fraction = (index + 1f) / frames.size
            publish { it.copy(epoch = epoch, fraction = fraction, status = "Epoch $epoch / $epochs, $passName") }
            TrainService.update(context, epoch, epochs, fraction)
        }
        val count = frames.size.toDouble()
        return TrainMetric(epoch, (l1 / count).toFloat(), (ssim / count).toFloat(), (psnr / count).toFloat(), depth = (depthLoss / count).toFloat())
    }

    private fun encodeVideo(
        context: Context,
        preview: VkPreview,
        frames: List<TrainFrame>,
        width: Int,
        height: Int,
        dest: File,
        epoch: Int,
        epochs: Int,
        seconds: Int,
    ) {
        if (frames.size < 2) return
        val poses = ArrayList<FloatArray>()
        val lenses = ArrayList<TrainFrame>()
        for (i in 0 until frames.lastIndex) {
            val a = frames[i]
            val b = frames[i + 1]
            val pieces = kotlin.math.max(1, kotlin.math.max(ceil(distance(a.pose, b.pose) / 0.02f).toInt(), ceil(angleDeg(a.pose, b.pose) / 2f).toInt()))
            for (step in 0 until pieces) {
                val t = step.toFloat() / pieces
                poses.add(blendPose(a.pose, b.pose, t))
                lenses.add(a)
            }
        }
        poses.add(frames.last().pose.copyOf())
        lenses.add(frames.last())
        val keep = seconds.coerceIn(1, 30) * 30
        val start = if (poses.size <= keep) 0 else (poses.size - keep) / 2
        val end = minOf(start + keep, poses.size)
        val tmp = File(dest.parentFile, "${dest.name}.tmp")
        tmp.delete()
        var finished = false
        try {
            VideoEncoder(width, height, tmp).use { encoder ->
                for (index in start until end) {
                    if (stop.get()) return@use
                    val lens = lenses[index]
                    val rgb = preview.splatRender(poses[index], lens.fx, lens.fy, lens.cx, lens.cy, corrected = false) ?: continue
                    encoder.write(rgb)
                    val done = index - start + 1
                    val fraction = done.toFloat() / (end - start)
                    publish { it.copy(status = "Rendering video", fraction = fraction) }
                    TrainService.update(context, epoch, epochs, fraction)
                }
                finished = !stop.get()
            }
        } catch (error: Exception) {
            note("Video: ${error.message ?: "encoder failed"}")
            tmp.delete()
            return
        }
        if (!finished) {
            tmp.delete()
            return
        }
        dest.delete()
        if (!tmp.renameTo(dest)) {
            tmp.copyTo(dest, overwrite = true)
            tmp.delete()
        }
    }

    private fun fail(context: Context, message: String) {
        TrainService.stop(context)
        publish { it.copy(status = message, running = false, finished = false) }
        worker = null
    }

    private fun publish(block: (TrainUi) -> TrainUi) {
        main.post { ui = block(ui) }
    }

    private fun note(line: String) {
        publish { it.copy(log = (it.log + line).takeLast(12)) }
    }

    private fun writeRun(
        dir: File,
        settings: ScanSettings,
        meanSteps: Int,
        gridSteps: Int,
        sourceTrainCount: Int,
        train: List<TrainFrame>,
        validation: List<TrainFrame>,
        selection: SplatFrameSelector.Selection?,
    ) {
        val filter = JSONObject()
            .put("enabled", settings.filterRedundantFrames)
            .put("sourceTrainCount", sourceTrainCount)
        if (selection != null) {
            filter.put("surfaceSamples", selection.surfaceSamples)
            filter.put("coveredSamples", selection.coveredSamples)
            filter.put("parallaxSamples", selection.parallaxSamples)
            filter.put("droppedTrainIds", JSONArray(selection.droppedIds))
            filter.put("unverifiedTrainIds", JSONArray(selection.unverifiedIds))
        }
        val run = JSONObject()
            .put("epochs", settings.epochs)
            .put("validation", settings.validation)
            .put("percent", settings.validationPercent)
            .put("every", settings.validationEvery)
            .put("budget", settings.gaussBudget)
            .put("videoSeconds", settings.videoSeconds)
            .put("meanSteps", meanSteps)
            .put("gridSteps", gridSteps)
            .put("trainIds", JSONArray(train.map { it.id }))
            .put("valIds", JSONArray(validation.map { it.id }))
            .put("frameSelection", filter)
        File(dir, "run.json").writeText(run.toString())
    }

    private fun writeHyper(dir: File, settings: ScanSettings, training: String) {
        val json = runCatching { JSONObject(training) }.getOrElse { JSONObject() }
        json.put("epochs", settings.epochs)
        json.put("validation", settings.validation)
        json.put("validationPercent", settings.validationPercent)
        json.put("validationEvery", settings.validationEvery)
        json.put("videoSeconds", settings.videoSeconds)
        File(dir, "hyperparameters.json").writeText(json.toString(2))
    }

    private fun append(file: File, epoch: Int, split: String, metric: TrainMetric) {
        file.appendText(
            """{"epoch":$epoch,"split":"$split","l1":${metric.l1},"ssim":${metric.ssim},"psnr":${metric.psnr},"depth":${metric.depth},"gaussians":${metric.gaussians}}""" + "\n",
        )
    }

    private fun fmt(value: Float) = String.format(Locale.US, "%.3f", value)

    fun runs(context: Context): List<File> {
            val root = context.getExternalFilesDir(null) ?: return emptyList()
            return root.listFiles { file -> file.isDirectory && file.name.startsWith("splat-") }
                ?.sortedByDescending { it.name }
                ?: emptyList()
        }

    fun readMetrics(dir: File): Pair<List<TrainMetric>, List<TrainMetric>> {
            val file = File(dir, "metrics.jsonl")
            if (!file.isFile) return emptyList<TrainMetric>() to emptyList()
            val train = ArrayList<TrainMetric>()
            val validation = ArrayList<TrainMetric>()
            file.forEachLine { line ->
                val epoch = number(line, "epoch")?.toInt() ?: return@forEachLine
                val metric = TrainMetric(
                    epoch,
                    number(line, "l1") ?: return@forEachLine,
                    number(line, "ssim") ?: return@forEachLine,
                    number(line, "psnr") ?: return@forEachLine,
                    number(line, "gaussians")?.toInt() ?: 0,
                    number(line, "depth") ?: 0f,
                )
                if (line.contains("\"val\"")) validation.add(metric) else train.add(metric)
            }
            return train to validation
        }

    private fun number(line: String, key: String): Float? {
            val token = "\"$key\":"
            val at = line.indexOf(token)
            if (at < 0) return null
            val start = at + token.length
            val end = line.indexOfAny(charArrayOf(',', '}'), start)
            if (end < 0) return null
            return line.substring(start, end).toFloatOrNull()
        }
}

private fun latestImage(dir: File): Bitmap? {
    val file = dir.listFiles { item -> item.extension == "png" }?.maxByOrNull { it.lastModified() } ?: return null
    return BitmapFactorySafe.decode(file)
}

private object BitmapFactorySafe {
    fun decode(file: File): Bitmap? = android.graphics.BitmapFactory.decodeFile(file.absolutePath)
}

private fun bitmapOf(rgb: ByteArray, width: Int, height: Int): Bitmap? {
    if (rgb.size < width * height * 3) return null
    val pixels = IntArray(width * height)
    for (i in pixels.indices) {
        val r = rgb[i * 3].toInt() and 255
        val g = rgb[i * 3 + 1].toInt() and 255
        val b = rgb[i * 3 + 2].toInt() and 255
        pixels[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
    }
    return Bitmap.createBitmap(pixels, width, height, Bitmap.Config.ARGB_8888)
}

private fun distance(a: FloatArray, b: FloatArray): Float {
    val dx = a[3] - b[3]
    val dy = a[7] - b[7]
    val dz = a[11] - b[11]
    return sqrt(dx * dx + dy * dy + dz * dz)
}

private fun angleDeg(a: FloatArray, b: FloatArray): Float {
    val qa = quatOf(a)
    val qb = quatOf(b)
    var dot = qa[0] * qb[0] + qa[1] * qb[1] + qa[2] * qb[2] + qa[3] * qb[3]
    dot = kotlin.math.abs(dot).coerceIn(0f, 1f)
    return Math.toDegrees(2.0 * acos(dot.toDouble())).toFloat()
}

private fun quatOf(m: FloatArray): FloatArray {
    val r00 = m[0]
    val r01 = m[1]
    val r02 = m[2]
    val r10 = m[4]
    val r11 = m[5]
    val r12 = m[6]
    val r20 = m[8]
    val r21 = m[9]
    val r22 = m[10]
    val trace = r00 + r11 + r22
    val q = FloatArray(4)
    if (trace > 0f) {
        val s = sqrt(trace + 1f) * 2f
        q[3] = 0.25f * s
        q[0] = (r21 - r12) / s
        q[1] = (r02 - r20) / s
        q[2] = (r10 - r01) / s
    } else if (r00 > r11 && r00 > r22) {
        val s = sqrt(1f + r00 - r11 - r22) * 2f
        q[3] = (r21 - r12) / s
        q[0] = 0.25f * s
        q[1] = (r01 + r10) / s
        q[2] = (r02 + r20) / s
    } else if (r11 > r22) {
        val s = sqrt(1f + r11 - r00 - r22) * 2f
        q[3] = (r02 - r20) / s
        q[0] = (r01 + r10) / s
        q[1] = 0.25f * s
        q[2] = (r12 + r21) / s
    } else {
        val s = sqrt(1f + r22 - r00 - r11) * 2f
        q[3] = (r10 - r01) / s
        q[0] = (r02 + r20) / s
        q[1] = (r12 + r21) / s
        q[2] = 0.25f * s
    }
    return q
}

private fun blendPose(a: FloatArray, b: FloatArray, t: Float): FloatArray {
    val qa = quatOf(a)
    val qb = quatOf(b)
    var dot = qa[0] * qb[0] + qa[1] * qb[1] + qa[2] * qb[2] + qa[3] * qb[3]
    val other = qb.copyOf()
    if (dot < 0f) {
        dot = -dot
        for (i in other.indices) other[i] = -other[i]
    }
    val q = if (dot > 0.9995f) {
        FloatArray(4) { i -> qa[i] + t * (other[i] - qa[i]) }
    } else {
        val theta = acos(dot.coerceIn(-1f, 1f))
        val s = sin(theta)
        val w1 = sin((1f - t) * theta) / s
        val w2 = sin(t * theta) / s
        FloatArray(4) { i -> w1 * qa[i] + w2 * other[i] }
    }
    val len = sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).coerceAtLeast(1e-8f)
    for (i in q.indices) q[i] /= len
    val x = q[0]
    val y = q[1]
    val z = q[2]
    val w = q[3]
    val out = FloatArray(16)
    out[0] = 1f - 2f * (y * y + z * z)
    out[1] = 2f * (x * y - z * w)
    out[2] = 2f * (x * z + y * w)
    out[4] = 2f * (x * y + z * w)
    out[5] = 1f - 2f * (x * x + z * z)
    out[6] = 2f * (y * z - x * w)
    out[8] = 2f * (x * z - y * w)
    out[9] = 2f * (y * z + x * w)
    out[10] = 1f - 2f * (x * x + y * y)
    out[3] = a[3] + (b[3] - a[3]) * t
    out[7] = a[7] + (b[7] - a[7]) * t
    out[11] = a[11] + (b[11] - a[11]) * t
    out[15] = 1f
    return out
}

private class VideoEncoder(private val width: Int, private val height: Int, dest: File) : AutoCloseable {
    private val codec = MediaCodec.createEncoderByType(MediaFormat.MIMETYPE_VIDEO_AVC)
    private val muxer = MediaMuxer(dest.absolutePath, MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4)
    private var track = -1
    private var started = false
    private var frame = 0L

    init {
        val format = MediaFormat.createVideoFormat(MediaFormat.MIMETYPE_VIDEO_AVC, width, height)
        format.setInteger(MediaFormat.KEY_COLOR_FORMAT, MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Flexible)
        format.setInteger(MediaFormat.KEY_BIT_RATE, 4_000_000)
        format.setInteger(MediaFormat.KEY_FRAME_RATE, 30)
        format.setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, 1)
        codec.configure(format, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE)
        codec.start()
    }

    fun write(rgb: ByteArray) {
        val index = codec.dequeueInputBuffer(10_000)
        if (index < 0) return
        val image = codec.getInputImage(index) ?: return
        val y = image.planes[0]
        val u = image.planes[1]
        val v = image.planes[2]
        for (row in 0 until height) {
            for (col in 0 until width) {
                val r = rgb[(row * width + col) * 3].toInt() and 255
                val g = rgb[(row * width + col) * 3 + 1].toInt() and 255
                val b = rgb[(row * width + col) * 3 + 2].toInt() and 255
                val yy = ((66 * r + 129 * g + 25 * b + 128) shr 8) + 16
                y.buffer.put(row * y.rowStride + col * y.pixelStride, yy.coerceIn(0, 255).toByte())
                if (row % 2 == 0 && col % 2 == 0) {
                    val uu = ((-38 * r - 74 * g + 112 * b + 128) shr 8) + 128
                    val vv = ((112 * r - 94 * g - 18 * b + 128) shr 8) + 128
                    val uvRow = row / 2
                    val uvCol = col / 2
                    u.buffer.put(uvRow * u.rowStride + uvCol * u.pixelStride, uu.coerceIn(0, 255).toByte())
                    v.buffer.put(uvRow * v.rowStride + uvCol * v.pixelStride, vv.coerceIn(0, 255).toByte())
                }
            }
        }
        codec.queueInputBuffer(index, 0, image.width * image.height * 3 / 2, frame * 33333L, 0)
        frame++
        drain(false)
    }

    override fun close() {
        val index = codec.dequeueInputBuffer(10_000)
        if (index >= 0) codec.queueInputBuffer(index, 0, 0, frame * 33333L, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
        drain(true)
        codec.stop()
        codec.release()
        if (started) muxer.stop()
        muxer.release()
    }

    private fun drain(end: Boolean) {
        val info = MediaCodec.BufferInfo()
        while (true) {
            val index = codec.dequeueOutputBuffer(info, if (end) 10_000 else 0)
            when {
                index == MediaCodec.INFO_TRY_AGAIN_LATER -> if (!end) return
                index == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {
                    track = muxer.addTrack(codec.outputFormat)
                    muxer.start()
                    started = true
                }
                index >= 0 -> {
                    val buffer = codec.getOutputBuffer(index)
                    if (buffer != null && info.size > 0 && started) {
                        buffer.position(info.offset)
                        buffer.limit(info.offset + info.size)
                        muxer.writeSampleData(track, buffer, info)
                    }
                    codec.releaseOutputBuffer(index, false)
                    if (info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) return
                }
            }
        }
    }
}

class TrainService : Service() {
    private var wake: PowerManager.WakeLock? = null

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        val notification = TrainNotes.build(this, 0, 100, 0f)
        if (Build.VERSION.SDK_INT >= 29) {
            startForeground(TrainNotes.ID, notification, ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC)
        } else {
            startForeground(TrainNotes.ID, notification)
        }
        if (wake == null) {
            val power = getSystemService(PowerManager::class.java)
            @Suppress("DEPRECATION")
            val level = PowerManager.SCREEN_BRIGHT_WAKE_LOCK or
                PowerManager.ACQUIRE_CAUSES_WAKEUP or
                PowerManager.ON_AFTER_RELEASE
            wake = power.newWakeLock(level, "pocketscanner:train").apply {
                setReferenceCounted(false)
                acquire()
            }
        }
        return START_NOT_STICKY
    }

    override fun onDestroy() {
        wake?.let { if (it.isHeld) it.release() }
        wake = null
        super.onDestroy()
    }

    companion object {
        fun start(context: Context) {
            val intent = Intent(context, TrainService::class.java)
            if (Build.VERSION.SDK_INT >= 26) context.startForegroundService(intent) else context.startService(intent)
        }

        fun update(context: Context?, epoch: Int, epochs: Int, fraction: Float) {
            if (context == null) return
            val manager = context.getSystemService(NotificationManager::class.java)
            manager.notify(TrainNotes.ID, TrainNotes.build(context, epoch, epochs, fraction))
        }

        fun stop(context: Context) {
            context.stopService(Intent(context, TrainService::class.java))
        }
    }
}

private object TrainNotes {
    const val ID = 42
    private const val CHANNEL = "splat-train"

    fun build(context: Context, epoch: Int, epochs: Int, fraction: Float): android.app.Notification {
        val manager = context.getSystemService(NotificationManager::class.java)
        if (Build.VERSION.SDK_INT >= 26 && manager.getNotificationChannel(CHANNEL) == null) {
            manager.createNotificationChannel(
                NotificationChannel(CHANNEL, "Splatting", NotificationManager.IMPORTANCE_LOW),
            )
        }
        val open = PendingIntent.getActivity(
            context,
            0,
            Intent(context, TrainActivity::class.java).addFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP),
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
        )
        return NotificationCompat.Builder(context, CHANNEL)
            .setSmallIcon(android.R.drawable.ic_menu_camera)
            .setContentTitle("Epoch $epoch / $epochs")
            .setContentText("Training")
            .setProgress(100, (fraction * 100f).roundToInt().coerceIn(0, 100), false)
            .setOngoing(true)
            .setContentIntent(open)
            .build()
    }
}
