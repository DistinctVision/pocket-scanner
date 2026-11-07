package com.pocketscanner.capture

import android.content.ContentValues
import android.content.Context
import android.hardware.usb.UsbManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Build
import android.os.Environment
import android.os.Handler
import android.os.Looper
import android.provider.MediaStore
import android.util.Log
import com.intel.realsense.librealsense.CameraInfo
import com.intel.realsense.librealsense.Colorizer
import com.intel.realsense.librealsense.Config
import com.intel.realsense.librealsense.DeviceListener
import com.intel.realsense.librealsense.Extension
import com.intel.realsense.librealsense.FrameSet
import com.intel.realsense.librealsense.GLRsSurfaceView
import com.intel.realsense.librealsense.Pipeline
import com.intel.realsense.librealsense.RsContext
import com.intel.realsense.librealsense.StreamFormat
import com.intel.realsense.librealsense.StreamType
import com.intel.realsense.librealsense.Sensor
import com.intel.realsense.librealsense.UsbUtilities
import com.intel.realsense.librealsense.VideoFrame
import com.intel.realsense.librealsense.VideoStreamProfile
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicReference

private data class UsbStatus(val headline: String, val detail: String)

data class CaptureUi(
    val headline: String = "Подключите D435",
    val deviceLine: String = "",
    val streamLine: String = "",
    val fps: Float = 0f,
    val detachCount: Int = 0,
    val saveLine: String = "",
    val colorBitmap: Bitmap? = null,
    val profiles: List<String> = emptyList(),
    val selectedProfile: String = "",
)

private data class StreamRequest(
    val label: String,
    val depthIndex: Int,
    val depthW: Int,
    val depthH: Int,
    val depthFps: Int,
    val colorIndex: Int,
    val colorW: Int,
    val colorH: Int,
    val colorFps: Int,
    val colorFormat: StreamFormat,
    val depthEnabled: Boolean = true,
    val colorEnabled: Boolean = true,
)

private data class VideoMode(
    val type: StreamType,
    val format: StreamFormat,
    val index: Int,
    val w: Int,
    val h: Int,
    val fps: Int,
)

/**
 * One camera thread owns the pipeline. The GL view receives a colorized depth
 * frame; MJPEG is decoded to a bitmap because the stock viewer cannot draw it.
 */
class RealSenseSession(
    private val context: Context,
    private val onUi: (CaptureUi) -> Unit,
) {
    private val main = Handler(Looper.getMainLooper())
    private val alive = AtomicBoolean(true)
    private val streaming = AtomicBoolean(false)
    private val snapshot = AtomicBoolean(false)
    private val detaches = AtomicInteger(0)
    private val wantedLabel = AtomicReference<String?>(null)

    @Volatile
    private var catalog: List<StreamRequest> = emptyList()
    private var glView: GLRsSurfaceView? = null
    private var colorView: YuyvPreview? = null
    private var loggedLuma = false
    private var worker: Thread? = null

    @Volatile
    private var ui = CaptureUi()

    fun attachPreview(view: GLRsSurfaceView?) {
        glView = view
    }

    fun attachColorPreview(view: YuyvPreview?) {
        colorView = view
    }

    fun selectProfile(label: String) {
        if (catalog.none { it.label == label }) return
        wantedLabel.set(label)
        streaming.set(false)
    }

    fun requestSnapshot() {
        snapshot.set(true)
    }

    fun start() {
        if (worker != null) return
        RsContext.init(context.applicationContext)
        worker = Thread({ loop() }, "realsense-capture").also { it.start() }
    }

    fun stop() {
        alive.set(false)
        streaming.set(false)
        worker?.join(2000)
        worker = null
    }

    private fun publish(transform: (CaptureUi) -> CaptureUi) {
        ui = transform(ui)
        val snapshotUi = ui
        main.post { onUi(snapshotUi) }
    }

    private fun usbStatus(context: Context): UsbStatus {
        val manager = context.getSystemService(Context.USB_SERVICE) as UsbManager
        val devices = manager.deviceList.values.toList()
        if (devices.isEmpty()) {
            return UsbStatus(
                headline = "Телефон не видит USB",
                detail = "Камера должна сидеть в OTG-хосте телефона. " +
                    "Хаб, запитанный от компьютера, оставляет телефон устройством: он заряжается, а D435 видит компьютер, не телефон.",
            )
        }
        val lines = devices.joinToString("\n") { device ->
            val access = if (manager.hasPermission(device)) "доступ есть" else "нет доступа"
            val name = device.productName ?: device.deviceName
            "%04X:%04X %s (%s)".format(device.vendorId, device.productId, name, access)
        }
        val intel = devices.filter { it.vendorId == 0x8086 }
        val headline = when {
            intel.isEmpty() -> "USB есть, RealSense нет"
            intel.any { manager.hasPermission(it) } -> "RealSense в USB, открываю"
            else -> "Нужен доступ к USB"
        }
        return UsbStatus(headline, lines)
    }

    private fun loop() {
        val rs = RsContext()
        val listener = object : DeviceListener {
            override fun onDeviceAttach() {
                streaming.set(true)
                publish { it.copy(headline = "Камера подключена") }
            }

            override fun onDeviceDetach() {
                streaming.set(false)
                val n = detaches.incrementAndGet()
                publish {
                    it.copy(
                        headline = "Камера отключена",
                        detachCount = n,
                        fps = 0f,
                        colorBitmap = null,
                    )
                }
            }
        }
        rs.setDevicesChangedCallback(listener)
        try {
            rs.queryDevices().use { list ->
                if (list.deviceCount > 0) {
                    streaming.set(true)
                }
            }
            while (alive.get()) {
                if (!streaming.get()) {
                    UsbUtilities.grantUsbPermissionIfNeeded(context)
                    val present = rs.queryDevices().use { it.deviceCount > 0 }
                    if (present) {
                        streaming.set(true)
                        continue
                    }
                    val usb = usbStatus(context)
                    publish { it.copy(headline = usb.headline, deviceLine = usb.detail) }
                    Thread.sleep(1000)
                    continue
                }
                try {
                    streamUntilStopped(rs)
                } catch (e: Exception) {
                    Log.e(TAG, "stream failed", e)
                    val line = (e.message ?: "Ошибка потока").lineSequence().first()
                    publish { it.copy(headline = line) }
                    streaming.set(false)
                    Thread.sleep(2000)
                }
            }
        } finally {
            rs.close()
        }
    }

    private fun streamUntilStopped(rs: RsContext) {
        val pipeline = Pipeline()
        val colorizer = Colorizer()
        try {
            val opened = openPipeline(rs, pipeline, colorizer)
            publish {
                it.copy(
                    headline = "Идёт поток",
                    deviceLine = opened.deviceLine,
                    streamLine = opened.streamLine,
                )
            }
            var framesInWindow = 0
            var windowStart = System.nanoTime()
            while (alive.get() && streaming.get()) {
                try {
                    val frames = pipeline.waitForFrames(2000)
                    frames.use { set -> consume(set, colorizer) }
                    framesInWindow++
                } catch (e: Exception) {
                    val message = e.message.orEmpty()
                    if (message.contains("arrive", ignoreCase = true) ||
                        message.contains("timeout", ignoreCase = true)
                    ) {
                        continue
                    }
                    throw e
                }
                val now = System.nanoTime()
                if (now - windowStart >= 1_000_000_000L) {
                    val fps = framesInWindow * 1e9f / (now - windowStart)
                    framesInWindow = 0
                    windowStart = now
                    publish { it.copy(fps = fps, detachCount = detaches.get()) }
                }
            }
        } finally {
            try {
                pipeline.stop()
            } catch (_: Exception) {
            }
            colorizer.close()
            pipeline.close()
        }
    }

    private data class Opened(val deviceLine: String, val streamLine: String)

    private fun resetDepthPreview() {
        val view = glView ?: return
        view.queueEvent { view.clear() }
    }

    private fun openPipeline(rs: RsContext, pipeline: Pipeline, colorizer: Colorizer): Opened {
        resetDepthPreview()
        val probe = probe(rs)
        val options = catalog(pipeline, probe.modes)
        catalog = options
        Log.i(TAG, "usb=${probe.usb} profiles=${options.joinToString(" | ") { it.label }}")
        if (options.isEmpty()) {
            throw IllegalStateException("Камера не отдала ни одной пары depth и color")
        }
        loggedLuma = false
        val request = pick(options)
        publish {
            it.copy(
                headline = "Открываю ${request.label}",
                profiles = options.map { option -> option.label },
                selectedProfile = request.label,
            )
        }
        try {
            val opened = start(pipeline, request)
            Log.i(TAG, "started ${request.label}")
            val frames = pipeline.waitForFrames(4000)
            frames.use { consume(it, colorizer) }
            Log.i(TAG, "first frame ${request.label}")
            return opened
        } catch (e: Exception) {
            Log.w(TAG, "profile ${request.label} failed: ${e.message}")
            try {
                pipeline.stop()
            } catch (_: Exception) {
            }
            throw IllegalStateException("${request.label}: ${e.message ?: "нет кадров"}")
        }
    }

    private fun pick(options: List<StreamRequest>): StreamRequest {
        val want = wantedLabel.get()
        return options.find { it.label == want } ?: options.first()
    }

    private data class Probe(val modes: List<VideoMode>, val usb: String)

    private fun probe(rs: RsContext): Probe {
        val modes = mutableListOf<VideoMode>()
        var usb = ""
        rs.queryDevices().use { list ->
            if (list.deviceCount == 0) return Probe(modes, usb)
            val device = list.createDevice(0) ?: return Probe(modes, usb)
            device.use {
                if (device.supportsInfo(CameraInfo.USB_TYPE_DESCRIPTOR)) {
                    usb = device.getInfo(CameraInfo.USB_TYPE_DESCRIPTOR)
                }
                for (sensor in device.querySensors()) {
                    modes += modesOf(sensor)
                }
            }
        }
        return Probe(modes, usb)
    }

    private fun modesOf(sensor: Sensor): List<VideoMode> {
        val modes = mutableListOf<VideoMode>()
        for (profile in sensor.streamProfiles) {
            if (!profile.`is`(Extension.VIDEO_PROFILE)) continue
            val video = profile.`as`<VideoStreamProfile>(Extension.VIDEO_PROFILE)
            modes += VideoMode(
                type = profile.type,
                format = profile.format,
                index = profile.index,
                w = video.width,
                h = video.height,
                fps = profile.frameRate,
            )
        }
        return modes
    }

    private fun catalog(pipeline: Pipeline, modes: List<VideoMode>): List<StreamRequest> {
        val depth = modes
            .filter { it.type == StreamType.DEPTH && it.format == StreamFormat.Z16 }
            .distinctBy { listOf(it.index, it.w, it.h, it.fps) }
        val color = colorModes(modes)
        val candidates = candidates(depth, color)
        val resolved = candidates.filter { resolves(pipeline, it) }.distinctBy { it.label }
        Log.i(
            TAG,
            "modes depth=${depth.size} color=${color.size} candidates=${candidates.size} resolved=${resolved.size}",
        )
        return resolved.sortedWith(profileOrder)
    }

    private fun colorModes(modes: List<VideoMode>): List<VideoMode> {
        fun take(formats: Set<StreamFormat>) = modes
            .filter { it.type == StreamType.COLOR && it.format in formats }
            .distinctBy { listOf(it.index, it.format, it.w, it.h, it.fps) }
        val preview = take(setOf(StreamFormat.YUYV, StreamFormat.MJPEG))
        if (preview.isNotEmpty()) return preview
        return take(
            setOf(
                StreamFormat.RGB8,
                StreamFormat.BGR8,
                StreamFormat.RGBA8,
                StreamFormat.BGRA8,
                StreamFormat.UYVY,
            ),
        )
    }

    private fun candidates(depth: List<VideoMode>, color: List<VideoMode>): List<StreamRequest> {
        if (depth.isEmpty()) return color.map { requestFor(null, it) }
        val requests = mutableListOf<StreamRequest>()
        val pairedDepth = mutableSetOf<VideoMode>()
        val pairedColor = mutableSetOf<VideoMode>()
        for (d in depth) {
            for (c in color) {
                if (d.w == c.w && d.h == c.h && d.fps == c.fps) {
                    requests += requestFor(d, c)
                    pairedDepth += d
                    pairedColor += c
                }
            }
        }
        for (d in depth) {
            val nearest = color
                .filter { it.fps == d.fps && (it.w != d.w || it.h != d.h) }
                .minWithOrNull(compareBy({ pixelDistance(it, d) }, { formatRank(it.format) }))
                ?: continue
            requests += requestFor(d, nearest)
            pairedDepth += d
            pairedColor += nearest
        }
        for (c in color) {
            if (c in pairedColor) continue
            val nearest = depth
                .filter { it.fps == c.fps }
                .minByOrNull { pixelDistance(it, c) }
                ?: continue
            requests += requestFor(nearest, c)
            pairedDepth += nearest
        }
        for (d in depth) {
            if (d !in pairedDepth) requests += requestFor(d, null)
        }
        return requests
    }

    private fun pixelDistance(a: VideoMode, b: VideoMode): Int = kotlin.math.abs(a.w * a.h - b.w * b.h)

    private fun formatRank(format: StreamFormat): Int = when (format) {
        StreamFormat.YUYV -> 0
        StreamFormat.MJPEG -> 1
        else -> 2
    }

    private val profileOrder = compareBy<StreamRequest>(
        { if (!it.colorEnabled || !it.depthEnabled) 1 else 0 },
        { if (it.depthW == it.colorW && it.depthH == it.colorH && it.depthFps == it.colorFps) 0 else 1 },
        { formatRank(it.colorFormat) },
        { if (it.depthW == 640 && it.depthH == 480 && it.depthFps == 30) 0 else 1 },
        { if (it.depthFps == 30) 0 else if (it.depthFps == 15) 1 else 2 },
        { -it.depthFps },
        { -(it.depthW * it.depthH) },
    )

    private fun resolves(pipeline: Pipeline, request: StreamRequest): Boolean {
        return try {
            Config().use { config ->
                if (request.depthEnabled) {
                    config.enableStream(
                        StreamType.DEPTH,
                        request.depthIndex,
                        request.depthW,
                        request.depthH,
                        StreamFormat.Z16,
                        request.depthFps,
                    )
                }
                if (request.colorEnabled) {
                    config.enableStream(
                        StreamType.COLOR,
                        request.colorIndex,
                        request.colorW,
                        request.colorH,
                        request.colorFormat,
                        request.colorFps,
                    )
                }
                config.canResolve(pipeline)
            }
        } catch (e: Exception) {
            Log.w(TAG, "resolve ${request.label}: ${e.message}")
            false
        }
    }

    private fun requestFor(depth: VideoMode?, color: VideoMode?): StreamRequest {
        val formatNote = when (color?.format) {
            null, StreamFormat.YUYV -> ""
            else -> " ${color.format.name}"
        }
        val label = when {
            depth == null && color != null ->
                "цвет ${color.w}×${color.h} @${color.fps}$formatNote"
            depth != null && color == null ->
                "глубина ${depth.w}×${depth.h} @${depth.fps}"
            depth != null && color != null &&
                depth.w == color.w && depth.h == color.h && depth.fps == color.fps ->
                "${depth.w}×${depth.h} @${depth.fps}$formatNote"
            depth != null && color != null ->
                "D ${depth.w}×${depth.h}@${depth.fps} · ${color.w}×${color.h}@${color.fps}$formatNote"
            else -> "пустой профиль"
        }
        return StreamRequest(
            label = label,
            depthIndex = depth?.index ?: 0,
            depthW = depth?.w ?: 0,
            depthH = depth?.h ?: 0,
            depthFps = depth?.fps ?: 0,
            colorIndex = color?.index ?: 0,
            colorW = color?.w ?: 0,
            colorH = color?.h ?: 0,
            colorFps = color?.fps ?: 0,
            colorFormat = color?.format ?: StreamFormat.ANY,
            depthEnabled = depth != null,
            colorEnabled = color != null,
        )
    }

    private fun start(pipeline: Pipeline, request: StreamRequest): Opened {
        Config().use { config ->
            if (request.depthEnabled) {
                config.enableStream(
                    StreamType.DEPTH,
                    request.depthIndex,
                    request.depthW,
                    request.depthH,
                    StreamFormat.Z16,
                    request.depthFps,
                )
            }
            if (request.colorEnabled) {
                config.enableStream(
                    StreamType.COLOR,
                    request.colorIndex,
                    request.colorW,
                    request.colorH,
                    request.colorFormat,
                    request.colorFps,
                )
            }
            pipeline.start(config).use { profile ->
                val device = profile.device
                val deviceLine = try {
                    capColorExposure(device)
                    deviceLine(device)
                } finally {
                    device.close()
                }
                return Opened(deviceLine, request.label)
            }
        }
    }

    private fun capColorExposure(device: com.intel.realsense.librealsense.Device) {
        for (sensor in device.querySensors()) {
            if (!sensor.`is`(Extension.COLOR_SENSOR)) continue
            try {
                val toggle = com.intel.realsense.librealsense.Option.OPTION_AUTO_EXPOSURE_LIMIT_TOGGLE
                val limit = com.intel.realsense.librealsense.Option.AUTO_EXPOSURE_LIMIT
                if (sensor.supports(toggle)) sensor.setValue(toggle, 1f)
                if (!sensor.supports(limit)) return
                val want = 8000f.coerceIn(sensor.getMinRange(limit), sensor.getMaxRange(limit))
                sensor.setValue(limit, want)
                Log.i(TAG, "color exposure limit ${want.toInt()} us")
            } catch (e: Exception) {
                Log.w(TAG, "exposure limit: ${e.message}")
            }
        }
    }

    private fun deviceLine(device: com.intel.realsense.librealsense.Device): String {
        fun info(which: CameraInfo): String =
            if (device.supportsInfo(which)) device.getInfo(which) else "—"
        return "${info(CameraInfo.NAME)}  fw ${info(CameraInfo.FIRMWARE_VERSION)}  USB ${info(CameraInfo.USB_TYPE_DESCRIPTOR)}"
    }

    private fun consume(frames: FrameSet, colorizer: Colorizer) {
        val color = frames.first(StreamType.COLOR)
        val depth = frames.first(StreamType.DEPTH, StreamFormat.Z16)
        try {
            if (color != null && color.`is`(Extension.VIDEO_FRAME)) {
                val video = color.`as`<VideoFrame>(Extension.VIDEO_FRAME)
                val format = color.profile.use { it.format }
                val raw = ByteArray(color.dataSize)
                color.getData(raw)
                if (format == StreamFormat.YUYV) {
                    if (!loggedLuma) {
                        loggedLuma = true
                        logLuma(raw, video.stride, video.height)
                    }
                    colorView?.submit(raw, video.width, video.height, video.stride)
                }
                val line = "color ${video.width}×${video.height} $format   depth ${depthSize(depth)}"
                if (ui.streamLine != line) {
                    val bitmap = if (format == StreamFormat.MJPEG) {
                        BitmapFactory.decodeByteArray(raw, 0, raw.size)
                    } else {
                        null
                    }
                    publish { it.copy(streamLine = line, colorBitmap = bitmap) }
                }
                if (snapshot.compareAndSet(true, false)) {
                    val depthBytes = depth?.let {
                        ByteArray(it.dataSize).also { buf -> it.getData(buf) }
                    }
                    val jpeg = when (format) {
                        StreamFormat.MJPEG -> trimJpeg(raw)
                        StreamFormat.YUYV -> encodeJpeg(yuyvToBitmap(raw, video.width, video.height, video.stride))
                        else -> ByteArray(0)
                    }
                    val note = buildString {
                        appendLine(ui.deviceLine)
                        appendLine(ui.streamLine)
                        appendLine("color_format=$format")
                        appendLine("color_bytes=${raw.size}")
                        appendLine("depth_bytes=${depthBytes?.size ?: 0}")
                        appendLine("timestamp_ms=${color.timestamp}")
                        appendLine("frame=${color.number}")
                    }
                    val saved = saveCapture(context, jpeg, depthBytes, note)
                    publish { it.copy(saveLine = saved) }
                }
            }
            if (depth != null) {
                val colored = depth.applyFilter(colorizer)
                try {
                    glView?.upload(colored)
                } finally {
                    colored.close()
                }
            }
        } finally {
            color?.close()
            depth?.close()
        }
    }

    private fun logLuma(raw: ByteArray, stride: Int, height: Int) {
        val rowBytes = if (stride >= 2) stride else 2
        val limit = minOf(raw.size, rowBytes * height)
        var minY = 255
        var maxY = 0
        var sum = 0L
        var count = 0
        var i = 0
        while (i + 3 < limit) {
            val y0 = raw[i].toInt() and 0xff
            if (y0 < minY) minY = y0
            if (y0 > maxY) maxY = y0
            sum += y0
            count++
            i += 4
        }
        val mean = if (count == 0) 0 else sum / count
        Log.i(TAG, "Y $minY..$maxY mean=$mean bytes=${raw.size} stride=$stride")
    }

    private fun depthSize(depth: com.intel.realsense.librealsense.Frame?): String {
        if (depth == null || !depth.`is`(Extension.VIDEO_FRAME)) return "нет"
        val video = depth.`as`<VideoFrame>(Extension.VIDEO_FRAME)
        return "${video.width}×${video.height} Z16"
    }

    private fun yuyvToBitmap(data: ByteArray, width: Int, height: Int, stride: Int): Bitmap {
        val pixels = IntArray(width * height)
        val rowBytes = if (stride >= width * 2) stride else width * 2
        for (y in 0 until height) {
            var src = y * rowBytes
            val row = y * width
            var x = 0
            while (x + 1 < width && src + 3 < data.size) {
                val y0 = data[src].toInt() and 0xff
                val u = data[src + 1].toInt() and 0xff
                val y1 = data[src + 2].toInt() and 0xff
                val v = data[src + 3].toInt() and 0xff
                pixels[row + x] = yuvToArgb(y0, u, v)
                pixels[row + x + 1] = yuvToArgb(y1, u, v)
                src += 4
                x += 2
            }
        }
        return Bitmap.createBitmap(pixels, width, height, Bitmap.Config.ARGB_8888)
    }

    private fun yuvToArgb(y: Int, u: Int, v: Int): Int {
        val c = y - 16
        val d = u - 128
        val e = v - 128
        val r = (298 * c + 409 * e + 128) shr 8
        val g = (298 * c - 100 * d - 208 * e + 128) shr 8
        val b = (298 * c + 516 * d + 128) shr 8
        return (0xff shl 24) or
            (r.coerceIn(0, 255) shl 16) or
            (g.coerceIn(0, 255) shl 8) or
            b.coerceIn(0, 255)
    }

    private fun encodeJpeg(bitmap: Bitmap): ByteArray {
        val out = java.io.ByteArrayOutputStream()
        bitmap.compress(Bitmap.CompressFormat.JPEG, 90, out)
        return out.toByteArray()
    }

    companion object {
        private const val TAG = "pocket-capture"

        fun trimJpeg(data: ByteArray): ByteArray {
            var end = data.size - 1
            while (end > 1) {
                if (data[end - 1] == 0xFF.toByte() && data[end] == 0xD9.toByte()) {
                    return if (end == data.size - 1) data else data.copyOf(end + 1)
                }
                end--
            }
            return data
        }

        fun saveCapture(context: Context, jpeg: ByteArray, depth: ByteArray?, note: String): String {
            val stamp = SimpleDateFormat("yyyyMMdd_HHmmss_SSS", Locale.US).format(Date())
            return try {
                if (Build.VERSION.SDK_INT >= 29) {
                    saveViaMediaStore(context, stamp, jpeg, depth, note)
                } else {
                    saveViaFiles(context, stamp, jpeg, depth, note)
                }
            } catch (e: Exception) {
                Log.w(TAG, "mediastore failed, writing app dir", e)
                saveViaFiles(context, stamp, jpeg, depth, note)
            }
        }

        private fun saveViaMediaStore(
            context: Context,
            stamp: String,
            jpeg: ByteArray,
            depth: ByteArray?,
            note: String,
        ): String {
            val relative = "Android/media/${context.packageName}/captures"
            writeStore(context, "$stamp.jpg", "image/jpeg", relative, jpeg)
            if (depth != null) {
                writeStore(context, "$stamp.z16", "application/octet-stream", relative, depth)
            }
            writeStore(context, "$stamp.txt", "text/plain", relative, note.toByteArray())
            return "Сохранено $stamp"
        }

        private fun writeStore(
            context: Context,
            name: String,
            mime: String,
            relative: String,
            bytes: ByteArray,
        ) {
            val values = ContentValues().apply {
                put(MediaStore.MediaColumns.DISPLAY_NAME, name)
                put(MediaStore.MediaColumns.MIME_TYPE, mime)
                put(MediaStore.MediaColumns.RELATIVE_PATH, relative)
                put(MediaStore.MediaColumns.IS_PENDING, 1)
            }
            val collection = MediaStore.Files.getContentUri(MediaStore.VOLUME_EXTERNAL_PRIMARY)
            val uri = context.contentResolver.insert(collection, values)
                ?: error("MediaStore не принял $name")
            context.contentResolver.openOutputStream(uri)?.use { it.write(bytes) }
                ?: error("Нет потока для $name")
            values.clear()
            values.put(MediaStore.MediaColumns.IS_PENDING, 0)
            context.contentResolver.update(uri, values, null, null)
        }

        private fun saveViaFiles(
            context: Context,
            stamp: String,
            jpeg: ByteArray,
            depth: ByteArray?,
            note: String,
        ): String {
            val media = context.externalMediaDirs.firstOrNull()
            val dir = if (media != null) {
                File(media, "captures")
            } else {
                File(
                    context.getExternalFilesDir(Environment.DIRECTORY_PICTURES),
                    "captures",
                )
            }
            dir.mkdirs()
            File(dir, "$stamp.jpg").writeBytes(jpeg)
            if (depth != null) File(dir, "$stamp.z16").writeBytes(depth)
            File(dir, "$stamp.txt").writeText(note)
            return "Сохранено ${dir.name}/$stamp"
        }
    }
}
