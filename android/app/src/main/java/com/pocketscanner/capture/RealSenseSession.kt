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
import com.intel.realsense.librealsense.Align
import com.intel.realsense.librealsense.CameraInfo
import com.intel.realsense.librealsense.Colorizer
import com.intel.realsense.librealsense.Config
import com.intel.realsense.librealsense.DepthSensor
import com.intel.realsense.librealsense.DeviceListener
import com.intel.realsense.librealsense.DistortionType
import com.intel.realsense.librealsense.Extension
import com.intel.realsense.librealsense.FrameSet
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
    val headline: String = "Plug in the Realsense-sensor",
    val deviceLine: String = "",
    val streamLine: String = "",
    val fps: Float = 0f,
    val track: TrackSample? = null,
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
 * One camera thread owns the pipeline. The Vulkan preview draws YUYV color and
 * colorized depth; MJPEG is decoded to a bitmap because it is not a live view.
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
    private var preview: VkPreview? = null
    private val showColor = AtomicBoolean(true)
    private var loggedLuma = false
    private var nextColorPreviewNs = 0L
    private var nextDepthPreviewNs = 0L
    private var worker: Thread? = null

    @Volatile
    private var ui = CaptureUi()

    fun attachPreview(view: VkPreview?) {
        preview = view
    }

    fun setShowsColor(color: Boolean) {
        showColor.set(color)
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

    @Synchronized
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
                headline = "Device is not connected",
                detail = "Waiting for connection...",
            )
        }
        val lines = devices.joinToString("\n") { device ->
            val access = if (manager.hasPermission(device)) "permission granted" else "no permission"
            val name = device.productName ?: device.deviceName
            "%04X:%04X %s (%s)".format(device.vendorId, device.productId, name, access)
        }
        val intel = devices.filter { it.vendorId == 0x8086 }
        val headline = when {
            intel.isEmpty() -> "USB device found, no RealSense"
            intel.any { manager.hasPermission(it) } -> "RealSense on USB, opening"
            else -> "USB permission required"
        }
        return UsbStatus(headline, lines)
    }

    private fun loop() {
        val tracker = StellaTracker(context) { sample ->
            publish { it.copy(track = sample) }
        }
        val align = Align(StreamType.COLOR)
        val rs = RsContext()
        val listener = object : DeviceListener {
            override fun onDeviceAttach() {
                streaming.set(true)
                publish { it.copy(headline = "Camera connected") }
            }

            override fun onDeviceDetach() {
                streaming.set(false)
                val n = detaches.incrementAndGet()
                publish {
                    it.copy(
                        headline = "Camera disconnected",
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
                    streamUntilStopped(rs, tracker, align)
                } catch (e: Exception) {
                    Log.e(TAG, "stream failed", e)
                    val line = (e.message ?: "Stream error").lineSequence().first()
                    publish { it.copy(headline = line) }
                    streaming.set(false)
                    Thread.sleep(2000)
                }
            }
        } finally {
            tracker.close()
            align.close()
            rs.close()
        }
    }

    private fun streamUntilStopped(rs: RsContext, tracker: StellaTracker, align: Align) {
        val pipeline = Pipeline()
        val colorizer = Colorizer()
        try {
            val opened = openPipeline(rs, pipeline, colorizer, align, tracker)
            tracker.configure(opened.camera)
            publish {
                it.copy(
                    headline = "Streaming",
                    deviceLine = opened.deviceLine,
                    streamLine = opened.streamLine,
                )
            }
            var framesInWindow = 0
            var windowStart = System.nanoTime()
            while (alive.get() && streaming.get()) {
                try {
                    val frames = pipeline.waitForFrames(2000)
                    frames.use { set -> consume(set, colorizer, align, tracker) }
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
            tracker.configure(null)
            try {
                pipeline.stop()
            } catch (_: Exception) {
            }
            colorizer.close()
            pipeline.close()
        }
    }

    private data class Opened(
        val deviceLine: String,
        val streamLine: String,
        val camera: SlamCamera?,
    )

    private fun openPipeline(
        rs: RsContext,
        pipeline: Pipeline,
        colorizer: Colorizer,
        align: Align,
        tracker: StellaTracker,
    ): Opened {
        val probe = probe(rs)
        val options = catalog(pipeline, probe.modes)
        catalog = options
        Log.i(TAG, "usb=${probe.usb} profiles=${options.joinToString(" | ") { it.label }}")
        if (options.isEmpty()) {
            throw IllegalStateException("Camera did not advertise a depth and color pair")
        }
        loggedLuma = false
        val request = pick(options)
        publish {
            it.copy(
                headline = "Opening ${request.label}",
                profiles = options.map { option -> option.label },
                selectedProfile = request.label,
            )
        }
        try {
            val opened = start(pipeline, request)
            Log.i(TAG, "started ${request.label}")
            val frames = pipeline.waitForFrames(4000)
            frames.use { consume(it, colorizer, align, tracker) }
            Log.i(TAG, "first frame ${request.label}")
            return opened
        } catch (e: Exception) {
            Log.w(TAG, "profile ${request.label} failed: ${e.message}")
            try {
                pipeline.stop()
            } catch (_: Exception) {
            }
            throw IllegalStateException("${request.label}: ${e.message ?: "no frames"}")
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
                "color ${color.w}×${color.h} @${color.fps}$formatNote"
            depth != null && color == null ->
                "depth ${depth.w}×${depth.h} @${depth.fps}"
            depth != null && color != null &&
                depth.w == color.w && depth.h == color.h && depth.fps == color.fps ->
                "${depth.w}×${depth.h} @${depth.fps}$formatNote"
            depth != null && color != null ->
                "D ${depth.w}×${depth.h}@${depth.fps} · ${color.w}×${color.h}@${color.fps}$formatNote"
            else -> "empty profile"
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
                try {
                    capColorExposure(device)
                    val line = deviceLine(device)
                    val camera = cameraOf(device, request)
                    return Opened(line, request.label, camera)
                } finally {
                    device.close()
                }
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

    private fun cameraOf(
        device: com.intel.realsense.librealsense.Device,
        request: StreamRequest,
    ): SlamCamera? {
        if (!request.depthEnabled || !request.colorEnabled || request.colorFormat != StreamFormat.YUYV) {
            return null
        }
        var factor = 1000.0
        var fx = 0.0
        var fy = 0.0
        var cx = 0.0
        var cy = 0.0
        var k1 = 0.0
        var k2 = 0.0
        var p1 = 0.0
        var p2 = 0.0
        var k3 = 0.0
        var note = "pinhole"
        var found = false
        for (sensor in device.querySensors()) {
            if (sensor.`is`(Extension.DEPTH_SENSOR)) {
                try {
                    val scale = sensor.`as`<DepthSensor>(Extension.DEPTH_SENSOR).depthScale
                    if (scale > 1e-9f) factor = 1.0 / scale.toDouble()
                } catch (e: Exception) {
                    Log.w(TAG, "depth scale: ${e.message}")
                }
            }
            val profiles = sensor.activeStreams.ifEmpty { sensor.streamProfiles }
            for (profile in profiles) {
                if (profile.type != StreamType.COLOR || profile.format != StreamFormat.YUYV) continue
                if (!profile.`is`(Extension.VIDEO_PROFILE)) continue
                val video = profile.`as`<VideoStreamProfile>(Extension.VIDEO_PROFILE)
                if (video.width != request.colorW || video.height != request.colorH) continue
                try {
                    val intrinsic = video.intrinsic
                    fx = intrinsic.fx.toDouble()
                    fy = intrinsic.fy.toDouble()
                    cx = intrinsic.ppx.toDouble()
                    cy = intrinsic.ppy.toDouble()
                    val coeffs = intrinsic.coeffs ?: FloatArray(5)
                    val model = intrinsic.model
                    if (model == DistortionType.BROWN_CONRADY) {
                        k1 = coeffs.getOrElse(0) { 0f }.toDouble()
                        k2 = coeffs.getOrElse(1) { 0f }.toDouble()
                        p1 = coeffs.getOrElse(2) { 0f }.toDouble()
                        p2 = coeffs.getOrElse(3) { 0f }.toDouble()
                        k3 = coeffs.getOrElse(4) { 0f }.toDouble()
                        note = "brown"
                    } else {
                        note = "pinhole, $model is not the OpenCV model"
                    }
                    found = fx > 0.0 && fy > 0.0
                    Log.i(TAG, "intrinsic fx=$fx fy=$fy cx=$cx cy=$cy model=$model factor=$factor $note")
                } catch (e: Exception) {
                    Log.w(TAG, "intrinsic: ${e.message}")
                }
            }
        }
        if (!found) return null
        return SlamCamera(
            w = request.colorW,
            h = request.colorH,
            fps = request.colorFps,
            fx = fx,
            fy = fy,
            cx = cx,
            cy = cy,
            k1 = k1,
            k2 = k2,
            p1 = p1,
            p2 = p2,
            k3 = k3,
            depthFactor = factor,
            note = note,
        )
    }

    private fun deviceLine(device: com.intel.realsense.librealsense.Device): String {
        fun info(which: CameraInfo): String =
            if (device.supportsInfo(which)) device.getInfo(which) else "—"
        return "${info(CameraInfo.NAME)}  fw ${info(CameraInfo.FIRMWARE_VERSION)}  USB ${info(CameraInfo.USB_TYPE_DESCRIPTOR)}"
    }

    private fun consume(frames: FrameSet, colorizer: Colorizer, align: Align, tracker: StellaTracker) {
        val color = frames.first(StreamType.COLOR)
        val depth = frames.first(StreamType.DEPTH, StreamFormat.Z16)
        try {
            if (color != null && color.`is`(Extension.VIDEO_FRAME)) {
                val video = color.`as`<VideoFrame>(Extension.VIDEO_FRAME)
                val format = color.profile.use { it.format }
                val now = System.nanoTime()
                val wantPreview = format == StreamFormat.YUYV && showColor.get() && now >= nextColorPreviewNs
                val wantSnapshot = snapshot.get()
                val raw = if (wantPreview || wantSnapshot || format != StreamFormat.YUYV) {
                    ByteArray(color.dataSize).also { color.getData(it) }
                } else {
                    null
                }
                if (format == StreamFormat.YUYV) {
                    if (raw != null && wantPreview && showColor.get()) {
                        if (!loggedLuma) {
                            loggedLuma = true
                            logLuma(raw, video.stride, video.height)
                        }
                        nextColorPreviewNs = now + 66_000_000L
                        preview?.submitYuyv(raw, video.width, video.height, video.stride)
                    }
                    if (depth != null && tracker.wantsFrame()) {
                        offerAligned(frames, align, tracker)
                    }
                }
                val line = "color ${video.width}×${video.height} $format   depth ${depthSize(depth)}"
                if (ui.streamLine != line) {
                    val bitmap = if (format == StreamFormat.MJPEG && raw != null) {
                        BitmapFactory.decodeByteArray(raw, 0, raw.size)
                    } else {
                        null
                    }
                    publish { it.copy(streamLine = line, colorBitmap = bitmap) }
                }
                if (raw != null && snapshot.compareAndSet(true, false)) {
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
            if (depth != null && preview != null && !showColor.get() && System.nanoTime() >= nextDepthPreviewNs) {
                nextDepthPreviewNs = System.nanoTime() + 125_000_000L
                val colored = depth.applyFilter(colorizer)
                try {
                    if (colored.`is`(Extension.VIDEO_FRAME)) {
                        val video = colored.`as`<VideoFrame>(Extension.VIDEO_FRAME)
                        val format = video.profile.use { it.format.value() }
                        val raw = ByteArray(colored.dataSize)
                        colored.getData(raw)
                        preview?.submitRgb(raw, video.width, video.height, video.stride, format)
                    }
                } finally {
                    colored.close()
                }
            }
        } finally {
            color?.close()
            depth?.close()
        }
    }

    private fun offerAligned(frames: FrameSet, align: Align, tracker: StellaTracker) {
        var aligned: FrameSet? = null
        var ownsAligned = false
        var color: com.intel.realsense.librealsense.Frame? = null
        var depth: com.intel.realsense.librealsense.Frame? = null
        try {
            aligned = align.process(frames)
            ownsAligned = aligned !== frames
            val colorFrame = aligned.first(StreamType.COLOR, StreamFormat.YUYV)
            val depthFrame = aligned.first(StreamType.DEPTH, StreamFormat.Z16)
            color = colorFrame
            depth = depthFrame
            if (colorFrame == null || depthFrame == null) return
            if (!colorFrame.`is`(Extension.VIDEO_FRAME) || !depthFrame.`is`(Extension.VIDEO_FRAME)) return
            val colorVideo = colorFrame.`as`<VideoFrame>(Extension.VIDEO_FRAME)
            val depthVideo = depthFrame.`as`<VideoFrame>(Extension.VIDEO_FRAME)
            if (colorVideo.width != depthVideo.width || colorVideo.height != depthVideo.height) return
            val colorBytes = ByteArray(colorFrame.dataSize)
            val depthBytes = ByteArray(depthFrame.dataSize)
            colorFrame.getData(colorBytes)
            depthFrame.getData(depthBytes)
            tracker.offer(
                color = colorBytes,
                width = colorVideo.width,
                height = colorVideo.height,
                stride = colorVideo.stride,
                depth = depthBytes,
                depthW = depthVideo.width,
                depthH = depthVideo.height,
                depthStride = depthVideo.stride,
                timestampSec = colorFrame.timestamp / 1000.0,
            )
        } catch (e: Exception) {
            Log.w(TAG, "align: ${e.message}")
        } finally {
            color?.close()
            depth?.close()
            if (ownsAligned) aligned?.close()
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
        if (depth == null || !depth.`is`(Extension.VIDEO_FRAME)) return "none"
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
            return "Saved $stamp"
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
                ?: error("MediaStore rejected $name")
            context.contentResolver.openOutputStream(uri)?.use { it.write(bytes) }
                ?: error("No output stream for $name")
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
            return "Saved ${dir.name}/$stamp"
        }
    }
}
