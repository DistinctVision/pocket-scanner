package com.pocketscanner.capture

import android.content.Context
import android.graphics.SurfaceTexture
import android.util.Log
import android.view.Surface
import android.view.TextureView
import android.os.Handler
import android.os.HandlerThread
import android.os.Looper
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicReference

/**
 * One Vulkan device draws both the color preview and the colorized depth preview.
 * YUYV is converted by a compute shader; depth is uploaded as RGBA.
 */
class VkPreview(context: Context) : TextureView(context), TextureView.SurfaceTextureListener {
    private val pending = AtomicReference<Pending?>(null)
    private val poseFrame = AtomicReference<PoseFrame?>(null)
    var onAttached: ((VkPreview) -> Unit)? = null

    init {
        isOpaque = true
        surfaceTextureListener = this
        if (anchor == null) anchor = this
    }

    fun submitYuyv(data: ByteArray, width: Int, height: Int, stride: Int) {
        pending.set(Pending.Yuyv(data, width, height, stride))
        engine()?.post { drain() }
    }

    fun submitRgb(data: ByteArray, width: Int, height: Int, stride: Int, format: Int) {
        pending.set(Pending.Rgb(data, width, height, stride, format))
        engine()?.post { drain() }
    }

    fun setMode(mode: Int) {
        engine()?.post {
            val handle = sharedHandle
            if (handle != 0L) nativeSetMode(handle, mode)
        }
    }

    fun setFrontier(enabled: Boolean) {
        engine()?.post {
            val handle = sharedHandle
            if (handle != 0L) nativeSetFrontier(handle, enabled)
        }
    }

    fun setView(pose: FloatArray, fx: Float, fy: Float, cx: Float, cy: Float, width: Int, height: Int) {
        val copy = pose.copyOf(16)
        engine()?.post {
            val handle = sharedHandle
            if (handle != 0L) nativeSetView(handle, copy, fx, fy, cx, cy, width, height)
        }
    }

    /** Show this color frame and this pose in one present, so the mesh matches the picture. */
    fun submitPoseFrame(
        data: ByteArray,
        width: Int,
        height: Int,
        stride: Int,
        pose: FloatArray,
        fx: Float,
        fy: Float,
        cx: Float,
        cy: Float,
        imageWidth: Int,
        imageHeight: Int,
    ): Boolean {
        val loop = engine() ?: return false
        poseFrame.set(
            PoseFrame(
                data.copyOf(),
                width,
                height,
                stride,
                pose.copyOf(16),
                fx,
                fy,
                cx,
                cy,
                imageWidth,
                imageHeight,
            ),
        )
        loop.post { drainPose() }
        return true
    }

    fun chunkCount(): Int = shownChunks.get()

    fun chunkPending(): Int = shownPending.get()

    fun chunkUndrawn(): Int = shownUndrawn.get()

    fun chunkCap(): Int = shownCap.get()

    fun meshConfigure(voxelSize: Float, maxDepth: Float): Boolean {
        val loop = engine() ?: return false
        loop.post {
            val handle = sharedHandle
            if (handle != 0L) {
                nativeMeshConfigure(handle, voxelSize, maxDepth)
                publishChunks(handle)
            }
        }
        return true
    }

    fun meshUpsert(
        id: Int,
        pose: FloatArray,
        depth: FloatArray,
        color: ByteArray,
        width: Int,
        height: Int,
        fx: Float,
        fy: Float,
        cx: Float,
        cy: Float,
        force: Boolean = false,
    ): Boolean {
        val loop = engine() ?: return false
        val poseCopy = pose.copyOf(16)
        val depthCopy = depth.copyOf()
        val colorCopy = color.copyOf()
        loop.post {
            if (poseFrame.get() != null) drainPose()
            val handle = sharedHandle
            if (handle != 0L) {
                nativeMeshUpsert(
                    handle, id, poseCopy, depthCopy, colorCopy, width, height, fx, fy, cx, cy, force,
                )
                publishChunks(handle)
            }
        }
        return true
    }

    fun meshErase(id: Int) {
        engine()?.post {
            val handle = sharedHandle
            if (handle != 0L) {
                nativeMeshErase(handle, id)
                publishChunks(handle)
            }
        }
    }

    fun reclaim() {
        val texture = surfaceTexture ?: return
        if (!isAvailable || width <= 0 || height <= 0) return
        bind(texture, width, height)
    }

    override fun onSurfaceTextureAvailable(surface: SurfaceTexture, width: Int, height: Int) {
        bind(surface, width, height)
    }

    private fun bind(texture: SurfaceTexture, width: Int, height: Int) {
        val created = Surface(texture)
        val loop = engineLoop()
        loop.post {
            if (owner === this && bound != null) {
                created.release()
                return@post
            }
            val previous = bound
            owner = this
            if (sharedHandle == 0L) {
                sharedHandle = nativeCreate(created)
                if (sharedHandle == 0L) {
                    Log.e(TAG, "vulkan preview failed to start")
                    created.release()
                    owner = null
                    return@post
                }
            } else if (!nativeAdopt(sharedHandle, created)) {
                Log.e(TAG, "vulkan preview failed to rebind")
            }
            val handle = sharedHandle
            if (handle != 0L) nativeResize(handle, width, height)
            if (previous !== created) previous?.release()
            bound = created
            drain()
            if (!redrawing) {
                redrawing = true
                loop.post(redraw)
            }
            Handler(Looper.getMainLooper()).post { onAttached?.invoke(this) }
        }
    }

    override fun onSurfaceTextureSizeChanged(surface: SurfaceTexture, width: Int, height: Int) {
        engine()?.post {
            val handle = sharedHandle
            if (handle != 0L && owner === this) nativeResize(handle, width, height)
            if (owner === this) drain()
        }
    }

    override fun onSurfaceTextureDestroyed(surface: SurfaceTexture): Boolean {
        val done = CountDownLatch(1)
        val loop = engine()
        if (loop == null) {
            done.countDown()
        } else {
            loop.post {
                if (owner === this) {
                    val handle = sharedHandle
                    if (handle != 0L) nativeDetach(handle)
                    bound?.release()
                    bound = null
                    owner = null
                }
                done.countDown()
            }
        }
        done.await(2, TimeUnit.SECONDS)
        return true
    }

    override fun onSurfaceTextureUpdated(surface: SurfaceTexture) = Unit

    private fun publishChunks(handle: Long) {
        shownChunks.set(nativeChunkCount(handle))
        shownPending.set(nativeChunkPending(handle))
        shownUndrawn.set(nativeChunkUndrawn(handle))
        shownCap.set(nativeChunkCap(handle))
    }

    private fun drain() {
        val frame = pending.getAndSet(null) ?: return
        val handle = sharedHandle
        if (handle == 0L) return
        when (frame) {
            is Pending.Yuyv -> nativeUploadYuyv(handle, frame.data, frame.width, frame.height, frame.stride)
            is Pending.Rgb -> nativeUploadRgb(handle, frame.data, frame.width, frame.height, frame.stride, frame.format)
        }
    }

    private fun drainPose() {
        val frame = poseFrame.getAndSet(null) ?: return
        val handle = sharedHandle
        if (handle == 0L) return
        nativeSetView(handle, frame.pose, frame.fx, frame.fy, frame.cx, frame.cy, frame.imageWidth, frame.imageHeight)
        nativeUploadYuyv(handle, frame.data, frame.width, frame.height, frame.stride)
    }

    private sealed class Pending {
        class Yuyv(val data: ByteArray, val width: Int, val height: Int, val stride: Int) : Pending()
        class Rgb(val data: ByteArray, val width: Int, val height: Int, val stride: Int, val format: Int) : Pending()
    }

    private class PoseFrame(
        val data: ByteArray,
        val width: Int,
        val height: Int,
        val stride: Int,
        val pose: FloatArray,
        val fx: Float,
        val fy: Float,
        val cx: Float,
        val cy: Float,
        val imageWidth: Int,
        val imageHeight: Int,
    )

    private external fun nativeCreate(surface: Surface): Long
    private external fun nativeAdopt(handle: Long, surface: Surface): Boolean
    private external fun nativeDetach(handle: Long)
    private external fun nativeResize(handle: Long, width: Int, height: Int)
    private external fun nativeUploadYuyv(handle: Long, data: ByteArray, width: Int, height: Int, stride: Int)
    private external fun nativeUploadRgb(handle: Long, data: ByteArray, width: Int, height: Int, stride: Int, format: Int)
    private external fun nativeDestroy(handle: Long)
    private external fun nativeSetMode(handle: Long, mode: Int)
    private external fun nativeSetFrontier(handle: Long, enabled: Boolean)
    private external fun nativeSetView(
        handle: Long, pose: FloatArray, fx: Float, fy: Float, cx: Float, cy: Float, width: Int, height: Int,
    )
    private external fun nativeMeshConfigure(handle: Long, voxelSize: Float, maxDepth: Float)
    private external fun nativeMeshUpsert(
        handle: Long, id: Int, pose: FloatArray, depth: FloatArray, color: ByteArray,
        width: Int, height: Int, fx: Float, fy: Float, cx: Float, cy: Float, force: Boolean,
    )
    private external fun nativeMeshErase(handle: Long, id: Int)
    private external fun nativeChunkCount(handle: Long): Int
    private external fun nativeChunkPending(handle: Long): Int
    private external fun nativeChunkUndrawn(handle: Long): Int
    private external fun nativeChunkCap(handle: Long): Int
    private external fun nativeRedraw(handle: Long): Boolean
    private external fun nativeCopySurface(handle: Long, voxelOut: FloatArray): FloatArray?
    private external fun nativeSaveMesh(handle: Long, path: String): Boolean
    private external fun nativeLoadMesh(handle: Long, path: String, depthMeters: Float): Boolean
    private external fun nativeDropQueue(handle: Long)
    private external fun nativeCaptureSeed(handle: Long): Boolean
    private external fun nativeSeedVoxel(handle: Long): Float
    private external fun nativeSplatInitFromSeed(handle: Long, views: Int, width: Int, height: Int): Boolean
    private external fun nativeDropMesh(handle: Long)
    private external fun nativeSplatInit(
        handle: Long, verts: FloatArray, voxel: Float, views: Int, width: Int, height: Int,
    ): Boolean
    private external fun nativeSplatStep(
        handle: Long, rgb: FloatArray, pose: FloatArray, fx: Float, fy: Float, cx: Float, cy: Float,
        view: Int, updateGauss: Boolean, updateGrid: Boolean, metricsAfter: Boolean, metrics: FloatArray,
    ): Boolean
    private external fun nativeSplatRender(
        handle: Long, pose: FloatArray, fx: Float, fy: Float, cx: Float, cy: Float, corrected: Boolean,
    ): ByteArray?
    private external fun nativeSplatWritePly(handle: Long, path: String): Boolean

    fun copySurface(): Pair<FloatArray, Float>? = onGpu(null) { view, handle ->
        if (handle == 0L) return@onGpu null
        val voxel = FloatArray(1)
        val data = view.nativeCopySurface(handle, voxel) ?: return@onGpu null
        data to voxel[0]
    }

    fun saveMesh(path: String): Boolean = onGpu(false) { view, handle ->
        handle != 0L && view.nativeSaveMesh(handle, path)
    }

    fun loadMesh(path: String, depthMeters: Float): Boolean = onGpu(false) { view, handle ->
        handle != 0L && view.nativeLoadMesh(handle, path, depthMeters)
    }

    fun dropQueue() {
        onGpu(Unit) { view, handle ->
            if (handle != 0L) view.nativeDropQueue(handle)
        }
    }

    fun captureSeed(): Boolean = onGpu(false) { view, handle ->
        handle != 0L && view.nativeCaptureSeed(handle)
    }

    fun seedVoxel(): Float = onGpu(0f) { view, handle ->
        if (handle == 0L) 0f else view.nativeSeedVoxel(handle)
    }

    fun splatInitFromSeed(views: Int, width: Int, height: Int): Boolean = onGpu(false) { view, handle ->
        handle != 0L && view.nativeSplatInitFromSeed(handle, views, width, height)
    }

    fun dropMesh() {
        onGpu(Unit) { view, handle ->
            if (handle != 0L) view.nativeDropMesh(handle)
        }
    }

    fun splatInit(verts: FloatArray, voxel: Float, views: Int, width: Int, height: Int): Boolean =
        onGpu(false) { view, handle ->
            handle != 0L && view.nativeSplatInit(handle, verts, voxel, views, width, height)
        }

    fun splatStep(
        rgb: FloatArray,
        pose: FloatArray,
        fx: Float,
        fy: Float,
        cx: Float,
        cy: Float,
        view: Int,
        updateGauss: Boolean,
        updateGrid: Boolean,
        metricsAfter: Boolean,
    ): FloatArray? = onGpu(null) { gpu, handle ->
        if (handle == 0L) return@onGpu null
        val metrics = FloatArray(3)
        val ok = gpu.nativeSplatStep(
            handle, rgb, pose, fx, fy, cx, cy, view, updateGauss, updateGrid, metricsAfter, metrics,
        )
        if (ok) metrics else null
    }

    fun splatRender(pose: FloatArray, fx: Float, fy: Float, cx: Float, cy: Float, corrected: Boolean): ByteArray? =
        onGpu(null) { view, handle ->
            if (handle == 0L) null else view.nativeSplatRender(handle, pose, fx, fy, cx, cy, corrected)
        }

    fun splatWritePly(path: String): Boolean = onGpu(false) { view, handle ->
        handle != 0L && view.nativeSplatWritePly(handle, path)
    }

    companion object {
        const val TAG = "pocket-vk"
        private val shownChunks = AtomicInteger(0)
        private val shownPending = AtomicInteger(0)
        private val shownUndrawn = AtomicInteger(0)
        private val shownCap = AtomicInteger(0)
        private val gate = Any()
        @Volatile private var sharedHandle = 0L
        private var worker: HandlerThread? = null
        private var loop: Handler? = null
        private var owner: VkPreview? = null
        private var anchor: VkPreview? = null
        private var bound: Surface? = null
        private var redrawing = false

        private val redraw = object : Runnable {
            override fun run() {
                val view = owner ?: anchor
                val handle = sharedHandle
                if (owner != null) {
                    owner?.drainPose()
                    owner?.drain()
                }
                if (view != null && handle != 0L) {
                    view.nativeRedraw(handle)
                    view.publishChunks(handle)
                }
                // A zero delay kept this thread inside marching cubes and the pose frame never ran.
                // Finished mesh used to fall back to 200 ms, which showed the review camera at 5 Hz.
                loop?.postDelayed(this, 50L)
            }
        }

        internal fun current(): VkPreview? = anchor

        private fun engine(): Handler? = loop

        internal fun <T> onGpu(fallback: T, body: (VkPreview, Long) -> T): T {
            val handler = loop ?: return fallback
            val view = anchor ?: return fallback
            if (Looper.myLooper() == handler.looper) return body(view, sharedHandle)
            val box = AtomicReference(fallback)
            val done = CountDownLatch(1)
            handler.post {
                try {
                    val current = anchor ?: view
                    box.set(body(current, sharedHandle))
                } catch (error: Throwable) {
                    Log.e(TAG, "gpu", error)
                } finally {
                    done.countDown()
                }
            }
            done.await(180, TimeUnit.SECONDS)
            return box.get()
        }

        private fun engineLoop(): Handler {
            synchronized(gate) {
                val existing = loop
                if (existing != null) return existing
                val thread = HandlerThread("pocket-vk")
                thread.start()
                worker = thread
                val created = Handler(thread.looper)
                loop = created
                return created
            }
        }

        fun shutdown() {
            val handler = loop ?: return
            val done = CountDownLatch(1)
            handler.post {
                val view = anchor
                val handle = sharedHandle
                sharedHandle = 0L
                redrawing = false
                handler.removeCallbacks(redraw)
                if (view != null && handle != 0L) view.nativeDestroy(handle)
                bound?.release()
                bound = null
                owner = null
                shownChunks.set(0)
                shownPending.set(0)
                shownCap.set(0)
                done.countDown()
            }
            done.await(2, TimeUnit.SECONDS)
            worker?.quitSafely()
            synchronized(gate) {
                loop = null
                worker = null
            }
        }

        init {
            System.loadLibrary("pocket_vk")
        }
    }
}
