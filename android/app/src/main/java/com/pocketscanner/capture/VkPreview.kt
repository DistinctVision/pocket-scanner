package com.pocketscanner.capture

import android.content.Context
import android.graphics.SurfaceTexture
import android.util.Log
import android.view.Surface
import android.view.TextureView
import android.os.Handler
import android.os.HandlerThread
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicReference

/**
 * One Vulkan device draws both the color preview and the colorized depth preview.
 * YUYV is converted by a compute shader; depth is uploaded as RGBA.
 */
class VkPreview(context: Context) : TextureView(context), TextureView.SurfaceTextureListener {
    private val pending = AtomicReference<Pending?>(null)
    private var thread: HandlerThread? = null
    private var handler: Handler? = null
    private var nativeHandle = 0L
    private var window: Surface? = null

    init {
        isOpaque = true
        surfaceTextureListener = this
    }

    fun submitYuyv(data: ByteArray, width: Int, height: Int, stride: Int) {
        pending.set(Pending.Yuyv(data, width, height, stride))
        handler?.post { drain() }
    }

    fun submitRgb(data: ByteArray, width: Int, height: Int, stride: Int, format: Int) {
        pending.set(Pending.Rgb(data, width, height, stride, format))
        handler?.post { drain() }
    }

    fun setMode(mode: Int) {
        handler?.post {
            val handle = nativeHandle
            if (handle != 0L) nativeSetMode(handle, mode)
        }
    }

    fun setView(pose: FloatArray, fx: Float, fy: Float, cx: Float, cy: Float, width: Int, height: Int) {
        val copy = pose.copyOf(16)
        handler?.post {
            val handle = nativeHandle
            if (handle != 0L) nativeSetView(handle, copy, fx, fy, cx, cy, width, height)
        }
    }

    fun meshConfigure(voxelSize: Float, maxDepth: Float) {
        handler?.post {
            val handle = nativeHandle
            if (handle != 0L) nativeMeshConfigure(handle, voxelSize, maxDepth)
        }
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
    ): Boolean {
        val loop = handler ?: return false
        val poseCopy = pose.copyOf(16)
        val depthCopy = depth.copyOf()
        val colorCopy = color.copyOf()
        loop.post {
            val handle = nativeHandle
            if (handle != 0L) {
                nativeMeshUpsert(handle, id, poseCopy, depthCopy, colorCopy, width, height, fx, fy, cx, cy)
            }
        }
        return true
    }

    fun meshErase(id: Int) {
        handler?.post {
            val handle = nativeHandle
            if (handle != 0L) nativeMeshErase(handle, id)
        }
    }

    override fun onSurfaceTextureAvailable(surface: SurfaceTexture, width: Int, height: Int) {
        val worker = HandlerThread("pocket-vk")
        worker.start()
        thread = worker
        val loop = Handler(worker.looper)
        handler = loop
        loop.post {
            val created = Surface(surface)
            window = created
            nativeHandle = nativeCreate(created)
            if (nativeHandle == 0L) {
                Log.e(TAG, "vulkan preview failed to start")
                return@post
            }
            nativeResize(nativeHandle, width, height)
            drain()
            loop.post(redraw)
        }
    }

    override fun onSurfaceTextureSizeChanged(surface: SurfaceTexture, width: Int, height: Int) {
        handler?.post {
            val handle = nativeHandle
            if (handle != 0L) nativeResize(handle, width, height)
            drain()
        }
    }

    override fun onSurfaceTextureDestroyed(surface: SurfaceTexture): Boolean {
        val done = CountDownLatch(1)
        handler?.removeCallbacks(redraw)
        handler?.post {
            val handle = nativeHandle
            nativeHandle = 0L
            if (handle != 0L) nativeDestroy(handle)
            window?.release()
            window = null
            done.countDown()
        } ?: done.countDown()
        done.await(2, TimeUnit.SECONDS)
        thread?.quitSafely()
        handler = null
        thread = null
        return true
    }

    override fun onSurfaceTextureUpdated(surface: SurfaceTexture) = Unit

    private val redraw = object : Runnable {
        override fun run() {
            val handle = nativeHandle
            if (handle != 0L) nativeRedraw(handle)
            handler?.postDelayed(this, 200)
        }
    }

    private fun drain() {
        val frame = pending.getAndSet(null) ?: return
        val handle = nativeHandle
        if (handle == 0L) return
        when (frame) {
            is Pending.Yuyv -> nativeUploadYuyv(handle, frame.data, frame.width, frame.height, frame.stride)
            is Pending.Rgb -> nativeUploadRgb(handle, frame.data, frame.width, frame.height, frame.stride, frame.format)
        }
    }

    private sealed class Pending {
        class Yuyv(val data: ByteArray, val width: Int, val height: Int, val stride: Int) : Pending()
        class Rgb(val data: ByteArray, val width: Int, val height: Int, val stride: Int, val format: Int) : Pending()
    }

    private external fun nativeCreate(surface: Surface): Long
    private external fun nativeResize(handle: Long, width: Int, height: Int)
    private external fun nativeUploadYuyv(handle: Long, data: ByteArray, width: Int, height: Int, stride: Int)
    private external fun nativeUploadRgb(handle: Long, data: ByteArray, width: Int, height: Int, stride: Int, format: Int)
    private external fun nativeDestroy(handle: Long)
    private external fun nativeSetMode(handle: Long, mode: Int)
    private external fun nativeSetView(
        handle: Long, pose: FloatArray, fx: Float, fy: Float, cx: Float, cy: Float, width: Int, height: Int,
    )
    private external fun nativeMeshConfigure(handle: Long, voxelSize: Float, maxDepth: Float)
    private external fun nativeMeshUpsert(
        handle: Long, id: Int, pose: FloatArray, depth: FloatArray, color: ByteArray,
        width: Int, height: Int, fx: Float, fy: Float, cx: Float, cy: Float,
    )
    private external fun nativeMeshErase(handle: Long, id: Int)
    private external fun nativeRedraw(handle: Long)

    private companion object {
        const val TAG = "pocket-vk"
        init {
            System.loadLibrary("pocket_vk")
        }
    }
}
