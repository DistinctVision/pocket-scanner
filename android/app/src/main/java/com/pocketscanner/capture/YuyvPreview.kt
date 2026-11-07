package com.pocketscanner.capture

import android.content.Context
import android.graphics.SurfaceTexture
import android.opengl.EGL14
import android.opengl.EGLConfig
import android.opengl.GLES30
import android.os.Handler
import android.os.HandlerThread
import android.util.Log
import android.view.Surface
import android.view.TextureView
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicReference

/**
 * YUYV goes up as one RGBA texture (two pixels per texel) and a fragment shader
 * expands it. TextureView stays in the normal view tree, so the depth SurfaceView
 * cannot cover this pane.
 */
class YuyvPreview(context: Context) : TextureView(context), TextureView.SurfaceTextureListener {
    private val pending = AtomicReference<Pending?>(null)
    private var thread: HandlerThread? = null
    private var handler: Handler? = null
    private var gl: Gl? = null

    init {
        isOpaque = true
        surfaceTextureListener = this
    }

    fun submit(data: ByteArray, width: Int, height: Int, stride: Int) {
        pending.set(Pending(data, width, height, stride))
        handler?.post { drain() }
    }

    override fun onSurfaceTextureAvailable(surface: SurfaceTexture, width: Int, height: Int) {
        val worker = HandlerThread("yuyv-gl")
        worker.start()
        thread = worker
        val loop = Handler(worker.looper)
        handler = loop
        loop.post {
            val created = Gl(surface)
            if (created.start(width, height)) {
                gl = created
                drain()
            } else {
                created.release()
            }
        }
    }

    override fun onSurfaceTextureSizeChanged(surface: SurfaceTexture, width: Int, height: Int) {
        handler?.post {
            gl?.resize(width, height)
            drain()
        }
    }

    override fun onSurfaceTextureDestroyed(surface: SurfaceTexture): Boolean {
        val done = CountDownLatch(1)
        handler?.post {
            gl?.release()
            gl = null
            done.countDown()
        } ?: done.countDown()
        done.await(1, TimeUnit.SECONDS)
        thread?.quitSafely()
        handler = null
        thread = null
        return true
    }

    override fun onSurfaceTextureUpdated(surface: SurfaceTexture) = Unit

    private fun drain() {
        val frame = pending.getAndSet(null) ?: return
        val renderer = gl ?: return
        renderer.upload(frame.data, frame.width, frame.height, frame.stride)
        renderer.draw()
    }

    private class Pending(val data: ByteArray, val width: Int, val height: Int, val stride: Int)

    private class Gl(private val surfaceTexture: SurfaceTexture) {
        private val display = EGL14.eglGetDisplay(EGL14.EGL_DEFAULT_DISPLAY)
        private var context = EGL14.EGL_NO_CONTEXT
        private var surface = EGL14.EGL_NO_SURFACE
        private var window: Surface? = null
        private var program = 0
        private var texture = 0
        private var fitLocation = 0
        private var sizeLocation = 0
        private var quad: FloatBuffer? = null
        private var direct: ByteBuffer? = null
        private var viewWidth = 1
        private var viewHeight = 1
        private var frameWidth = 0
        private var frameHeight = 0
        private var uploaded = false

        fun start(width: Int, height: Int): Boolean {
            if (display == EGL14.EGL_NO_DISPLAY) return false
            val version = IntArray(2)
            if (!EGL14.eglInitialize(display, version, 0, version, 1)) return false
            val attribs = intArrayOf(
                EGL14.EGL_RED_SIZE, 8,
                EGL14.EGL_GREEN_SIZE, 8,
                EGL14.EGL_BLUE_SIZE, 8,
                EGL14.EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT,
                EGL14.EGL_NONE,
            )
            val configs = arrayOfNulls<EGLConfig>(1)
            val count = IntArray(1)
            if (!EGL14.eglChooseConfig(display, attribs, 0, configs, 0, 1, count, 0) || count[0] == 0) {
                Log.e(TAG, "no GLES3 config")
                return false
            }
            val config = configs[0] ?: return false
            val ctxAttribs = intArrayOf(EGL14.EGL_CONTEXT_CLIENT_VERSION, 3, EGL14.EGL_NONE)
            context = EGL14.eglCreateContext(display, config, EGL14.EGL_NO_CONTEXT, ctxAttribs, 0)
            window = Surface(surfaceTexture)
            val surfaceAttribs = intArrayOf(EGL14.EGL_NONE)
            surface = EGL14.eglCreateWindowSurface(display, config, window, surfaceAttribs, 0)
            if (surface == EGL14.EGL_NO_SURFACE ||
                !EGL14.eglMakeCurrent(display, surface, surface, context)
            ) {
                Log.e(TAG, "eglMakeCurrent failed ${EGL14.eglGetError()}")
                return false
            }
            GLES30.glClearColor(0f, 0f, 0f, 1f)
            program = link(VERTEX, FRAGMENT)
            fitLocation = GLES30.glGetUniformLocation(program, "uFit")
            sizeLocation = GLES30.glGetUniformLocation(program, "uSize")
            val textures = IntArray(1)
            GLES30.glGenTextures(1, textures, 0)
            texture = textures[0]
            GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, texture)
            GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MIN_FILTER, GLES30.GL_LINEAR)
            GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MAG_FILTER, GLES30.GL_LINEAR)
            GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_S, GLES30.GL_CLAMP_TO_EDGE)
            GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_T, GLES30.GL_CLAMP_TO_EDGE)
            quad = floatBuffer(
                floatArrayOf(
                    -1f, -1f, 0f, 1f,
                    1f, -1f, 1f, 1f,
                    -1f, 1f, 0f, 0f,
                    1f, 1f, 1f, 0f,
                ),
            )
            resize(width, height)
            return program != 0
        }

        fun resize(width: Int, height: Int) {
            viewWidth = width.coerceAtLeast(1)
            viewHeight = height.coerceAtLeast(1)
            GLES30.glViewport(0, 0, viewWidth, viewHeight)
        }

        fun upload(data: ByteArray, width: Int, height: Int, stride: Int) {
            if (program == 0 || width < 2 || height < 1) return
            val rowBytes = if (stride >= width * 2) stride else width * 2
            if (rowBytes % 4 != 0 || data.size < rowBytes * height) return
            val sameSize = uploaded && width == frameWidth && height == frameHeight
            frameWidth = width
            frameHeight = height
            val buffer = direct?.takeIf { it.capacity() >= data.size }
                ?: ByteBuffer.allocateDirect(data.size).order(ByteOrder.nativeOrder()).also { direct = it }
            buffer.clear()
            buffer.put(data)
            buffer.flip()
            GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, texture)
            GLES30.glPixelStorei(GLES30.GL_UNPACK_ALIGNMENT, 1)
            GLES30.glPixelStorei(GLES30.GL_UNPACK_ROW_LENGTH, rowBytes / 4)
            if (sameSize) {
                GLES30.glTexSubImage2D(
                    GLES30.GL_TEXTURE_2D, 0, 0, 0, width / 2, height,
                    GLES30.GL_RGBA, GLES30.GL_UNSIGNED_BYTE, buffer,
                )
            } else {
                GLES30.glTexImage2D(
                    GLES30.GL_TEXTURE_2D, 0, GLES30.GL_RGBA, width / 2, height, 0,
                    GLES30.GL_RGBA, GLES30.GL_UNSIGNED_BYTE, buffer,
                )
                uploaded = true
            }
            GLES30.glPixelStorei(GLES30.GL_UNPACK_ROW_LENGTH, 0)
            val error = GLES30.glGetError()
            if (error != GLES30.GL_NO_ERROR) Log.e(TAG, "texImage error $error")
        }

        fun draw() {
            if (surface == EGL14.EGL_NO_SURFACE) return
            GLES30.glClear(GLES30.GL_COLOR_BUFFER_BIT)
            if (uploaded && frameWidth > 0 && program != 0) {
                val viewAspect = viewWidth.toFloat() / viewHeight
                val frameAspect = frameWidth.toFloat() / frameHeight
                val fitX: Float
                val fitY: Float
                if (viewAspect > frameAspect) {
                    fitX = frameAspect / viewAspect
                    fitY = 1f
                } else {
                    fitX = 1f
                    fitY = viewAspect / frameAspect
                }
                GLES30.glUseProgram(program)
                GLES30.glUniform2f(fitLocation, fitX, fitY)
                GLES30.glUniform2f(sizeLocation, frameWidth.toFloat(), frameHeight.toFloat())
                GLES30.glActiveTexture(GLES30.GL_TEXTURE0)
                GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, texture)
                val vertices = quad
                if (vertices != null) {
                    vertices.position(0)
                    GLES30.glEnableVertexAttribArray(0)
                    GLES30.glVertexAttribPointer(0, 2, GLES30.GL_FLOAT, false, 16, vertices)
                    vertices.position(2)
                    GLES30.glEnableVertexAttribArray(1)
                    GLES30.glVertexAttribPointer(1, 2, GLES30.GL_FLOAT, false, 16, vertices)
                    GLES30.glDrawArrays(GLES30.GL_TRIANGLE_STRIP, 0, 4)
                }
            }
            EGL14.eglSwapBuffers(display, surface)
        }

        fun release() {
            if (display == EGL14.EGL_NO_DISPLAY) return
            EGL14.eglMakeCurrent(display, EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_CONTEXT)
            if (surface != EGL14.EGL_NO_SURFACE) EGL14.eglDestroySurface(display, surface)
            if (context != EGL14.EGL_NO_CONTEXT) EGL14.eglDestroyContext(display, context)
            EGL14.eglTerminate(display)
            window?.release()
            surface = EGL14.EGL_NO_SURFACE
            context = EGL14.EGL_NO_CONTEXT
            window = null
        }

        private fun floatBuffer(values: FloatArray): FloatBuffer =
            ByteBuffer.allocateDirect(values.size * 4)
                .order(ByteOrder.nativeOrder())
                .asFloatBuffer()
                .apply {
                    put(values)
                    position(0)
                }

        private fun link(vertexSrc: String, fragmentSrc: String): Int {
            val vertex = compile(GLES30.GL_VERTEX_SHADER, vertexSrc)
            val fragment = compile(GLES30.GL_FRAGMENT_SHADER, fragmentSrc)
            val linked = GLES30.glCreateProgram()
            GLES30.glAttachShader(linked, vertex)
            GLES30.glAttachShader(linked, fragment)
            GLES30.glBindAttribLocation(linked, 0, "aPos")
            GLES30.glBindAttribLocation(linked, 1, "aUv")
            GLES30.glLinkProgram(linked)
            val status = IntArray(1)
            GLES30.glGetProgramiv(linked, GLES30.GL_LINK_STATUS, status, 0)
            if (status[0] == 0) Log.e(TAG, "link: ${GLES30.glGetProgramInfoLog(linked)}")
            GLES30.glDeleteShader(vertex)
            GLES30.glDeleteShader(fragment)
            return linked
        }

        private fun compile(type: Int, source: String): Int {
            val shader = GLES30.glCreateShader(type)
            GLES30.glShaderSource(shader, source)
            GLES30.glCompileShader(shader)
            val status = IntArray(1)
            GLES30.glGetShaderiv(shader, GLES30.GL_COMPILE_STATUS, status, 0)
            if (status[0] == 0) Log.e(TAG, "compile: ${GLES30.glGetShaderInfoLog(shader)}")
            return shader
        }
    }

    private companion object {
        const val TAG = "pocket-capture"
        const val EGL_OPENGL_ES3_BIT = 0x00000040

        val VERTEX = """
            #version 300 es
            layout(location = 0) in vec2 aPos;
            layout(location = 1) in vec2 aUv;
            uniform vec2 uFit;
            out vec2 vUv;
            void main() {
                vUv = aUv;
                gl_Position = vec4(aPos * uFit, 0.0, 1.0);
            }
        """.trimIndent()

        val FRAGMENT = """
            #version 300 es
            precision highp float;
            uniform highp sampler2D uTex;
            uniform vec2 uSize;
            in vec2 vUv;
            out vec4 fragColor;
            void main() {
                int x = clamp(int(vUv.x * uSize.x), 0, int(uSize.x) - 1);
                int y = clamp(int(vUv.y * uSize.y), 0, int(uSize.y) - 1);
                vec4 t = texelFetch(uTex, ivec2(x / 2, y), 0);
                float luma = (x % 2 == 0) ? t.r : t.b;
                float c = luma * 255.0 - 16.0;
                float d = t.g * 255.0 - 128.0;
                float e = t.a * 255.0 - 128.0;
                float r = (298.0 * c + 409.0 * e + 128.0) / 256.0;
                float g = (298.0 * c - 100.0 * d - 208.0 * e + 128.0) / 256.0;
                float b = (298.0 * c + 516.0 * d + 128.0) / 256.0;
                fragColor = vec4(
                    clamp(r, 0.0, 255.0) / 255.0,
                    clamp(g, 0.0, 255.0) / 255.0,
                    clamp(b, 0.0, 255.0) / 255.0,
                    1.0
                );
            }
        """.trimIndent()
    }
}
