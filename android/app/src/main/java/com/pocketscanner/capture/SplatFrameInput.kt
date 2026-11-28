package com.pocketscanner.capture

import java.nio.ByteBuffer
import java.nio.ByteOrder

/** Reusable input storage for a pass at one training resolution. */
internal class SplatFrameInput(
    private val sourceWidth: Int,
    private val sourceHeight: Int,
    val width: Int,
    val height: Int,
) {
    val rgb = FloatArray(width * height * 3)
    val depth = FloatArray(width * height)
    val scaleX = width.toFloat() / sourceWidth
    val scaleY = height.toFloat() / sourceHeight
    private val xEdges = IntArray(width + 1) { (it * sourceWidth + width / 2) / width }
    private val yEdges = IntArray(height + 1) { (it * sourceHeight + height / 2) / height }
    private val depthX = IntArray(width) { (it * 2 + 1) * sourceWidth / (2 * width) }
    private val depthY = IntArray(height) { (it * 2 + 1) * sourceHeight / (2 * height) }

    fun load(frame: TrainFrame) {
        val bytes = frame.rgb.readBytes()
        val stride = bytes.size / sourceHeight
        require(bytes.size % sourceHeight == 0 && stride >= sourceWidth * 3) { "Invalid RGB size" }
        if (width == sourceWidth && height == sourceHeight) {
            for (y in 0 until height) {
                val target = y * width * 3
                for (x in 0 until width * 3) rgb[target + x] = (bytes[y * stride + x].toInt() and 255) / 255f
            }
        } else {
            for (y in 0 until height) {
                val top = yEdges[y]
                val rows = yEdges[y + 1] - top
                for (x in 0 until width) {
                    val left = xEdges[x]
                    val columns = xEdges[x + 1] - left
                    val target = (y * width + x) * 3
                    for (channel in 0..2) {
                        var sum = 0
                        for (sourceY in top until top + rows) {
                            var row = 0
                            for (sourceX in left until left + columns) {
                                row += bytes[sourceY * stride + sourceX * 3 + channel].toInt() and 255
                            }
                            sum += (row + columns / 2) / columns
                        }
                        rgb[target + channel] = ((sum + rows / 2) / rows) / 255f
                    }
                }
            }
        }

        if (!frame.depth.isFile) {
            depth.fill(0f)
            return
        }
        val raw = frame.depth.readBytes()
        require(raw.size == sourceWidth * sourceHeight * Float.SIZE_BYTES) { "Invalid depth size" }
        val samples = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
        if (width == sourceWidth && height == sourceHeight) {
            samples.get(depth)
            return
        }
        for (y in 0 until height) {
            val row = depthY[y] * sourceWidth
            for (x in 0 until width) depth[y * width + x] = samples[row + depthX[x]]
        }
    }
}
