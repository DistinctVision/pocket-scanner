package com.pocketscanner.capture

import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.cos
import kotlin.math.sin
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder

class SplatFrameSelectorTest {
    @get:Rule
    val files = TemporaryFolder()

    @Test
    fun preservesTranslatedViewAndKeepsUnverifiedFrames() {
        // Frames 1 and 2 have the same center in different pose buckets. Their
        // two observations do not replace frame 3's measured parallax.
        val frames = listOf(
            frame(1),
            frame(2, yaw = 0.20f),
            frame(3, translation = 0.12f),
            frame(4, depth = 0f),
        )
        val result = checkNotNull(SplatFrameSelector.select(frames, floatArrayOf(0f, 0f, 1f)) { false })
        assertEquals(listOf(1, 3, 4), result.train.map { it.id })
        assertEquals(listOf(2), result.droppedIds)
        assertEquals(listOf(4), result.unverifiedIds)
        assertEquals(1, result.coveredSamples)
        assertEquals(1, result.parallaxSamples)
    }

    @Test
    fun stopCancelsBeforeReadingTrainingFrames() {
        assertNull(SplatFrameSelector.select(listOf(frame(1)), floatArrayOf(0f, 0f, 1f)) { true })
    }

    private fun frame(id: Int, translation: Float = 0f, yaw: Float = 0f, depth: Float = 1f): TrainFrame {
        val width = 64
        val height = 64
        val rgb = files.newFile("$id.rgb").apply { writeBytes(ByteArray(width * height * 3)) }
        val depthFile = files.newFile("$id.f32").apply {
            val bytes = ByteBuffer.allocate(width * height * 4).order(ByteOrder.LITTLE_ENDIAN)
            repeat(width * height) { bytes.putFloat(depth) }
            writeBytes(bytes.array())
        }
        val c = cos(yaw)
        val s = sin(yaw)
        val pose = floatArrayOf(
            c, 0f, s, translation,
            0f, 1f, 0f, 0f,
            -s, 0f, c, 0f,
            0f, 0f, 0f, 1f,
        )
        return TrainFrame(id, pose, width, height, 32f, 32f, 32f, 32f, rgb, depthFile)
    }
}
