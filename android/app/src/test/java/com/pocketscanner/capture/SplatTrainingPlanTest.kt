package com.pocketscanner.capture

import java.nio.ByteBuffer
import java.nio.ByteOrder
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder

class SplatTrainingPlanTest {
    @get:Rule
    val files = TemporaryFolder()

    @Test
    fun longProfileKeepsTheSameStepBudgetWithAndWithoutFiltering() {
        val filtered = SplatTrainingPlan(true, 100, 5, 3_000_000, 189, 22)
        val control = SplatTrainingPlan(true, 100, 5, 3_000_000, 406, 22)
        assertEquals(6144, filtered.meanSteps)
        assertEquals(filtered.meanSteps, control.meanSteps)
        assertEquals(4608, filtered.coarseEpochs * filtered.stepsPerEpoch)
        assertEquals(6912, filtered.epochs * filtered.stepsPerEpoch)
        assertEquals(0, filtered.gridSteps)
        assertEquals(100_000, filtered.budget)
        assertFalse(filtered.validatesAt(1))
        assertTrue(filtered.validatesAt(24))
        assertTrue(filtered.validatesAt(27))
        val frames = (1..189).toList()
        val order = filtered.trainingOrder(frames)
        val firstEpoch = List(256) { order.next() }
        assertEquals(frames.toSet(), firstEpoch.take(189).toSet())
        val sameOrder = filtered.trainingOrder(frames)
        assertEquals(firstEpoch, List(256) { sameOrder.next() })
        val secondTraversal = firstEpoch.drop(189) + List(122) { order.next() }
        assertEquals(frames.toSet(), secondTraversal.toSet())
    }

    @Test
    fun customProfileStillTraversesEachViewOncePerEpoch() {
        val plan = SplatTrainingPlan(false, 10, 5, 500_000, 3, 2)
        assertEquals(10, plan.epochs)
        assertEquals(0, plan.coarseEpochs)
        assertEquals(30, plan.meanSteps)
        assertEquals(34, plan.gridSteps)
        val order = plan.trainingOrder(listOf(1, 2, 3))
        assertEquals(listOf(1, 2, 3, 1, 2, 3), List(6) { order.next() })
    }

    @Test
    fun scaledInputUsesBoxColorAndNearestDepth() {
        val rgb = files.newFile().apply { writeBytes(ByteArray(48) { (it / 3).toByte() }) }
        val depth = files.newFile().apply {
            val bytes = ByteBuffer.allocate(64).order(ByteOrder.LITTLE_ENDIAN)
            repeat(16) { bytes.putFloat(it.toFloat()) }
            writeBytes(bytes.array())
        }
        val frame = TrainFrame(1, FloatArray(16), 4, 4, 2f, 2f, 2f, 2f, rgb, depth)
        val input = SplatFrameInput(4, 4, 2, 2)
        input.load(frame)
        assertArrayEquals(floatArrayOf(5f, 7f, 13f, 15f), input.depth, 0f)
        assertArrayEquals(FloatArray(12) { intArrayOf(3, 5, 11, 13)[it / 3] / 255f }, input.rgb, 1e-6f)
        assertEquals(.5f, input.scaleX, 0f)
        assertEquals(.5f, input.scaleY, 0f)
    }
}
