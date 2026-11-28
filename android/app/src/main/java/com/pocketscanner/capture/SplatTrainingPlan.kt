package com.pocketscanner.capture

import kotlin.random.Random

internal class SplatTrainingPlan(
    val longTraining: Boolean,
    customEpochs: Int,
    customValidationEvery: Int,
    customBudget: Int,
    trainCount: Int,
    validationCount: Int,
) {
    val mainEpochs = if (longTraining) 24 else customEpochs
    val epochs = mainEpochs + if (longTraining) 3 else 0
    val coarseEpochs = if (longTraining) 18 else 0
    val stepsPerEpoch = if (longTraining) 256 else trainCount
    val validationEvery = if (longTraining) 2 else customValidationEvery.coerceAtLeast(1)
    val budget = if (longTraining) LONG_BUDGET else customBudget
    val meanSteps = mainEpochs * stepsPerEpoch
    val gridSteps = if (longTraining) 0 else meanSteps + (epochs / validationEvery) * validationCount

    fun validatesAt(epoch: Int): Boolean =
        epoch % validationEvery == 0 || (longTraining && epoch == epochs)

    fun <T> trainingOrder(items: List<T>): Iterator<T> {
        require(items.isNotEmpty())
        return sequence {
            val traversal = items.toMutableList()
            val random = Random(SHUFFLE_SEED)
            while (true) {
                if (longTraining) traversal.shuffle(random)
                yieldAll(traversal)
            }
        }.iterator()
    }

    companion object {
        const val LONG_BUDGET = 100_000
        const val SHUFFLE_SEED = 42
    }
}
