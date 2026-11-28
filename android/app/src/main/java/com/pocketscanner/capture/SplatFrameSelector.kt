package com.pocketscanner.capture

import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.Arrays
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.cos
import kotlin.math.floor
import kotlin.math.roundToInt
import kotlin.math.sqrt

/** CPU-side selection on the training worker, before allocating per-view GPU grids. */
internal object SplatFrameSelector {
    private const val MIN_DEPTH = 0.05f
    private const val INSET = 6
    private const val PAIR_DISTANCE = 0.05f
    private const val PATCH_VOXELS = 5
    private const val PATCH_COVERAGE_PERCENT = 92
    private const val PATCH_PARALLAX_PERCENT = 75
    private const val GLOBAL_COVERAGE_PERCENT = 99
    private const val GLOBAL_PARALLAX_PERCENT = 95
    const val MAIN_POSITION = 0.35f
    const val MAIN_ROTATION = 18
    const val POLISH_POSITION = 0.12f
    const val POLISH_ROTATION = 8
    private val pairAngleTerm = (2 * (1 - cos(3 * PI / 180))).toFloat()

    data class Selection(
        val train: List<TrainFrame>,
        val droppedIds: List<Int>,
        val unverifiedIds: List<Int>,
        val surfaceSamples: Int,
        val coveredSamples: Int,
        val parallaxSamples: Int,
        val retainedCoveredSamples: Int,
        val retainedParallaxSamples: Int,
    )

    data class Stages(val main: Selection, val polish: Selection?)

    private data class Observation(
        val indices: IntArray,
        val distances: FloatArray,
        val agreement: Double,
        val sharpness: Double,
    )

    private data class Patch(val x: Int, val y: Int, val z: Int)
    private data class Partner(val index: Int, val baselineSquared: Float)

    /** Returns null when the run is stopped. */
    fun select(
        frames: List<TrainFrame>,
        surface: FloatArray,
        voxel: Float,
        stopped: () -> Boolean,
    ): Selection? = selectStages(frames, surface, voxel, includePolish = false, stopped = stopped)?.main

    fun selectStages(
        frames: List<TrainFrame>,
        surface: FloatArray,
        voxel: Float,
        includePolish: Boolean,
        stopped: () -> Boolean,
    ): Stages? {
        val observations = ArrayList<Observation>(frames.size)
        for (frame in frames) {
            if (stopped()) return null
            observations.add(observe(frame, surface, stopped) ?: return null)
        }
        val patchSize = PATCH_VOXELS * voxel
        val patchIds = HashMap<Patch, Int>()
        val patches = IntArray(surface.size / 3) { i ->
            val patch = Patch(
                floor(surface[i * 3] / patchSize).toInt(),
                floor(surface[i * 3 + 1] / patchSize).toInt(),
                floor(surface[i * 3 + 2] / patchSize).toInt(),
            )
            patchIds.getOrPut(patch) { patchIds.size }
        }
        val main = prune(frames, patches, patchIds.size, observations, MAIN_POSITION, MAIN_ROTATION, stopped)
            ?: return null
        val polish = if (includePolish) {
            prune(frames, patches, patchIds.size, observations, POLISH_POSITION, POLISH_ROTATION, stopped) ?: return null
        } else {
            null
        }
        return Stages(main, polish)
    }

    private fun observe(frame: TrainFrame, surface: FloatArray, stopped: () -> Boolean): Observation? {
        val raw = frame.depth.readBytes()
        val pixels = frame.width * frame.height
        if (raw.size != pixels * Float.SIZE_BYTES) {
            return Observation(IntArray(0), FloatArray(0), 0.0, 0.0)
        }
        val depth = FloatArray(pixels)
        ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(depth)
        val neighbors = FloatArray(9)
        var indices = IntArray(minOf(surface.size / 3, 1024))
        var distances = FloatArray(indices.size)
        var count = 0
        var comparable = 0
        val pose = frame.pose
        for (i in 0 until surface.size / 3) {
            if (i % 2048 == 0 && stopped()) return null
            val worldX = surface[3 * i] - pose[3]
            val worldY = surface[3 * i + 1] - pose[7]
            val worldZ = surface[3 * i + 2] - pose[11]
            val z = pose[2] * worldX + pose[6] * worldY + pose[10] * worldZ
            if (z <= MIN_DEPTH) continue
            val x = pose[0] * worldX + pose[4] * worldY + pose[8] * worldZ
            val y = pose[1] * worldX + pose[5] * worldY + pose[9] * worldZ
            val px = frame.fx * x / z + frame.cx
            val py = frame.fy * y / z + frame.cy
            if (!(px >= INSET && px < frame.width - INSET && py >= INSET && py < frame.height - INSET)) continue
            val pixelX = Math.rint(px.toDouble()).toInt()
            val pixelY = Math.rint(py.toDouble()).toInt()
            var valid = 0
            for (dy in -1..1) {
                for (dx in -1..1) {
                    val value = depth[(pixelY + dy) * frame.width + pixelX + dx]
                    if (value.isFinite() && value > MIN_DEPTH) neighbors[valid++] = value
                }
            }
            if (valid < 5) continue
            comparable++
            Arrays.sort(neighbors, 0, valid)
            if (abs(z - neighbors[valid / 2]) > 0.04f + 0.015f * z) continue
            if (count == indices.size) {
                indices = indices.copyOf(minOf(indices.size * 2, surface.size / 3))
                distances = distances.copyOf(indices.size)
            }
            indices[count] = i
            distances[count] = sqrt(worldX * worldX + worldY * worldY + worldZ * worldZ)
            count++
        }
        return Observation(
            indices.copyOf(count),
            distances.copyOf(count),
            count.toDouble() / maxOf(comparable, 1),
            if (count == 0) 0.0 else sharpness(frame),
        )
    }

    private fun sharpness(frame: TrainFrame): Double {
        val rgb = frame.rgb.readBytes()
        if (rgb.size != frame.width * frame.height * 3) return 0.0
        val width = 320
        val height = 240
        val xEdges = IntArray(width + 1) { (it.toDouble() * frame.width / width).roundToInt() }
        val yEdges = IntArray(height + 1) { (it.toDouble() * frame.height / height).roundToInt() }
        // Separable box sampling of luma, matching the Python thumbnail.
        val rows = IntArray(width * frame.height)
        for (y in 0 until frame.height) {
            for (x in 0 until width) {
                val start = minOf(xEdges[x], frame.width - 1)
                val end = maxOf(xEdges[x + 1], start + 1)
                var sum = 0
                for (src in start until end) {
                    val pixel = (y * frame.width + src) * 3
                    val r = rgb[pixel].toInt() and 255
                    val g = rgb[pixel + 1].toInt() and 255
                    val b = rgb[pixel + 2].toInt() and 255
                    sum += (19595 * r + 38470 * g + 7471 * b + 32768) shr 16
                }
                rows[y * width + x] = (sum + (end - start) / 2) / (end - start)
            }
        }
        val luma = IntArray(width * height)
        for (y in 0 until height) {
            val start = minOf(yEdges[y], frame.height - 1)
            val end = maxOf(yEdges[y + 1], start + 1)
            for (x in 0 until width) {
                var sum = 0
                for (src in start until end) sum += rows[src * width + x]
                luma[y * width + x] = (sum + (end - start) / 2) / (end - start)
            }
        }
        val histogram = IntArray(4 * 255 + 1)
        for (y in 1 until height - 1) {
            for (x in 1 until width - 1) {
                val at = y * width + x
                val value = luma[at - width] + luma[at + width] + luma[at - 1] + luma[at + 1] - 4 * luma[at]
                histogram[abs(value)]++
            }
        }
        val rank = ((width - 2) * (height - 2) - 1) * 0.85
        val lowerRank = floor(rank).toInt()
        val upperRank = ceil(rank).toInt()
        var cumulative = 0
        var lower = -1
        for (value in histogram.indices) {
            cumulative += histogram[value]
            if (cumulative > lowerRank && lower < 0) lower = value
            if (cumulative > upperRank) return lower + (value - lower) * (rank - lowerRank)
        }
        return 0.0
    }

    private fun rotationTrace(pose: FloatArray, other: FloatArray): Float {
        var trace = 0f
        for (row in 0..8 step 4) {
            for (column in 0..2) trace += pose[row + column] * other[row + column]
        }
        return trace
    }

    private inline fun forSharedParallax(
        lookup: FloatArray,
        other: Observation,
        baselineSquared: Float,
        accept: (Int) -> Unit,
    ) {
        for (i in other.indices.indices) {
            val sample = other.indices[i]
            val left = lookup[sample]
            if (left <= 0f) continue
            val right = other.distances[i]
            val delta = left - right
            // Law of cosines for the minimum 3° angle at the surface sample.
            if (baselineSquared >= delta * delta + pairAngleTerm * (left * right)) accept(sample)
        }
    }

    private fun prune(
        frames: List<TrainFrame>,
        patches: IntArray,
        patchCount: Int,
        observations: List<Observation>,
        nearPosition: Float,
        nearRotation: Int,
        stopped: () -> Boolean,
    ): Selection? {
        val sampleCount = patches.size
        val active = BooleanArray(frames.size) { true }
        val lookup = FloatArray(sampleCount)
        val covered = IntArray(sampleCount)
        val pairs = LongArray(sampleCount)
        val neighbors = Array(frames.size) { ArrayList<Partner>() }
        val nearby = Array(frames.size) { ArrayList<Int>() }
        val nearRotationTrace = (1 + 2 * cos(nearRotation * PI / 180)).toFloat()
        for (i in observations.indices) {
            if (stopped()) return null
            val observation = observations[i]
            if (observation.indices.isEmpty()) continue
            nearby[i].add(i)
            for (k in observation.indices.indices) {
                val sample = observation.indices[k]
                covered[sample]++
                lookup[sample] = observation.distances[k]
            }
            val pose = frames[i].pose
            for (j in i + 1 until frames.size) {
                if (j % 16 == 0 && stopped()) return null
                if (observations[j].indices.isEmpty()) continue
                val other = frames[j].pose
                val dx = pose[3] - other[3]
                val dy = pose[7] - other[7]
                val dz = pose[11] - other[11]
                val baseline = dx * dx + dy * dy + dz * dz
                if (baseline <= nearPosition * nearPosition && rotationTrace(pose, other) >= nearRotationTrace) {
                    nearby[i].add(j)
                    nearby[j].add(i)
                }
                if (baseline < PAIR_DISTANCE * PAIR_DISTANCE) continue
                var shared = false
                forSharedParallax(lookup, observations[j], baseline) { sample ->
                    pairs[sample]++
                    shared = true
                }
                if (shared) {
                    neighbors[i].add(Partner(j, baseline))
                    neighbors[j].add(Partner(i, baseline))
                }
            }
            for (sample in observation.indices) lookup[sample] = 0f
        }

        val coveredSamples = covered.count { it > 0 }
        val parallaxSamples = pairs.count { it > 0 }
        var remainingCoverageLoss = (coveredSamples.toLong() * (100 - GLOBAL_COVERAGE_PERCENT) / 100).toInt()
        var remainingParallaxLoss = (parallaxSamples.toLong() * (100 - GLOBAL_PARALLAX_PERCENT) / 100).toInt()
        val coverageAllowance = IntArray(patchCount)
        val parallaxAllowance = IntArray(patchCount)
        for (sample in patches.indices) {
            if (covered[sample] > 0) coverageAllowance[patches[sample]]++
            if (pairs[sample] > 0) parallaxAllowance[patches[sample]]++
        }
        // Round allowable loss down so small patches keep all measured support.
        for (patch in 0 until patchCount) {
            coverageAllowance[patch] = coverageAllowance[patch] * (100 - PATCH_COVERAGE_PERCENT) / 100
            parallaxAllowance[patch] = parallaxAllowance[patch] * (100 - PATCH_PARALLAX_PERCENT) / 100
        }
        val representatives = IntArray(frames.size) { nearby[it].size }
        val incidentPairs = IntArray(sampleCount)
        val coverageLoss = IntArray(patchCount)
        val parallaxLoss = IntArray(patchCount)
        val order = frames.indices.sortedWith(
            compareBy<Int> { observations[it].agreement }
                .thenBy { observations[it].sharpness }
                .thenByDescending { frames[it].id },
        )
        for (i in order) {
            if (stopped()) return null
            val observation = observations[i]
            val samples = observation.indices
            if (samples.isEmpty() || nearby[i].any { representatives[it] <= 1 }) continue
            coverageLoss.fill(0)
            var uncovered = 0
            for (sample in samples) {
                if (covered[sample] == 1) {
                    coverageLoss[patches[sample]]++
                    uncovered++
                }
            }
            if (uncovered > remainingCoverageLoss || coverageLoss.indices.any { coverageLoss[it] > coverageAllowance[it] }) {
                continue
            }
            for (k in samples.indices) lookup[samples[k]] = observation.distances[k]
            for (partner in neighbors[i]) {
                if (active[partner.index]) {
                    forSharedParallax(lookup, observations[partner.index], partner.baselineSquared) { sample ->
                        incidentPairs[sample]++
                    }
                }
            }
            parallaxLoss.fill(0)
            var lostParallax = 0
            for (sample in samples) {
                if (incidentPairs[sample] > 0 && pairs[sample] == incidentPairs[sample].toLong()) {
                    parallaxLoss[patches[sample]]++
                    lostParallax++
                }
            }
            if (lostParallax <= remainingParallaxLoss && parallaxLoss.indices.none { parallaxLoss[it] > parallaxAllowance[it] }) {
                active[i] = false
                for (sample in samples) {
                    covered[sample]--
                    pairs[sample] = pairs[sample] - incidentPairs[sample]
                }
                for (patch in 0 until patchCount) {
                    coverageAllowance[patch] -= coverageLoss[patch]
                    parallaxAllowance[patch] -= parallaxLoss[patch]
                }
                remainingCoverageLoss -= uncovered
                remainingParallaxLoss -= lostParallax
                for (j in nearby[i]) representatives[j]--
            }
            for (sample in samples) {
                incidentPairs[sample] = 0
                lookup[sample] = 0f
            }
        }
        return Selection(
            frames.filterIndexed { i, _ -> active[i] },
            frames.indices.filter { !active[it] }.map { frames[it].id },
            observations.indices.filter { observations[it].indices.isEmpty() }.map { frames[it].id },
            sampleCount,
            coveredSamples,
            parallaxSamples,
            covered.count { it > 0 },
            pairs.count { it > 0 },
        )
    }
}
