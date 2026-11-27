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
    private val pairAngleTerm = (2 * (1 - cos(3 * PI / 180))).toFloat()

    data class Selection(
        val train: List<TrainFrame>,
        val droppedIds: List<Int>,
        val unverifiedIds: List<Int>,
        val surfaceSamples: Int,
        val coveredSamples: Int,
        val parallaxSamples: Int,
    )

    private data class Observation(
        val indices: IntArray,
        val distances: FloatArray,
        val agreement: Double,
        val sharpness: Double,
    )

    private data class PoseBucket(val x: Int, val y: Int, val z: Int, val fx: Int, val fy: Int, val fz: Int)
    private data class Partner(val index: Int, val baselineSquared: Float)

    /** Returns null when the run is stopped. */
    fun select(
        frames: List<TrainFrame>,
        surface: FloatArray,
        stopped: () -> Boolean,
    ): Selection? {
        val observations = ArrayList<Observation>(frames.size)
        for (frame in frames) {
            if (stopped()) return null
            observations.add(observe(frame, surface))
        }
        return prune(frames, surface.size / 3, observations, removalOrder(frames, observations), stopped)
    }

    private fun observe(frame: TrainFrame, surface: FloatArray): Observation {
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

    private fun removalOrder(frames: List<TrainFrame>, observations: List<Observation>): List<Int> {
        val ranked = frames.indices.sortedWith(
            compareBy<Int> { observations[it].agreement }
                .thenBy { observations[it].sharpness }
                .thenByDescending { frames[it].id },
        )
        val bestByPose = HashMap<PoseBucket, Int>()
        for (i in ranked) {
            val pose = frames[i].pose
            val length = sqrt(pose[2] * pose[2] + pose[6] * pose[6] + pose[10] * pose[10])
            fun bucket(value: Float): Int = if (value < 0) -floor(-value + 0.5f).toInt() else floor(value + 0.5f).toInt()
            val direction = if (length > 0f) 1f / (length * 0.12f) else 0f
            bestByPose[PoseBucket(
                bucket(pose[3] / 0.14f), bucket(pose[7] / 0.14f), bucket(pose[11] / 0.14f),
                bucket(pose[2] * direction), bucket(pose[6] * direction), bucket(pose[10] * direction),
            )] = i
        }
        val best = bestByPose.values.toHashSet()
        return ranked.filter { it !in best } + ranked.filter { it in best }
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
        sampleCount: Int,
        observations: List<Observation>,
        order: List<Int>,
        stopped: () -> Boolean,
    ): Selection? {
        val active = BooleanArray(frames.size) { true }
        val lookup = FloatArray(sampleCount)
        val covered = IntArray(sampleCount)
        val pairs = LongArray(sampleCount)
        val neighbors = Array(frames.size) { ArrayList<Partner>() }
        for (i in observations.indices) {
            if (stopped()) return null
            val observation = observations[i]
            if (observation.indices.isEmpty()) continue
            for (k in observation.indices.indices) {
                val sample = observation.indices[k]
                covered[sample]++
                lookup[sample] = observation.distances[k]
            }
            val pose = frames[i].pose
            for (j in i + 1 until frames.size) {
                val other = frames[j].pose
                val dx = pose[3] - other[3]
                val dy = pose[7] - other[7]
                val dz = pose[11] - other[11]
                val baseline = dx * dx + dy * dy + dz * dz
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
        val minimumViews = IntArray(sampleCount) { minOf(covered[it], 2) }
        val lostPairs = IntArray(sampleCount)
        for (i in order) {
            if (stopped()) return null
            val observation = observations[i]
            val samples = observation.indices
            if (samples.isEmpty() || samples.any { covered[it] <= minimumViews[it] }) continue
            for (k in samples.indices) lookup[samples[k]] = observation.distances[k]
            for (partner in neighbors[i]) {
                if (active[partner.index]) {
                    forSharedParallax(lookup, observations[partner.index], partner.baselineSquared) { sample ->
                        lostPairs[sample]++
                    }
                }
            }
            if (samples.none { lostPairs[it] > 0 && pairs[it] == lostPairs[it].toLong() }) {
                active[i] = false
                for (sample in samples) {
                    covered[sample]--
                    pairs[sample] = pairs[sample] - lostPairs[sample]
                }
            }
            for (sample in samples) {
                lostPairs[sample] = 0
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
        )
    }
}
