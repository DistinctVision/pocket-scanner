package com.pocketscanner.capture

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import org.json.JSONArray
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.OutputStream
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.TimeZone
import java.util.zip.Deflater
import java.util.zip.ZipEntry
import java.util.zip.ZipFile
import java.util.zip.ZipOutputStream

data class ViewLens(
    val fx: Float,
    val fy: Float,
    val cx: Float,
    val cy: Float,
    val width: Int,
    val height: Int,
)

internal class ShotMeta(
    val id: Int,
    val pose: FloatArray,
    val width: Int,
    val height: Int,
    val fx: Float,
    val fy: Float,
    val cx: Float,
    val cy: Float,
    val depthScale: Float,
    val file: java.io.File,
)

internal class PackageShot(
    val id: Int,
    val pose: FloatArray,
    val width: Int,
    val height: Int,
    val fx: Float,
    val fy: Float,
    val cx: Float,
    val cy: Float,
    val depthScale: Float,
    val yuyv: ByteArray,
    val yuyvStride: Int,
    val z16: ByteArray,
    val z16Stride: Int,
)

internal class PackageFrame(
    val id: Int,
    val width: Int,
    val height: Int,
    val fx: Float,
    val fy: Float,
    val cx: Float,
    val cy: Float,
    val depthScale: Float,
    val pose: FloatArray,
    val rgb: ByteArray,
    val depth: ByteArray,
)

internal class ScanHeader(
    val voxelCm: Int,
    val reviewVoxelCm: Int,
    val depthScale: Float,
)

internal object ScanPackage {
    fun cachedMesh(context: Context): File = File(context.getExternalFilesDir(null), "packages/loaded-mesh.bin")

    /** Copies mesh.bin into an already saved package. The original stays unchanged if the write fails. */
    fun attachMesh(context: Context, uri: Uri, mesh: File) {
        val dir = File(context.getExternalFilesDir(null), "packages")
        dir.mkdirs()
        val source = File(dir, "mesh-src.pscan")
        val rewritten = File(dir, "mesh-out.pscan")
        context.contentResolver.openInputStream(uri)?.use { input ->
            source.outputStream().use { input.copyTo(it) }
        } ?: error("The scan file could not be opened")
        ZipFile(source).use { zip ->
            ZipOutputStream(rewritten.outputStream().buffered()).use { out ->
                val entries = zip.entries()
                while (entries.hasMoreElements()) {
                    val entry = entries.nextElement()
                    if (entry.name == "mesh.bin") continue
                    out.putNextEntry(ZipEntry(entry.name))
                    zip.getInputStream(entry).use { it.copyTo(out) }
                    out.closeEntry()
                }
                out.putNextEntry(ZipEntry("mesh.bin"))
                mesh.inputStream().use { it.copyTo(out) }
                out.closeEntry()
            }
        }
        val output = context.contentResolver.openOutputStream(uri, "wt")
            ?: error("The scan file is not writable")
        output.use { stream ->
            rewritten.inputStream().use { it.copyTo(stream) }
        }
        rewritten.delete()
        source.delete()
    }

    fun write(dest: OutputStream, scan: ScanSettings, shots: List<ShotMeta>, mesh: File?, load: (ShotMeta) -> PackageShot) {
        val depthScale = shots.first().depthScale
        val manifest = JSONObject()
        manifest.put("format", "pocket-scan")
        manifest.put("version", 1)
        manifest.put("created", stamp())
        manifest.put("voxelCm", scan.voxelCm)
        manifest.put("reviewVoxelCm", scan.reviewVoxelCm)
        manifest.put("depthScale", depthScale.toDouble())
        val frames = JSONArray()
        for (shot in shots) {
            val frame = JSONObject()
            frame.put("id", shot.id)
            frame.put("width", shot.width)
            frame.put("height", shot.height)
            frame.put("fx", shot.fx.toDouble())
            frame.put("fy", shot.fy.toDouble())
            frame.put("cx", shot.cx.toDouble())
            frame.put("cy", shot.cy.toDouble())
            frame.put("pose", JSONArray().apply { shot.pose.forEach { put(it.toDouble()) } })
            frame.put("color", "frames/${shot.id}.png")
            frame.put("depth", "frames/${shot.id}.z16")
            frames.put(frame)
        }
        manifest.put("frames", frames)
        ZipOutputStream(dest.buffered()).use { zip ->
            zip.setLevel(Deflater.DEFAULT_COMPRESSION)
            putBytes(zip, "manifest.json", manifest.toString().toByteArray())
            for (meta in shots) {
                val shot = load(meta)
                putBytes(zip, "frames/${shot.id}.png", png(shot))
                putBytes(zip, "frames/${shot.id}.z16", packDepth(shot))
            }
            if (mesh != null && mesh.isFile && mesh.length() > 16) {
                val entry = ZipEntry("mesh.bin")
                entry.method = ZipEntry.DEFLATED
                zip.putNextEntry(entry)
                mesh.inputStream().use { it.copyTo(zip) }
                zip.closeEntry()
            }
        }
    }

    fun read(context: Context, uri: Uri, prepare: (ScanHeader) -> Unit, accept: (PackageFrame) -> Unit) {
        val kept = File(context.getExternalFilesDir(null), "packages/loaded.pscan")
        kept.parentFile?.mkdirs()
        val input = context.contentResolver.openInputStream(uri)
            ?: error("The selected file could not be opened")
        input.use { stream ->
            kept.outputStream().use { stream.copyTo(it) }
        }
        ZipFile(kept).use { zip ->
            val meshOut = cachedMesh(context)
            val meshEntry = zip.getEntry("mesh.bin")
            if (meshEntry != null) {
                meshOut.parentFile?.mkdirs()
                zip.getInputStream(meshEntry).use { input ->
                    meshOut.outputStream().use { input.copyTo(it) }
                }
            } else {
                meshOut.delete()
            }
            val manifestEntry = zip.getEntry("manifest.json") ?: error("Not a scan package")
            val manifest = JSONObject(zip.getInputStream(manifestEntry).bufferedReader().readText())
            if (manifest.optString("format") != "pocket-scan") error("Not a scan package")
            if (manifest.optInt("version") != 1) error("Unsupported scan package")
            val depthScale = manifest.getDouble("depthScale").toFloat()
            prepare(
                ScanHeader(
                    manifest.optInt("voxelCm", 8),
                    manifest.optInt("reviewVoxelCm", 2),
                    depthScale,
                ),
            )
            val listed = manifest.getJSONArray("frames")
            var count = 0
            for (index in 0 until listed.length()) {
                val frame = listed.getJSONObject(index)
                val id = frame.getInt("id")
                val width = frame.getInt("width")
                val height = frame.getInt("height")
                val poseJson = frame.getJSONArray("pose")
                if (poseJson.length() != 16) error("Frame $id has no pose")
                val pose = FloatArray(16) { poseJson.getDouble(it).toFloat() }
                val colorName = frame.getString("color")
                val depthName = frame.getString("depth")
                require(safeEntry(colorName) && safeEntry(depthName)) { "Bad frame path" }
                val png = entryBytes(zip, colorName)
                val z16 = entryBytes(zip, depthName)
                if (z16.size != width * height * 2) error("Frame $id depth size")
                accept(
                    PackageFrame(
                        id,
                        width,
                        height,
                        frame.getDouble("fx").toFloat(),
                        frame.getDouble("fy").toFloat(),
                        frame.getDouble("cx").toFloat(),
                        frame.getDouble("cy").toFloat(),
                        depthScale,
                        pose,
                        rgb(png, width, height),
                        z16,
                    ),
                )
                count++
            }
            if (count == 0) error("The package has no frames")
        }
    }

    private fun putBytes(zip: ZipOutputStream, name: String, bytes: ByteArray) {
        val entry = ZipEntry(name)
        entry.method = ZipEntry.DEFLATED
        zip.putNextEntry(entry)
        zip.write(bytes)
        zip.closeEntry()
    }

    private fun png(shot: PackageShot): ByteArray {
        val pixels = IntArray(shot.width * shot.height)
        val rowBytes = if (shot.yuyvStride >= shot.width * 2) shot.yuyvStride else shot.width * 2
        for (y in 0 until shot.height) {
            var src = y * rowBytes
            val row = y * shot.width
            var x = 0
            while (x + 1 < shot.width && src + 3 < shot.yuyv.size) {
                val y0 = shot.yuyv[src].toInt() and 0xff
                val u = shot.yuyv[src + 1].toInt() and 0xff
                val y1 = shot.yuyv[src + 2].toInt() and 0xff
                val v = shot.yuyv[src + 3].toInt() and 0xff
                pixels[row + x] = argb(y0, u, v)
                pixels[row + x + 1] = argb(y1, u, v)
                src += 4
                x += 2
            }
        }
        val bitmap = Bitmap.createBitmap(pixels, shot.width, shot.height, Bitmap.Config.ARGB_8888)
        val encoded = ByteArrayOutputStream()
        bitmap.compress(Bitmap.CompressFormat.PNG, 100, encoded)
        bitmap.recycle()
        return encoded.toByteArray()
    }

    private fun argb(luma: Int, u: Int, v: Int): Int {
        val c = luma - 16
        val d = u - 128
        val e = v - 128
        val red = ((298 * c + 409 * e + 128) shr 8).coerceIn(0, 255)
        val green = ((298 * c - 100 * d - 208 * e + 128) shr 8).coerceIn(0, 255)
        val blue = ((298 * c + 516 * d + 128) shr 8).coerceIn(0, 255)
        return (0xff shl 24) or (red shl 16) or (green shl 8) or blue
    }

    private fun packDepth(shot: PackageShot): ByteArray {
        val tight = shot.width * 2
        if (shot.z16Stride == tight && shot.z16.size == tight * shot.height) return shot.z16
        val packed = ByteArray(tight * shot.height)
        val stride = if (shot.z16Stride >= tight) shot.z16Stride else tight
        for (y in 0 until shot.height) {
            val from = y * stride
            if (from + tight > shot.z16.size) break
            shot.z16.copyInto(packed, y * tight, from, from + tight)
        }
        return packed
    }

    private fun rgb(png: ByteArray, width: Int, height: Int): ByteArray {
        val bitmap = BitmapFactory.decodeByteArray(png, 0, png.size) ?: error("Color frame is not a PNG")
        if (bitmap.width != width || bitmap.height != height) {
            bitmap.recycle()
            error("Color frame size")
        }
        val pixels = IntArray(width * height)
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height)
        bitmap.recycle()
        val rgb = ByteArray(width * height * 3)
        for (index in pixels.indices) {
            val pixel = pixels[index]
            val out = index * 3
            rgb[out] = (pixel shr 16).toByte()
            rgb[out + 1] = (pixel shr 8).toByte()
            rgb[out + 2] = pixel.toByte()
        }
        return rgb
    }

    private fun entryBytes(zip: ZipFile, name: String): ByteArray {
        val entry = zip.getEntry(name) ?: error("Missing $name")
        return zip.getInputStream(entry).use { it.readBytes() }
    }

    private fun safeEntry(name: String): Boolean {
        return !name.startsWith("/") && !name.contains("..") && name.startsWith("frames/")
    }

    private fun stamp(): String {
        val format = SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss'Z'", Locale.US)
        format.timeZone = TimeZone.getTimeZone("UTC")
        return format.format(Date())
    }
}
