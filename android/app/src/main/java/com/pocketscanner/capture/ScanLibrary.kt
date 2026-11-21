package com.pocketscanner.capture

import android.content.Context
import android.content.Intent
import android.net.Uri
import android.provider.DocumentsContract
import android.provider.OpenableColumns
import androidx.activity.result.contract.ActivityResultContracts
import org.json.JSONArray
import org.json.JSONObject
import java.io.File

internal data class SavedScan(
    val name: String,
    val uri: String,
    val savedAt: Long,
    val frames: Int,
)

internal object ScanLibrary {
    private val lock = Any()

    fun list(context: Context): List<SavedScan> = synchronized(lock) {
        read(context).sortedByDescending { it.savedAt }
    }

    fun persist(context: Context, uri: Uri) {
        val resolver = context.contentResolver
        try {
            resolver.takePersistableUriPermission(uri, Intent.FLAG_GRANT_READ_URI_PERMISSION)
        } catch (_: SecurityException) {
        }
        try {
            resolver.takePersistableUriPermission(uri, Intent.FLAG_GRANT_WRITE_URI_PERMISSION)
        } catch (_: SecurityException) {
        }
    }

    fun forget(context: Context, uri: Uri) = synchronized(lock) {
        write(context, read(context).filter { it.uri != uri.toString() })
        erase(context, uri)
    }

    fun forgetAll(context: Context) = synchronized(lock) {
        val gone = read(context)
        write(context, emptyList())
        for (scan in gone) erase(context, Uri.parse(scan.uri))
    }

    fun remember(context: Context, uri: Uri, name: String, frames: Int) = synchronized(lock) {
        val next = read(context).filter { it.uri != uri.toString() }.toMutableList()
        next.add(SavedScan(name, uri.toString(), System.currentTimeMillis(), frames))
        write(context, next)
    }

    fun displayName(context: Context, uri: Uri): String {
        val cursor = context.contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)
        cursor?.use {
            if (it.moveToFirst()) {
                val name = it.getString(0)
                if (!name.isNullOrBlank()) return name
            }
        }
        return uri.lastPathSegment ?: "scan.pscan"
    }

    private fun erase(context: Context, uri: Uri) {
        try {
            DocumentsContract.deleteDocument(context.contentResolver, uri)
        } catch (_: Exception) {
        }
        val resolver = context.contentResolver
        val both = Intent.FLAG_GRANT_READ_URI_PERMISSION or Intent.FLAG_GRANT_WRITE_URI_PERMISSION
        try {
            resolver.releasePersistableUriPermission(uri, both)
        } catch (_: SecurityException) {
            try {
                resolver.releasePersistableUriPermission(uri, Intent.FLAG_GRANT_READ_URI_PERMISSION)
            } catch (_: SecurityException) {
            }
        }
    }

    private fun file(context: Context) = File(context.filesDir, "saved-scans.json")

    private fun read(context: Context): List<SavedScan> {
        val stored = file(context)
        if (!stored.exists()) return emptyList()
        return try {
            val array = JSONArray(stored.readText())
            buildList {
                for (index in 0 until array.length()) {
                    val item = array.getJSONObject(index)
                    add(
                        SavedScan(
                            item.optString("name", "scan.pscan"),
                            item.getString("uri"),
                            item.optLong("savedAt"),
                            item.optInt("frames"),
                        ),
                    )
                }
            }
        } catch (_: Exception) {
            emptyList()
        }
    }

    private fun write(context: Context, scans: List<SavedScan>) {
        val array = JSONArray()
        for (scan in scans) {
            array.put(
                JSONObject()
                    .put("name", scan.name)
                    .put("uri", scan.uri)
                    .put("savedAt", scan.savedAt)
                    .put("frames", scan.frames),
            )
        }
        file(context).writeText(array.toString())
    }
}

internal class CreatePersistableDocument :
    ActivityResultContracts.CreateDocument("application/octet-stream") {
    override fun createIntent(context: Context, input: String): Intent {
        return super.createIntent(context, input).addFlags(
            Intent.FLAG_GRANT_READ_URI_PERMISSION or
                Intent.FLAG_GRANT_WRITE_URI_PERMISSION or
                Intent.FLAG_GRANT_PERSISTABLE_URI_PERMISSION,
        )
    }
}

internal class OpenPersistableDocument : ActivityResultContracts.OpenDocument() {
    override fun createIntent(context: Context, input: Array<String>): Intent {
        return super.createIntent(context, input).addFlags(
            Intent.FLAG_GRANT_READ_URI_PERMISSION or
                Intent.FLAG_GRANT_PERSISTABLE_URI_PERMISSION,
        )
    }
}
