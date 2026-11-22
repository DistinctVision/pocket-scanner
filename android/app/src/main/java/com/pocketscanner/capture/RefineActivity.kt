package com.pocketscanner.capture

import android.os.Bundle
import android.view.WindowManager
import androidx.activity.ComponentActivity
import androidx.activity.addCallback
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.LaunchedEffect
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class RefineActivity : ComponentActivity() {
    private var session: RealSenseSession? = null
    private var preview: VkPreview? = null
    private var retained = false
    private val createScan = registerForActivityResult(CreatePersistableDocument()) { uri ->
        if (uri != null) session?.savePackage(uri)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        enableEdgeToEdge()
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        onBackPressedDispatcher.addCallback(this) { leave() }
        val created = CaptureRuntime.retain(this)
        session = created
        retained = true
        created.setPreviewKind(MESH)
        setContent {
            val ui = CaptureRuntime.ui
            LaunchedEffect(ui.phase) {
                if (ui.phase == "idle") finishBack()
            }
            MaterialTheme(colorScheme = Scheme) {
                RefineScreen(
                    ui = ui,
                    onPreview = { view ->
                        preview = view
                        view.onAttached = { attached -> created.attachPreview(attached) }
                    },
                    onPreviewGone = { created.detachPreview(it) },
                    onView = { pose, fx, fy, cx, cy, width, height ->
                        created.setView(pose, fx, fy, cx, cy, width, height)
                    },
                    onSettings = { openSettings() },
                    onBack = { leave() },
                    onSplat = {
                        startActivity(android.content.Intent(this, TrainActivity::class.java))
                        finish()
                    },
                    onSave = {
                        if (!CaptureRuntime.ui.saving) {
                            val stamp = SimpleDateFormat("yyyyMMdd-HHmmss", Locale.US).format(Date())
                            createScan.launch("scan-$stamp.pscan")
                        }
                    },
                    onScans = { openScans(fromRefine = true) },
                    firstPose = { created.firstKeyframePose() },
                    lens = { created.viewLens() },
                    chunkCount = { preview?.chunkCount() ?: 0 },
                    chunkPending = { preview?.chunkPending() ?: 0 },
                    onMeshIdle = { created.cacheReviewMesh() },
                )
            }
        }
    }

    override fun onResume() {
        super.onResume()
        session?.setPreviewKind(MESH)
        preview?.reclaim()
    }

    private fun leave() {
        session?.backToScan()
        finishBack()
    }

    override fun onDestroy() {
        if (retained) {
            retained = false
            CaptureRuntime.release()
            session = null
        }
        super.onDestroy()
    }

    private companion object {
        const val MESH = 2
    }
}
