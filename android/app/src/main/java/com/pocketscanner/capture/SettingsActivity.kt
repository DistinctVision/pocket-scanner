package com.pocketscanner.capture

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.addCallback
import androidx.activity.compose.setContent
import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue

class SettingsActivity : ComponentActivity() {
    private var retained = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        onBackPressedDispatcher.addCallback(this) { finishSettings() }
        val session = CaptureRuntime.retain(this)
        retained = true
        setContent {
            val ui = CaptureRuntime.ui
            val applied = remember { session.scanSettings() }
            var frontier by remember { mutableStateOf(session.frontierEnabled()) }
            MaterialTheme(colorScheme = Scheme) {
                SettingsPanel(
                    ui = ui,
                    applied = applied,
                    showFrontier = frontier,
                    onFrontier = {
                        frontier = it
                        session.setFrontier(it)
                    },
                    onSelect = { session.selectProfile(it) },
                    onApply = {
                        session.applyScan(it)
                        finishSettings()
                    },
                    onReset = {
                        session.applyScan(session.scanSettings())
                        finishSettings()
                    },
                    onRefine = { loops, voxel -> session.updateRefine(loops, voxel) },
                    onSplatting = { epochs, validation, percent -> session.setSplatting(epochs, validation, percent) },
                    onClose = { finishSettings() },
                )
            }
        }
    }

    override fun onDestroy() {
        if (retained) {
            retained = false
            CaptureRuntime.release()
        }
        super.onDestroy()
    }
}
