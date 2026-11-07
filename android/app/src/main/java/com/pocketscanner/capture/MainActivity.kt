package com.pocketscanner.capture

import android.Manifest
import android.app.GameManager
import android.app.GameState
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.view.WindowManager
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawing
import androidx.compose.foundation.layout.windowInsetsPadding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.ExposedDropdownMenuBox
import androidx.compose.material3.ExposedDropdownMenuDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.MenuAnchorType
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import com.intel.realsense.librealsense.GLRsSurfaceView

class MainActivity : ComponentActivity() {
    private var ui by mutableStateOf(CaptureUi())
    private var session by mutableStateOf<RealSenseSession?>(null)
    private var colorPreview: YuyvPreview? = null
    private var depthPreview: GLRsSurfaceView? = null

    private val cameraPermission = registerForActivityResult(
        ActivityResultContracts.RequestPermission(),
    ) { granted ->
        if (granted) startSession()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        requestGameMode()
        setContent {
            MaterialTheme(colorScheme = darkColorScheme()) {
                CaptureScaffold(
                    ui = ui,
                    onColor = { view ->
                        colorPreview = view
                        session?.attachColorPreview(view)
                    },
                    onDepth = { view ->
                        depthPreview = view
                        session?.attachPreview(view)
                    },
                    onSelect = { session?.selectProfile(it) },
                    onSnapshot = { session?.requestSnapshot() },
                )
            }
        }
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED
        ) {
            startSession()
        } else {
            cameraPermission.launch(Manifest.permission.CAMERA)
        }
    }

    override fun onResume() {
        super.onResume()
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        requestGameMode()
        depthPreview?.onResume()
    }

    private fun requestGameMode() {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.TIRAMISU) return
        val manager = getSystemService(GameManager::class.java) ?: return
        manager.setGameState(GameState(false, GameState.MODE_GAMEPLAY_UNINTERRUPTIBLE))
    }

    override fun onPause() {
        depthPreview?.onPause()
        super.onPause()
    }

    private fun startSession() {
        if (session != null) return
        val created = RealSenseSession(applicationContext) { ui = it }
        session = created
        colorPreview?.let { created.attachColorPreview(it) }
        depthPreview?.let { created.attachPreview(it) }
        created.start()
    }

    override fun onDestroy() {
        session?.stop()
        session = null
        super.onDestroy()
    }
}

private val Ink = Color(0xFF101010)
private val Muted = Color(0xFFBDBDBD)
private val Chip = Color(0xCC000000)

@Composable
private fun CaptureScaffold(
    ui: CaptureUi,
    onColor: (YuyvPreview) -> Unit,
    onDepth: (GLRsSurfaceView) -> Unit,
    onSelect: (String) -> Unit,
    onSnapshot: () -> Unit,
) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(Color.Black)
            .windowInsetsPadding(WindowInsets.safeDrawing)
            .padding(horizontal = 16.dp, vertical = 12.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp),
    ) {
        Text(ui.headline, color = Color.White, style = MaterialTheme.typography.titleMedium)
        if (ui.deviceLine.isNotEmpty()) {
            Text(ui.deviceLine, color = Muted, style = MaterialTheme.typography.bodyMedium)
        }
        val detail = buildString {
            if (ui.streamLine.isNotEmpty()) append(ui.streamLine)
            if (isNotEmpty()) append("   ")
            append("fps ${"%.1f".format(ui.fps)}   отвалы ${ui.detachCount}")
        }
        Text(detail, color = Muted, style = MaterialTheme.typography.bodySmall)
        if (ui.saveLine.isNotEmpty()) {
            Text(ui.saveLine, color = Color.White, style = MaterialTheme.typography.bodyMedium)
        }
        if (ui.profiles.isNotEmpty()) {
            ProfileMenu(
                profiles = ui.profiles,
                selected = ui.selectedProfile.ifEmpty { ui.profiles.first() },
                onSelect = onSelect,
            )
        }
        PreviewCard(title = "Цвет", modifier = Modifier.weight(1f)) {
            AndroidView(
                modifier = Modifier.fillMaxSize(),
                factory = { ctx -> YuyvPreview(ctx).also(onColor) },
            )
        }
        PreviewCard(title = "Глубина", modifier = Modifier.weight(1f)) {
            BoxWithConstraints(
                modifier = Modifier.fillMaxSize(),
                contentAlignment = Alignment.Center,
            ) {
                val frame = 16f / 9f
                val wide = maxWidth / maxHeight > frame
                AndroidView(
                    modifier = if (wide) {
                        Modifier.fillMaxHeight().aspectRatio(frame)
                    } else {
                        Modifier.fillMaxWidth().aspectRatio(frame)
                    },
                    factory = { ctx -> GLRsSurfaceView(ctx).also(onDepth) },
                    onRelease = { view ->
                        view.onPause()
                        view.close()
                    },
                )
            }
        }
        Button(onClick = onSnapshot, modifier = Modifier.fillMaxWidth()) {
            Text("Снять кадр")
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun ProfileMenu(
    profiles: List<String>,
    selected: String,
    onSelect: (String) -> Unit,
) {
    var expanded by remember { mutableStateOf(false) }
    ExposedDropdownMenuBox(
        expanded = expanded,
        onExpandedChange = { expanded = it },
        modifier = Modifier.fillMaxWidth(),
    ) {
        OutlinedTextField(
            value = selected,
            onValueChange = {},
            readOnly = true,
            singleLine = true,
            label = { Text("Профиль") },
            trailingIcon = { ExposedDropdownMenuDefaults.TrailingIcon(expanded = expanded) },
            colors = ExposedDropdownMenuDefaults.outlinedTextFieldColors(
                focusedTextColor = Color.White,
                unfocusedTextColor = Color.White,
                focusedBorderColor = Color.White,
                unfocusedBorderColor = Muted,
                focusedLabelColor = Color.White,
                unfocusedLabelColor = Muted,
            ),
            modifier = Modifier
                .menuAnchor(MenuAnchorType.PrimaryNotEditable)
                .fillMaxWidth(),
        )
        ExposedDropdownMenu(
            expanded = expanded,
            onDismissRequest = { expanded = false },
        ) {
            profiles.forEach { label ->
                DropdownMenuItem(
                    text = { Text(label) },
                    onClick = {
                        expanded = false
                        if (label != selected) onSelect(label)
                    },
                )
            }
        }
    }
}

@Composable
private fun PreviewCard(
    title: String,
    modifier: Modifier,
    content: @Composable () -> Unit,
) {
    Box(
        modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(18.dp))
            .background(Ink),
    ) {
        content()
        Text(
            title,
            color = Color.White,
            style = MaterialTheme.typography.labelLarge,
            modifier = Modifier
                .align(Alignment.TopStart)
                .padding(10.dp)
                .background(Chip, RoundedCornerShape(8.dp))
                .padding(horizontal = 10.dp, vertical = 4.dp),
        )
    }
}
