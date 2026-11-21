package com.pocketscanner.capture

import android.app.Activity
import android.content.Context
import android.content.Intent
import android.os.Build
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue

internal object CaptureRuntime {
    var ui by mutableStateOf(CaptureUi())
    var session: RealSenseSession? = null
        private set
    private var refs = 0

    fun retain(context: Context): RealSenseSession {
        refs++
        val existing = session
        if (existing != null) return existing
        val created = RealSenseSession(context.applicationContext) { ui = it }
        session = created
        created.start()
        return created
    }

    fun release() {
        refs = (refs - 1).coerceAtLeast(0)
        if (refs > 0) return
        session?.stop()
        session = null
    }
}

internal fun Activity.openRefine() {
    startMotion(Intent(this, RefineActivity::class.java), R.anim.enter_from_right, R.anim.exit_to_left)
}

internal fun Activity.openSettings() {
    startMotion(Intent(this, SettingsActivity::class.java), R.anim.enter_from_bottom, R.anim.fade_hold)
}

internal fun Activity.finishBack() {
    finishMotion(R.anim.enter_from_left, R.anim.exit_to_right)
}

internal fun Activity.finishSettings() {
    finishMotion(R.anim.fade_hold, R.anim.exit_to_bottom)
}

private fun Activity.startMotion(intent: Intent, enter: Int, exit: Int) {
    if (Build.VERSION.SDK_INT >= 34) {
        overrideActivityTransition(Activity.OVERRIDE_TRANSITION_OPEN, enter, exit)
    }
    startActivity(intent)
    if (Build.VERSION.SDK_INT < 34) {
        @Suppress("DEPRECATION")
        overridePendingTransition(enter, exit)
    }
}

private fun Activity.finishMotion(enter: Int, exit: Int) {
    if (Build.VERSION.SDK_INT >= 34) {
        overrideActivityTransition(Activity.OVERRIDE_TRANSITION_CLOSE, enter, exit)
    }
    finish()
    if (Build.VERSION.SDK_INT < 34) {
        @Suppress("DEPRECATION")
        overridePendingTransition(enter, exit)
    }
}
