package com.mjhelper.screenshot

import android.annotation.SuppressLint
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Intent
import android.content.pm.ServiceInfo
import android.content.res.Configuration
import android.graphics.PixelFormat
import android.os.Build
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.util.DisplayMetrics
import android.view.Gravity
import android.view.MotionEvent
import android.view.View
import android.view.WindowManager
import android.webkit.JavascriptInterface
import android.webkit.WebChromeClient
import android.webkit.WebSettings
import android.webkit.WebView
import android.webkit.WebViewClient
import android.widget.FrameLayout
import android.widget.LinearLayout
import android.widget.TextView

class FloatingService : Service() {

    companion object {
        var isRunning = false
        private const val CHANNEL_ID = "mj_floating"
        private const val NOTIFICATION_ID = 2
    }

    private var windowManager: WindowManager? = null
    private var floatingView: View? = null
    private var webView: WebView? = null
    private var tvStatus: TextView? = null
    private val handler = Handler(Looper.getMainLooper())
    private var isExpanded = true

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == "STOP") {
            stopSelf()
            return START_NOT_STICKY
        }
        return START_STICKY
    }

    override fun onCreate() {
        super.onCreate()
        createNotificationChannel()
        val notification = createNotification()
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.UPSIDE_DOWN_CAKE) {
            startForeground(NOTIFICATION_ID, notification, ServiceInfo.FOREGROUND_SERVICE_TYPE_SPECIAL_USE)
        } else {
            startForeground(NOTIFICATION_ID, notification)
        }
        isRunning = true
        showFloatingWindow()
    }

    override fun onDestroy() {
        isRunning = false
        handler.removeCallbacksAndMessages(null)
        try {
            webView?.destroy()
        } catch (_: Exception) {}
        try {
            floatingView?.let { windowManager?.removeView(it) }
        } catch (_: Exception) {}
        super.onDestroy()
    }

    override fun onConfigurationChanged(newConfig: Configuration) {
        super.onConfigurationChanged(newConfig)
        // Resize floating window when screen rotates
        handler.postDelayed({ resizeFloatingWindow() }, 500)
    }

    private fun resizeFloatingWindow() {
        try {
            val params = floatingView?.layoutParams as? WindowManager.LayoutParams ?: return

            val realW: Int
            val realH: Int
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
                val bounds = windowManager?.currentWindowMetrics?.bounds
                realW = bounds?.width() ?: 1080
                realH = bounds?.height() ?: 2344
            } else {
                val dm = DisplayMetrics()
                @Suppress("DEPRECATION")
                windowManager?.defaultDisplay?.getRealMetrics(dm)
                realW = dm.widthPixels
                realH = dm.heightPixels
            }
            val screenWidth = realW
            val screenHeight = realH
            val isLandscape = screenWidth > screenHeight

            if (isLandscape) {
                if (isExpanded) {
                    params.width = (screenWidth * 0.12).toInt().coerceIn(200, 320)
                    params.height = (screenHeight * 0.6).toInt()
                } else {
                    params.width = 120
                    params.height = WindowManager.LayoutParams.WRAP_CONTENT
                }
                params.gravity = Gravity.END or Gravity.TOP
                params.x = 0
                params.y = (screenHeight * 0.1).toInt()
            } else {
                if (isExpanded) {
                    params.width = (screenWidth * 0.7).toInt()
                    params.height = (params.width * 0.9).toInt()
                } else {
                    params.width = (screenWidth * 0.4).toInt()
                    params.height = WindowManager.LayoutParams.WRAP_CONTENT
                }
                params.gravity = Gravity.TOP or Gravity.CENTER_HORIZONTAL
                params.x = 0
                params.y = 40
            }

            windowManager?.updateViewLayout(floatingView, params)
        } catch (e: Exception) {}
    }

    @SuppressLint("SetJavaScriptEnabled", "ClickableViewAccessibility")
    private fun showFloatingWindow() {
        windowManager = getSystemService(WINDOW_SERVICE) as WindowManager

        // Get REAL screen size (respecting current rotation)
        val realW: Int
        val realH: Int
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            val bounds = windowManager?.currentWindowMetrics?.bounds
            realW = bounds?.width() ?: 1080
            realH = bounds?.height() ?: 2344
        } else {
            val dm = DisplayMetrics()
            @Suppress("DEPRECATION")
            windowManager?.defaultDisplay?.getRealMetrics(dm)
            realW = dm.widthPixels
            realH = dm.heightPixels
        }
        val screenWidth = realW
        val screenHeight = realH
        val isLandscape = screenWidth > screenHeight

        // Mini floating panel
        val container = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(0xDD1c2333.toInt()) // semi-transparent dark
            setPadding(8, 6, 8, 6)
        }

        // Top drag bar with status
        val topBar = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
        }

        tvStatus = TextView(this).apply {
            text = "🀄 提示器 v17"
            setTextColor(0xFFe8edf5.toInt())
            textSize = 12f
            setPadding(8, 4, 8, 4)
            setBackgroundColor(0xFF5b21b6.toInt())
        }

        val tvCollapse = TextView(this).apply {
            text = "  ▼  "
            setTextColor(0xFFf59e0b.toInt())
            textSize = 14f
            setPadding(4, 2, 4, 2)
            setOnClickListener {
                toggleExpand()
            }
        }

        topBar.addView(tvStatus, LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f))
        topBar.addView(tvCollapse)
        container.addView(topBar)

        // WebView for the helper (compact)
        val wv = WebView(this)
        webView = wv
        val wvParams = LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.MATCH_PARENT,
            0,
            1f
        )
        container.addView(wv, wvParams)

        wv.settings.apply {
            javaScriptEnabled = true
            domStorageEnabled = true
            allowFileAccess = true
            mixedContentMode = WebSettings.MIXED_CONTENT_ALWAYS_ALLOW
            cacheMode = WebSettings.LOAD_NO_CACHE
            setSupportZoom(false)
            builtInZoomControls = false
        }
        wv.webViewClient = object : WebViewClient() {
            override fun onReceivedError(view: WebView?, errorCode: Int, description: String?, failingUrl: String?) {
                if (errorCode == -2 || errorCode == -6) {
                    wv.postDelayed({ wv.loadUrl("http://127.0.0.1:8666") }, 2000)
                }
            }
        }
        wv.webChromeClient = WebChromeClient()

        // Add JS interface for status updates
        wv.addJavascriptInterface(object : Any() {
            @JavascriptInterface
            fun updateStatus(text: String) {
                handler.post {
                    tvStatus?.text = text
                }
            }
        }, "AndroidStatus")

        wv.loadUrl("http://127.0.0.1:8666")

        floatingView = container

        // Window params - sidebar for landscape, compact for portrait
        var winW: Int
        var winH: Int
        var winGravity: Int
        if (isLandscape) {
            // Landscape: narrow right sidebar, compact height
            winW = (screenWidth * 0.12).toInt().coerceIn(200, 320)
            winH = (screenHeight * 0.6).toInt()
            winGravity = Gravity.END or Gravity.TOP
        } else {
            // Portrait: compact floating card at top
            winW = (screenWidth * 0.7).toInt()
            winH = (winW * 0.9).toInt()
            winGravity = Gravity.TOP or Gravity.CENTER_HORIZONTAL
        }

        val params = WindowManager.LayoutParams(
            winW,
            winH,
            WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY,
            WindowManager.LayoutParams.FLAG_NOT_TOUCH_MODAL or
                WindowManager.LayoutParams.FLAG_LAYOUT_NO_LIMITS,
            PixelFormat.TRANSLUCENT
        ).apply {
            gravity = winGravity
            x = 0
            y = if (isLandscape) (screenHeight * 0.1).toInt() else 40
        }

        // Make the top bar draggable
        var initialX = 0
        var initialY = 0
        var initialTouchX = 0f
        var initialTouchY = 0f
        var isDragging = false

        topBar.setOnTouchListener { _, event ->
            when (event.action) {
                MotionEvent.ACTION_DOWN -> {
                    initialX = params.x
                    initialY = params.y
                    initialTouchX = event.rawX
                    initialTouchY = event.rawY
                    isDragging = false
                    true
                }
                MotionEvent.ACTION_MOVE -> {
                    val dx = event.rawX - initialTouchX
                    val dy = event.rawY - initialTouchY
                    if (Math.abs(dx) > 10 || Math.abs(dy) > 10) isDragging = true
                    params.x = initialX + dx.toInt()
                    params.y = initialY + dy.toInt()
                    windowManager?.updateViewLayout(floatingView, params)
                    true
                }
                MotionEvent.ACTION_UP -> {
                    !isDragging // consume if wasn't dragging
                }
                else -> false
            }
        }

        windowManager?.addView(floatingView, params)
    }

    private fun toggleExpand() {
        isExpanded = !isExpanded
        webView?.visibility = if (isExpanded) View.VISIBLE else View.GONE

        val realW: Int
        val realH: Int
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            val bounds = windowManager?.currentWindowMetrics?.bounds
            realW = bounds?.width() ?: 1080
            realH = bounds?.height() ?: 2344
        } else {
            val dm = DisplayMetrics()
            @Suppress("DEPRECATION")
            windowManager?.defaultDisplay?.getRealMetrics(dm)
            realW = dm.widthPixels
            realH = dm.heightPixels
        }
        val screenWidth = realW
        val screenHeight = realH
        val isLandscape = screenWidth > screenHeight

        val params = floatingView?.layoutParams as? WindowManager.LayoutParams ?: return
        if (isExpanded) {
            if (isLandscape) {
                params.width = (screenWidth * 0.12).toInt().coerceIn(200, 320)
                params.height = (screenHeight * 0.6).toInt()
                params.gravity = Gravity.END or Gravity.TOP
                params.y = (screenHeight * 0.1).toInt()
            } else {
                params.width = (screenWidth * 0.7).toInt()
                params.height = (params.width * 0.9).toInt()
                params.gravity = Gravity.TOP or Gravity.CENTER_HORIZONTAL
                params.y = 40
            }
        } else {
            params.width = if (isLandscape) 120 else (screenWidth * 0.4).toInt()
            params.height = WindowManager.LayoutParams.WRAP_CONTENT
        }
        windowManager?.updateViewLayout(floatingView, params)
    }

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CHANNEL_ID, "麻将悬浮窗", NotificationManager.IMPORTANCE_LOW
            ).apply { description = "悬浮提示器运行中" }
            val manager = getSystemService(NotificationManager::class.java)
            manager.createNotificationChannel(channel)
        }
    }

    private fun createNotification(): Notification {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            Notification.Builder(this, CHANNEL_ID)
                .setContentTitle("🀄 捉鸡麻将提示器")
                .setContentText("悬浮窗运行中 - 切到微乐麻将使用")
                .setSmallIcon(android.R.drawable.ic_menu_compass)
                .setOngoing(true)
                .build()
        } else {
            @Suppress("DEPRECATION")
            Notification.Builder(this)
                .setContentTitle("🀄 捉鸡麻将提示器")
                .setContentText("悬浮窗运行中")
                .setSmallIcon(android.R.drawable.ic_menu_compass)
                .setOngoing(true)
                .build()
        }
    }
}
