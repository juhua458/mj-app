package com.mjhelper.screenshot

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.graphics.Bitmap
import android.graphics.PixelFormat
import android.hardware.display.DisplayManager
import android.hardware.display.VirtualDisplay
import android.media.Image
import android.media.ImageReader
import android.media.projection.MediaProjection
import android.media.projection.MediaProjectionManager
import android.os.Build
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.util.DisplayMetrics
import android.view.WindowManager
import java.io.ByteArrayOutputStream

class ScreenCaptureService : Service() {

    companion object {
        var isRunning = false
        var latestScreenshot: ByteArray? = null
            private set
        var captureCount: Int = 0
            private set
        var lastCaptureTime: Long = 0
            private set
        var lastError: String = ""
            private set
        private const val CHANNEL_ID = "mj_screenshot"
        private const val NOTIFICATION_ID = 1
    }

    private var mediaProjection: MediaProjection? = null
    private var virtualDisplay: VirtualDisplay? = null
    private var imageReader: ImageReader? = null
    private var resultCode: Int = 0
    private var resultData: Intent? = null
    private val handler = Handler(Looper.getMainLooper())
    private var screenWidth = 540
    private var screenHeight = 1200
    private var screenDensity = 160
    private var consecutiveFails = 0

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onCreate() {
        super.onCreate()
        createNotificationChannel()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == "STOP") {
            stopSelf()
            return START_NOT_STICKY
        }

        // 如果已经在运行，不重复启动
        if (isRunning && mediaProjection != null) {
            return START_STICKY
        }

        resultCode = intent?.getIntExtra("RESULT_CODE", 0) ?: 0
        resultData = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            intent?.getParcelableExtra("RESULT_DATA", Intent::class.java)
        } else {
            @Suppress("DEPRECATION")
            intent?.getParcelableExtra("RESULT_DATA")
        }

        // 获取屏幕尺寸（setupVirtualDisplay会重新获取，这里只初始化）
        // screenWidth/screenHeight 会在 setupVirtualDisplay 中正确设置

        val notification = createNotification()
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.UPSIDE_DOWN_CAKE) {
            startForeground(
                NOTIFICATION_ID, notification,
                ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PROJECTION
            )
        } else {
            startForeground(NOTIFICATION_ID, notification)
        }

        startScreenCapture()
        isRunning = true

        return START_STICKY
    }

    private fun startScreenCapture() {
        try {
            val projectionManager = getSystemService(Context.MEDIA_PROJECTION_SERVICE) as MediaProjectionManager
            mediaProjection = projectionManager.getMediaProjection(resultCode, resultData!!)
            
            mediaProjection?.registerCallback(object : MediaProjection.Callback() {
                override fun onStop() {
                    lastError = "MediaProjection被停止"
                    isRunning = false
                }
            }, handler)

            setupVirtualDisplay()

            lastError = ""
            captureCount = 0

            // 每800ms截一次屏
            handler.postDelayed(object : Runnable {
                override fun run() {
                    if (isRunning) {
                        checkAndRecreateDisplay()
                        captureScreen()
                        handler.postDelayed(this, 800)
                    }
                }
            }, 500)
        } catch (e: Exception) {
            lastError = "启动失败: ${e.message}"
            e.printStackTrace()
            isRunning = false
            stopSelf()
        }
    }

    private fun setupVirtualDisplay() {
        val wm = getSystemService(Context.WINDOW_SERVICE) as WindowManager
        val metrics = DisplayMetrics()
        @Suppress("DEPRECATION")
        wm.defaultDisplay.getMetrics(metrics)
        val newWidth = metrics.widthPixels * 3 / 4
        val newHeight = metrics.heightPixels * 3 / 4
        val newDensity = metrics.densityDpi * 3 / 4

        screenWidth = newWidth
        screenHeight = newHeight
        screenDensity = newDensity

        imageReader = ImageReader.newInstance(screenWidth, screenHeight, PixelFormat.RGBA_8888, 2)
        virtualDisplay = mediaProjection?.createVirtualDisplay(
            "MJScreenCapture",
            screenWidth, screenHeight, screenDensity,
            DisplayManager.VIRTUAL_DISPLAY_FLAG_AUTO_MIRROR,
            imageReader?.surface,
            null, null
        )
    }

    private fun checkAndRecreateDisplay() {
        try {
            val wm = getSystemService(Context.WINDOW_SERVICE) as WindowManager
            val metrics = DisplayMetrics()
            @Suppress("DEPRECATION")
            wm.defaultDisplay.getMetrics(metrics)
            val newWidth = metrics.widthPixels * 3 / 4
            val newHeight = metrics.heightPixels * 3 / 4

            // If dimensions changed (screen rotated), recreate VirtualDisplay
            if (newWidth != screenWidth || newHeight != screenHeight) {
                try {
                    virtualDisplay?.release()
                } catch (_: Exception) {}
                try {
                    imageReader?.close()
                } catch (_: Exception) {}

                screenWidth = newWidth
                screenHeight = newHeight
                screenDensity = metrics.densityDpi * 3 / 4

                imageReader = ImageReader.newInstance(screenWidth, screenHeight, PixelFormat.RGBA_8888, 2)
                virtualDisplay = mediaProjection?.createVirtualDisplay(
                    "MJScreenCapture",
                    screenWidth, screenHeight, screenDensity,
                    DisplayManager.VIRTUAL_DISPLAY_FLAG_AUTO_MIRROR,
                    imageReader?.surface,
                    null, null
                )
            }
        } catch (e: Exception) {
            lastError = "重建显示失败: ${e.message}"
        }
    }

    private fun captureScreen() {
        try {
            val image: Image? = imageReader?.acquireLatestImage()
            if (image != null) {
                val planes = image.planes
                val buffer = planes[0].buffer
                val pixelStride = planes[0].pixelStride
                val rowStride = planes[0].rowStride
                val rowPadding = rowStride - pixelStride * screenWidth

                val bitmap = Bitmap.createBitmap(
                    screenWidth + rowPadding / pixelStride,
                    screenHeight, Bitmap.Config.ARGB_8888
                )
                bitmap.copyPixelsFromBuffer(buffer)
                image.close()

                val stream = ByteArrayOutputStream()
                bitmap.compress(Bitmap.CompressFormat.JPEG, 75, stream)
                latestScreenshot = stream.toByteArray()
                bitmap.recycle()
                
                captureCount++
                lastCaptureTime = System.currentTimeMillis()
                consecutiveFails = 0
                lastError = ""
            } else {
                consecutiveFails++
                if (consecutiveFails > 10) {
                    lastError = "连续${consecutiveFails}次未获取到帧"
                }
            }
        } catch (e: Exception) {
            consecutiveFails++
            lastError = "截屏错误: ${e.message}"
        }
    }

    override fun onDestroy() {
        isRunning = false
        handler.removeCallbacksAndMessages(null)
        virtualDisplay?.release()
        imageReader?.close()
        mediaProjection?.stop()
        latestScreenshot = null
        super.onDestroy()
    }

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CHANNEL_ID, "麻将截屏", NotificationManager.IMPORTANCE_LOW
            ).apply { description = "后台截屏服务运行中" }
            val manager = getSystemService(NotificationManager::class.java)
            manager.createNotificationChannel(channel)
        }
    }

    private fun createNotification(): Notification {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            Notification.Builder(this, CHANNEL_ID)
                .setContentTitle("🀄 麻将截屏助手")
                .setContentText("截屏服务运行中")
                .setSmallIcon(android.R.drawable.ic_menu_camera)
                .setOngoing(true)
                .build()
        } else {
            @Suppress("DEPRECATION")
            Notification.Builder(this)
                .setContentTitle("🀄 麻将截屏助手")
                .setContentText("截屏服务运行中")
                .setSmallIcon(android.R.drawable.ic_menu_camera)
                .setOngoing(true)
                .build()
        }
    }
}