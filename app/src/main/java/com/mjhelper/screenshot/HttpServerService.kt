package com.mjhelper.screenshot

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Build
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import fi.iki.elonen.NanoHTTPD
import java.io.ByteArrayInputStream
import java.io.InputStreamReader

class HttpServerService : Service() {

    private var server: NanoHTTPD? = null
    private var helperHtml: String? = null
    private var detector: MahjongOnnxDetector? = null
    private var modelLoaded = false
    private val CHANNEL_ID = "mj_http"
    private val NOTIFICATION_ID = 3
    private val handler = Handler(Looper.getMainLooper())

    override fun onBind(intent: Intent?): IBinder? = null

    private fun loadHelperHtml(): String {
        if (helperHtml == null) {
            try {
                val is_ = assets.open("helper.html")
                val reader = InputStreamReader(is_, "UTF-8")
                helperHtml = reader.readText()
                reader.close()
            } catch (e: Exception) {
                helperHtml = "<html><body><h2>提示器页面加载失败</h2><p>${e.message}</p></body></html>"
            }
        }
        return helperHtml ?: ""
    }

    private fun initDetector() {
        if (detector != null && modelLoaded) return
        try {
            detector = MahjongOnnxDetector(this)
            val inputStream = assets.open("mahjong-yolon-best.onnx")
            modelLoaded = detector?.loadModel(inputStream) ?: false
            inputStream.close()
        } catch (e: Exception) {
            modelLoaded = false
        }
    }

    override fun onCreate() {
        super.onCreate()
        createNotificationChannel()
        // 在后台线程初始化模型
        Thread {
            initDetector()
        }.start()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == "STOP") {
            server?.stop()
            server = null
            stopSelf()
            return START_NOT_STICKY
        }

        val notification = createNotification()
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.UPSIDE_DOWN_CAKE) {
            startForeground(NOTIFICATION_ID, notification, android.content.pm.ServiceInfo.FOREGROUND_SERVICE_TYPE_SPECIAL_USE)
        } else {
            startForeground(NOTIFICATION_ID, notification)
        }

        if (server == null) {
            server = object : NanoHTTPD(8666) {
                override fun serve(session: IHTTPSession): Response {
                    return when {
                        session.uri == "/" || session.uri == "/helper" || session.uri == "/index.html" -> {
                            val html = loadHelperHtml()
                            newFixedLengthResponse(Response.Status.OK, "text/html; charset=utf-8", html).apply {
                                addHeader("Access-Control-Allow-Origin", "*")
                                addHeader("Cache-Control", "no-cache, no-store")
                            }
                        }
                        session.uri == "/api/screenshot" -> {
                            val data = ScreenCaptureService.latestScreenshot
                            if (data != null) {
                                newFixedLengthResponse(
                                    Response.Status.OK,
                                    "image/jpeg",
                                    ByteArrayInputStream(data),
                                    data.size.toLong()
                                ).apply {
                                    addHeader("Access-Control-Allow-Origin", "*")
                                    addHeader("Cache-Control", "no-cache, no-store")
                                }
                            } else {
                                newFixedLengthResponse(Response.Status.NOT_FOUND, MIME_PLAINTEXT, "no screenshot yet").apply {
                                    addHeader("Access-Control-Allow-Origin", "*")
                                }
                            }
                        }
                        session.uri == "/api/recognize" -> {
                            // YOLO识别接口 - 返回识别到的手牌
                            if (!modelLoaded) {
                                initDetector()
                            }
                            val data = ScreenCaptureService.latestScreenshot
                            if (data == null) {
                                newFixedLengthResponse(Response.Status.OK, "application/json", 
                                    """{"error":"no screenshot","tiles":[],"names":[]}""").apply {
                                    addHeader("Access-Control-Allow-Origin", "*")
                                }
                            } else if (!modelLoaded) {
                                newFixedLengthResponse(Response.Status.OK, "application/json",
                                    """{"error":"model not loaded","tiles":[],"names":[]}""").apply {
                                    addHeader("Access-Control-Allow-Origin", "*")
                                }
                            } else {
                                try {
                                    val bitmap = BitmapFactory.decodeByteArray(data, 0, data.size)
                                    val result = detector?.recognize(bitmap)
                                    bitmap.recycle()
                                    
                                    if (result != null) {
                                        val tilesJson = result.handTiles.joinToString(",") { tile ->
                                            """{"name":"${tile.className}","short":"${tile.shortName}","conf":${"%.2f".format(tile.confidence)},"x":${tile.centerX.toInt()}}"""
                                        }
                                        val json = """{"tiles":[$tilesJson],"names":${result.handTileNames},"avgConf":${"%.2f".format(result.confidence)},"total":${result.allDetections.size}}"""
                                        newFixedLengthResponse(Response.Status.OK, "application/json", json).apply {
                                            addHeader("Access-Control-Allow-Origin", "*")
                                            addHeader("Cache-Control", "no-cache, no-store")
                                        }
                                    } else {
                                        newFixedLengthResponse(Response.Status.OK, "application/json",
                                            """{"error":"recognition failed","tiles":[],"names":[]}""").apply {
                                            addHeader("Access-Control-Allow-Origin", "*")
                                        }
                                    }
                                } catch (e: Exception) {
                                    newFixedLengthResponse(Response.Status.OK, "application/json",
                                        """{"error":"${e.message}","tiles":[],"names":[]}""").apply {
                                        addHeader("Access-Control-Allow-Origin", "*")
                                    }
                                }
                            }
                        }
                        session.uri == "/api/status" -> {
                            val capture = ScreenCaptureService
                            val timeSinceLast = if (capture.lastCaptureTime > 0) 
                                (System.currentTimeMillis() - capture.lastCaptureTime) / 1000 else -1
                            val json = """{"running":${capture.isRunning},"hasScreenshot":${capture.latestScreenshot != null},"captureCount":${capture.captureCount},"timeSinceLast":${timeSinceLast},"error":"${capture.lastError}","modelLoaded":$modelLoaded}"""
                            newFixedLengthResponse(Response.Status.OK, "application/json", json).apply {
                                addHeader("Access-Control-Allow-Origin", "*")
                                addHeader("Cache-Control", "no-cache, no-store")
                            }
                        }
                        else -> newFixedLengthResponse(Response.Status.NOT_FOUND, MIME_PLAINTEXT, "not found")
                    }
                }
            }
            try {
                server?.start()
            } catch (e: Exception) {
                e.printStackTrace()
            }
        }
        return START_STICKY
    }

    override fun onDestroy() {
        server?.stop()
        server = null
        detector?.close()
        detector = null
        modelLoaded = false
        super.onDestroy()
    }

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CHANNEL_ID, "麻将HTTP服务", NotificationManager.IMPORTANCE_LOW
            ).apply { description = "HTTP服务运行中" }
            val manager = getSystemService(NotificationManager::class.java)
            manager.createNotificationChannel(channel)
        }
    }

    private fun createNotification(): Notification {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            Notification.Builder(this, CHANNEL_ID)
                .setContentTitle("🀄 麻将截屏助手")
                .setContentText("HTTP服务运行中")
                .setSmallIcon(android.R.drawable.ic_menu_share)
                .setOngoing(true)
                .build()
        } else {
            @Suppress("DEPRECATION")
            Notification.Builder(this)
                .setContentTitle("🀄 麻将截屏助手")
                .setContentText("HTTP服务运行中")
                .setSmallIcon(android.R.drawable.ic_menu_share)
                .setOngoing(true)
                .build()
        }
    }
}
