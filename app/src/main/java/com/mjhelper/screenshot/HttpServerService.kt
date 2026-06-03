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
import org.json.JSONArray
import org.json.JSONObject
import java.io.ByteArrayInputStream
import java.io.InputStreamReader

import java.net.HttpURLConnection
import java.net.URL
import java.io.OutputStream
class HttpServerService : Service() {

    companion object {
        var cloudInferUrl: String = ""  // Cloud inference URL, e.g. "https://xxx.hf.space"
        var useCloudInfer: Boolean = true  // Default: use cloud if available
    }

    private var server: NanoHTTPD? = null
    private var helperHtml: String? = null
    private var detector: MahjongOnnxDetector? = null
    private var modelLoaded = false
    private var modelError = ""
    private var modelBytes: ByteArray? = null
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
            modelBytes = inputStream.readBytes()
            inputStream.close()
            modelLoaded = detector?.loadModel(ByteArrayInputStream(modelBytes!!)) ?: false
            modelError = if (modelLoaded) "" else (detector?.lastError ?: "未知错误")
            
            // CPU加载成功后，尝试NNAPI加速
            if (modelLoaded && modelBytes != null) {
                val nnapiOk = detector?.tryEnableNNAPI(modelBytes!!) ?: false
            }
        } catch (e: Exception) {
            modelLoaded = false
            modelError = "初始化异常: ${e.message}"
        }
    }

    override fun onCreate() {
        super.onCreate()
        createNotificationChannel()
        Thread { initDetector() }.start()
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
                            helperHtml = null
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
            val data = ScreenCaptureService.latestScreenshot
            if (data == null) {
                val json = JSONObject().apply {
                    put("error", "no screenshot")
                    put("tiles", JSONArray())
                    put("names", JSONArray())
                    put("debug", "no_screenshot")
                }.toString()
                newFixedLengthResponse(Response.Status.OK, "application/json", json).apply {
                    addHeader("Access-Control-Allow-Origin", "*")
                }
            } else if (useCloudInfer && cloudInferUrl.isNotEmpty()) {
                // ★ Cloud inference: send screenshot to cloud server
                try {
                    val cloudResult = cloudRecognize(data)
                    if (cloudResult != null) {
                        newFixedLengthResponse(Response.Status.OK, "application/json", cloudResult).apply {
                            addHeader("Access-Control-Allow-Origin", "*")
                            addHeader("Cache-Control", "no-cache, no-store")
                        }
                    } else {
                        // Cloud failed, fall back to local
                        val localResult = localRecognize(data)
                        if (localResult != null) {
                            newFixedLengthResponse(Response.Status.OK, "application/json", localResult).apply {
                                addHeader("Access-Control-Allow-Origin", "*")
                                addHeader("Cache-Control", "no-cache, no-store")
                            }
                        } else {
                            newFixedLengthResponse(Response.Status.OK, "application/json", """{"error":"both cloud and local failed"}""").apply {
                                addHeader("Access-Control-Allow-Origin", "*")
                            }
                        }
                    }
                } catch (e: Exception) {
                    val localResult = localRecognize(data)
                    if (localResult != null) {
                        newFixedLengthResponse(Response.Status.OK, "application/json", localResult).apply {
                            addHeader("Access-Control-Allow-Origin", "*")
                            addHeader("Cache-Control", "no-cache, no-store")
                        }
                    } else {
                        newFixedLengthResponse(Response.Status.OK, "application/json", """{"error":"both cloud and local failed"}""").apply {
                            addHeader("Access-Control-Allow-Origin", "*")
                        }
                    }
                }
            } else {
                val localResult = localRecognize(data)
                if (localResult != null) {
                    newFixedLengthResponse(Response.Status.OK, "application/json", localResult).apply {
                        addHeader("Access-Control-Allow-Origin", "*")
                        addHeader("Cache-Control", "no-cache, no-store")
                    }
                } else {
                    newFixedLengthResponse(Response.Status.OK, "application/json", """{"error":"local inference failed"}""").apply {
                        addHeader("Access-Control-Allow-Origin", "*")
                    }
                }
            }
                        }
                        session.uri == "/api/cloud_status" -> {
                            val json = JSONObject().apply {
                                put("cloudUrl", cloudInferUrl)
                                put("useCloud", useCloudInfer)
                                put("cloudAvailable", cloudInferUrl.isNotEmpty())
                            }.toString()
                            newFixedLengthResponse(Response.Status.OK, "application/json", json).apply {
                                addHeader("Access-Control-Allow-Origin", "*")
                            }
                        }
                        session.uri == "/api/cloud_set" -> {
                            // POST: set cloud URL and toggle
                            try {
                                val body = java.util.HashMap<String, String>()
                                session.parseBody(body)
                                val params = JSONObject(body["postData"] ?: "{}")
                                if (params.has("url")) cloudInferUrl = params.getString("url")
                                if (params.has("use")) useCloudInfer = params.getBoolean("use")
                            } catch (_: Exception) {}
                            val json = JSONObject().apply {
                                put("cloudUrl", cloudInferUrl)
                                put("useCloud", useCloudInfer)
                            }.toString()
                            newFixedLengthResponse(Response.Status.OK, "application/json", json).apply {
                                addHeader("Access-Control-Allow-Origin", "*")
                            }
                        }
                        session.uri == "/api/status" -> {
                            val capture = ScreenCaptureService
                            val timeSinceLast = if (capture.lastCaptureTime > 0) 
                                (System.currentTimeMillis() - capture.lastCaptureTime) / 1000 else -1
                            val nnapi = detector?.useNNAPI ?: false
                            val dbg = detector?.lastDebug ?: ""
                            val panelW = FloatingService.currentPanelWidth
                            val json = JSONObject().apply {
                                put("running", capture.isRunning)
                                put("hasScreenshot", capture.latestScreenshot != null)
                                put("captureCount", capture.captureCount)
                                put("timeSinceLast", timeSinceLast)
                                put("error", capture.lastError)
                                put("modelLoaded", modelLoaded)
                                put("modelError", modelError)
                                put("nnapi", nnapi)
                                put("panelWidth", panelW)
                                put("debug", dbg)
                            }.toString()
                            newFixedLengthResponse(Response.Status.OK, "application/json", json).apply {
                                addHeader("Access-Control-Allow-Origin", "*")
                                addHeader("Cache-Control", "no-cache, no-store")
                            }
                        }
                        session.uri == "/api/reload" -> {
                            Thread {
                                try { detector?.close() } catch (_: Exception) {}
                                detector = null
                                modelLoaded = false
                                modelError = "重新加载中..."
                                initDetector()
                            }.start()
                            newFixedLengthResponse(Response.Status.OK, "application/json", 
                                """{"reloading":true}""").apply {
                                addHeader("Access-Control-Allow-Origin", "*")
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


    private fun cloudRecognize(screenshotData: ByteArray): String? {
        try {
            val url = URL("$cloudInferUrl/api/recognize")
            val conn = url.openConnection() as HttpURLConnection
            conn.requestMethod = "POST"
            conn.doOutput = true
            conn.connectTimeout = 10000
            conn.readTimeout = 15000
            val boundary = "----CloudInfer${System.currentTimeMillis()}"
            conn.setRequestProperty("Content-Type", "multipart/form-data; boundary=$boundary")

            val os = conn.outputStream
            // Write multipart form data
            os.write("--$boundary\r\n".toByteArray())
            os.write("Content-Disposition: form-data; name=\"file\"; filename=\"screenshot.jpg\"\r\n".toByteArray())
            os.write("Content-Type: image/jpeg\r\n\r\n".toByteArray())
            os.write(screenshotData)
            os.write("\r\n--$boundary--\r\n".toByteArray())
            os.flush()
            os.close()

            val responseCode = conn.responseCode
            if (responseCode == 200) {
                val response = conn.inputStream.bufferedReader().readText()
                conn.disconnect()

                // Convert cloud response format to local format
                val cloudJson = JSONObject(response)
                val cloudTiles = cloudJson.optJSONArray("tiles") ?: JSONArray()
                val cloudDetails = cloudJson.optJSONArray("tile_details") ?: JSONArray()
                val cloudDebug = cloudJson.optJSONObject("debug") ?: JSONObject()

                val tilesArray = JSONArray()
                val namesArray = JSONArray()
                for (i in 0 until cloudTiles.length()) {
                    val tileName = cloudTiles.getString(i)
                    val detail = if (i < cloudDetails.length()) cloudDetails.getJSONObject(i) else null
                    tilesArray.put(JSONObject().apply {
                        put("name", tileName)
                        put("short", tileName)
                        put("conf", detail?.optDouble("conf", 0.0) ?: 0.0)
                        put("x", detail?.optInt("x1", 0) ?: 0)
                    })
                    namesArray.put(tileName)
                }

                return JSONObject().apply {
                    put("tiles", tilesArray)
                    put("names", namesArray)
                    put("avgConf", 0.0)
                    put("total", cloudDebug.optInt("after_nms", 0))
                    put("debug", "CLOUD_OK raw=${cloudDebug.optInt("raw_detections",0)} nms=${cloudDebug.optInt("after_nms",0)} hand=${cloudDebug.optInt("hand_count",0)} ${cloudDebug.optString("input_size","")} ${cloudDebug.optInt("elapsed_ms",0)}ms")
                    put("imgSize", cloudDebug.optString("input_size", ""))
                    put("cropRight", cloudDebug.optInt("crop_right", 0))
                    put("inferBackend", "cloud")
                }.toString()
            } else {
                conn.disconnect()
                return null
            }
        } catch (e: Exception) {
            return null
        }
    }

    private fun localRecognize(screenshotData: ByteArray): String? {
        if (!modelLoaded) { initDetector() }
        if (!modelLoaded) {
            return JSONObject().apply {
                put("error", "model not loaded: $modelError")
                put("tiles", JSONArray())
                put("names", JSONArray())
                put("debug", "model_not_loaded")
            }.toString()
        }
        try {
            val bitmap = BitmapFactory.decodeByteArray(screenshotData, 0, screenshotData.size)
            val imgW = bitmap.width
            val imgH = bitmap.height
            val panelWidth = FloatingService.currentPanelWidth
            val isLandscape = imgW > imgH
            val cropRight = if (isLandscape && panelWidth > 0) panelWidth else 0
            val result = detector?.recognize(bitmap, cropRight = cropRight)
            bitmap.recycle()

            if (result != null) {
                val tilesArray = JSONArray()
                for (tile in result.handTiles) {
                    tilesArray.put(JSONObject().apply {
                        put("name", tile.className)
                        put("short", tile.shortName)
                        put("conf", tile.confidence)
                        put("x", tile.centerX.toInt())
                    })
                }
                val namesArray = JSONArray(result.handTileNames)
                return JSONObject().apply {
                    put("tiles", tilesArray)
                    put("names", namesArray)
                    put("avgConf", result.confidence)
                    put("total", result.allDetections.size)
                    put("debug", result.debugInfo)
                    put("imgSize", "${imgW}x${imgH}")
                    put("cropRight", cropRight)
                    put("inferBackend", "local")
                }.toString()
            } else {
                val err = detector?.lastError ?: "recognition failed"
                val dbg = detector?.lastDebug ?: ""
                return JSONObject().apply {
                    put("error", err)
                    put("tiles", JSONArray())
                    put("names", JSONArray())
                    put("debug", dbg)
                }.toString()
            }
        } catch (e: Exception) {
            return JSONObject().apply {
                put("error", e.message ?: "unknown error")
                put("tiles", JSONArray())
                put("names", JSONArray())
                put("debug", "exception")
            }.toString()
        }
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
