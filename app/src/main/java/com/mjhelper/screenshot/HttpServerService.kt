package com.mjhelper.screenshot

import android.app.Service
import android.content.Intent
import android.os.IBinder
import fi.iki.elonen.NanoHTTPD
import java.io.ByteArrayInputStream
import java.io.InputStreamReader

class HttpServerService : Service() {

    private var server: NanoHTTPD? = null
    private var helperHtml: String? = null

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

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == "STOP") {
            server?.stop()
            server = null
            return START_NOT_STICKY
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
                        session.uri == "/api/status" -> {
                            val json = """{"running":${ScreenCaptureService.isRunning},"hasScreenshot":${ScreenCaptureService.latestScreenshot != null}}"""
                            newFixedLengthResponse(Response.Status.OK, "application/json", json).apply {
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

    override fun onDestroy() {
        server?.stop()
        server = null
        super.onDestroy()
    }
}
