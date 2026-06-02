package com.mjhelper.screenshot

import android.os.Bundle
import android.util.Log
import android.webkit.WebChromeClient
import android.webkit.WebSettings
import android.webkit.WebView
import android.webkit.WebViewClient
import android.widget.FrameLayout
import androidx.appcompat.app.AppCompatActivity

class HelperActivity : AppCompatActivity() {

    private val TAG = "MJHelper"
    private var webView: WebView? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        try {
            val layout = FrameLayout(this)
            setContentView(layout)

            val wv = WebView(this)
            layout.addView(wv, FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT,
                FrameLayout.LayoutParams.MATCH_PARENT
            ))
            webView = wv

            val settings = wv.settings
            settings.javaScriptEnabled = true
            settings.domStorageEnabled = true
            settings.allowFileAccess = true
            settings.mixedContentMode = WebSettings.MIXED_CONTENT_ALWAYS_ALLOW
            settings.cacheMode = WebSettings.LOAD_NO_CACHE
            settings.useWideViewPort = true
            settings.loadWithOverviewMode = true
            settings.setSupportZoom(false)
            settings.builtInZoomControls = false

            wv.webViewClient = object : WebViewClient() {
                override fun onReceivedError(view: WebView?, errorCode: Int, description: String?, failingUrl: String?) {
                    Log.e(TAG, "WebView error: $errorCode $description at $failingUrl")
                    // Retry after 2 seconds if server not ready yet
                    if (errorCode == -2 || errorCode == -6) {
                        wv.postDelayed({ wv.loadUrl("http://127.0.0.1:8666") }, 2000)
                    }
                }
            }
            wv.webChromeClient = WebChromeClient()

            // Load the helper page from local HTTP server
            wv.loadUrl("http://127.0.0.1:8666")
        } catch (e: Exception) {
            Log.e(TAG, "onCreate error", e)
            finish()
        }
    }

    override fun onBackPressed() {
        val wv = webView
        if (wv != null && wv.canGoBack()) {
            wv.goBack()
        } else {
            super.onBackPressed()
        }
    }

    override fun onDestroy() {
        try {
            webView?.destroy()
        } catch (e: Exception) {
            Log.e(TAG, "destroy error", e)
        }
        super.onDestroy()
    }
}
