package com.mjhelper.screenshot

import android.app.Activity
import android.content.Context
import android.content.Intent
import android.media.projection.MediaProjectionManager
import android.os.Bundle
import android.widget.Button
import android.widget.Switch
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity

class MainActivity : AppCompatActivity() {

    private lateinit var tvStatus: TextView
    private lateinit var tvAddress: TextView
    private lateinit var tvHint: TextView
    private lateinit var btnStart: Button
    private lateinit var swAutoStart: Switch
    private var isRunning = false

    private val screenCaptureLauncher = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        if (result.resultCode == Activity.RESULT_OK && result.data != null) {
            val serviceIntent = Intent(this, ScreenCaptureService::class.java).apply {
                putExtra("RESULT_CODE", result.resultCode)
                putExtra("RESULT_DATA", result.data)
                action = "START"
            }
            startForegroundService(serviceIntent)

            val httpIntent = Intent(this, HttpServerService::class.java).apply { action = "START" }
            startService(httpIntent)

            isRunning = true
            updateUI()
            Toast.makeText(this, "🀄 截屏服务已启动！", Toast.LENGTH_SHORT).show()
        } else {
            Toast.makeText(this, "需要授权才能截屏", Toast.LENGTH_SHORT).show()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        tvStatus = findViewById(R.id.tvStatus)
        tvAddress = findViewById(R.id.tvAddress)
        tvHint = findViewById(R.id.tvHint)
        btnStart = findViewById(R.id.btnStart)
        swAutoStart = findViewById(R.id.swAutoStart)

        isRunning = ScreenCaptureService.isRunning
        updateUI()

        btnStart.setOnClickListener {
            if (isRunning) stopServices() else requestScreenCapture()
        }

        swAutoStart.setOnCheckedChangeListener { _, isChecked ->
            getSharedPreferences("mj_helper", Context.MODE_PRIVATE)
                .edit().putBoolean("auto_start", isChecked).apply()
        }
        swAutoStart.isChecked = getSharedPreferences("mj_helper", Context.MODE_PRIVATE)
            .getBoolean("auto_start", false)

        val addr = "http://${getLocalIpAddress()}:8666"
        tvAddress.text = addr
    }

    private fun requestScreenCapture() {
        val mgr = getSystemService(Context.MEDIA_PROJECTION_SERVICE) as MediaProjectionManager
        screenCaptureLauncher.launch(mgr.createScreenCaptureIntent())
    }

    private fun stopServices() {
        startService(Intent(this, ScreenCaptureService::class.java).apply { action = "STOP" })
        startService(Intent(this, HttpServerService::class.java).apply { action = "STOP" })
        isRunning = false
        updateUI()
    }

    private fun updateUI() {
        if (isRunning) {
            tvStatus.text = "✅ 运行中"
            tvStatus.setTextColor(getColor(android.R.color.holo_green_dark))
            btnStart.text = "⏹ 停止截屏"
            tvHint.text = "👉 用Edge浏览器打开上面的地址\n即可使用麻将提示器"
            tvHint.setTextColor(getColor(android.R.color.holo_orange_dark))
        } else {
            tvStatus.text = "⏸ 未启动"
            tvStatus.setTextColor(getColor(android.R.color.darker_gray))
            btnStart.text = "🚀 开始截屏"
            tvHint.text = "点击上方按钮启动截屏服务"
            tvHint.setTextColor(getColor(android.R.color.darker_gray))
        }
    }

    private fun getLocalIpAddress(): String {
        try {
            val en = java.net.NetworkInterface.getNetworkInterfaces()
            while (en.hasMoreElements()) {
                val addrs = en.nextElement().inetAddresses
                while (addrs.hasMoreElements()) {
                    val a = addrs.nextElement()
                    if (!a.isLoopbackAddress && a is java.net.Inet4Address) return a.hostAddress ?: "127.0.0.1"
                }
            }
        } catch (_: Exception) {}
        return "127.0.0.1"
    }

    override fun onResume() {
        super.onResume()
        isRunning = ScreenCaptureService.isRunning
        updateUI()
    }
}
