package com.mjhelper.screenshot

import android.app.Activity
import android.content.Context
import android.content.Intent
import android.media.projection.MediaProjectionManager
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.Switch
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity

class MainActivity : AppCompatActivity() {

    private val TAG = "MJMainActivity"

    private lateinit var tvStatus: TextView
    private lateinit var tvHint: TextView
    private lateinit var btnStart: Button
    private lateinit var btnHelper: Button
    private var isRunning = false

    private val screenCaptureLauncher = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        try {
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
                Toast.makeText(this, "🀄 截屏已启动！", Toast.LENGTH_SHORT).show()

                // Auto-open helper after short delay
                btnHelper.postDelayed({
                    try {
                        startActivity(Intent(this, HelperActivity::class.java))
                    } catch (e: Exception) {
                        Log.e(TAG, "Failed to open helper", e)
                    }
                }, 800)
            } else {
                Toast.makeText(this, "需要授权才能截屏", Toast.LENGTH_SHORT).show()
            }
        } catch (e: Exception) {
            Log.e(TAG, "Capture callback error", e)
            Toast.makeText(this, "启动出错: ${e.message}", Toast.LENGTH_LONG).show()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        try {
            setContentView(R.layout.activity_main)

            tvStatus = findViewById(R.id.tvStatus)
            tvHint = findViewById(R.id.tvHint)
            btnStart = findViewById(R.id.btnStart)
            btnHelper = findViewById(R.id.btnHelper)
            val swAutoStart: Switch = findViewById(R.id.swAutoStart)

            isRunning = ScreenCaptureService.isRunning
            updateUI()

            btnStart.setOnClickListener {
                try {
                    if (isRunning) stopServices() else requestScreenCapture()
                } catch (e: Exception) {
                    Log.e(TAG, "Btn click error", e)
                }
            }

            btnHelper.setOnClickListener {
                try {
                    startActivity(Intent(this, HelperActivity::class.java))
                } catch (e: Exception) {
                    Log.e(TAG, "Helper error", e)
                    Toast.makeText(this, "打开提示器失败: ${e.message}", Toast.LENGTH_LONG).show()
                }
            }

            swAutoStart.setOnCheckedChangeListener { _, isChecked ->
                getSharedPreferences("mj_helper", Context.MODE_PRIVATE)
                    .edit().putBoolean("auto_start", isChecked).apply()
            }
            swAutoStart.isChecked = getSharedPreferences("mj_helper", Context.MODE_PRIVATE)
                .getBoolean("auto_start", false)
        } catch (e: Exception) {
            Log.e(TAG, "onCreate error", e)
        }
    }

    private fun requestScreenCapture() {
        try {
            val mgr = getSystemService(Context.MEDIA_PROJECTION_SERVICE) as MediaProjectionManager
            screenCaptureLauncher.launch(mgr.createScreenCaptureIntent())
        } catch (e: Exception) {
            Log.e(TAG, "Request capture error", e)
            Toast.makeText(this, "请求截屏失败: ${e.message}", Toast.LENGTH_LONG).show()
        }
    }

    private fun stopServices() {
        try {
            startService(Intent(this, ScreenCaptureService::class.java).apply { action = "STOP" })
            startService(Intent(this, HttpServerService::class.java).apply { action = "STOP" })
            isRunning = false
            updateUI()
        } catch (e: Exception) {
            Log.e(TAG, "Stop error", e)
        }
    }

    private fun updateUI() {
        try {
            if (isRunning) {
                tvStatus.text = "✅ 运行中"
                tvStatus.setTextColor(getColor(android.R.color.holo_green_dark))
                btnStart.text = "⏹ 停止截屏"
                btnHelper.visibility = android.view.View.VISIBLE
                tvHint.text = "👇 点「打开提示器」开始使用"
                tvHint.setTextColor(getColor(android.R.color.holo_orange_dark))
            } else {
                tvStatus.text = "⏸ 未启动"
                tvStatus.setTextColor(getColor(android.R.color.darker_gray))
                btnStart.text = "🚀 开始截屏"
                btnHelper.visibility = android.view.View.GONE
                tvHint.text = "先点上方按钮启动截屏"
                tvHint.setTextColor(getColor(android.R.color.darker_gray))
            }
        } catch (e: Exception) {
            Log.e(TAG, "updateUI error", e)
        }
    }

    override fun onResume() {
        super.onResume()
        try {
            isRunning = ScreenCaptureService.isRunning
            updateUI()
        } catch (e: Exception) {
            Log.e(TAG, "onResume error", e)
        }
    }
}
