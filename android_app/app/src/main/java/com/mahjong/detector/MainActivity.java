package com.mahjong.detector;

import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.provider.Settings;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

/**
 * 主界面 - 请求权限、启动截屏服务
 */
public class MainActivity extends AppCompatActivity {

    private static final String TAG = "MainActivity";
    private static final int REQUEST_MEDIA_PROJECTION = 1001;
    private static final int REQUEST_OVERLAY_PERMISSION = 1002;

    private Button btnStartCapture;
    private Button btnStopService;
    private TextView tvStatus;
    private TextView tvResult;

    private boolean isServiceRunning = false;

    private BroadcastReceiver resultReceiver = new BroadcastReceiver() {
        @Override
        public void onReceive(Context context, Intent intent) {
            if (ScreenCaptureService.ACTION_DETECT_RESULT.equals(intent.getAction())) {
                int count = intent.getIntExtra(ScreenCaptureService.EXTRA_DETECT_COUNT, 0);
                updateResult(count);
            }
        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        btnStartCapture = findViewById(R.id.btnStartCapture);
        btnStopService = findViewById(R.id.btnStopService);
        tvStatus = findViewById(R.id.tvStatus);
        tvResult = findViewById(R.id.tvResult);

        btnStartCapture.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                startCaptureFlow();
            }
        });

        btnStopService.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                stopServices();
            }
        });
    }

    /**
     * 开始截屏流程：检查权限 -> 请求截屏 -> 启动服务
     */
    private void startCaptureFlow() {
        // 1. 检查悬浮窗权限
        if (!checkOverlayPermission()) {
            requestOverlayPermission();
            return;
        }

        // 2. 请求截屏权限
        requestScreenCapture();
    }

    /**
     * 检查是否有悬浮窗权限
     */
    private boolean checkOverlayPermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            return Settings.canDrawOverlays(this);
        }
        return true;
    }

    /**
     * 请求悬浮窗权限
     */
    private void requestOverlayPermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            Intent intent = new Intent(
                    Settings.ACTION_MANAGE_OVERLAY_PERMISSION,
                    Uri.parse("package:" + getPackageName())
            );
            startActivityForResult(intent, REQUEST_OVERLAY_PERMISSION);
        }
    }

    /**
     * 请求截屏权限（MediaProjection）
     */
    private void requestScreenCapture() {
        android.media.projection.MediaProjectionManager projectionManager =
                (android.media.projection.MediaProjectionManager) getSystemService(Context.MEDIA_PROJECTION_SERVICE);

        if (projectionManager != null) {
            startActivityForResult(
                    projectionManager.createScreenCaptureIntent(),
                    REQUEST_MEDIA_PROJECTION
            );
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if (requestCode == REQUEST_OVERLAY_PERMISSION) {
            if (checkOverlayPermission()) {
                // 悬浮窗权限已获取，继续请求截屏
                requestScreenCapture();
            } else {
                Toast.makeText(this, R.string.permission_overlay_denied, Toast.LENGTH_SHORT).show();
            }
        } else if (requestCode == REQUEST_MEDIA_PROJECTION) {
            if (resultCode == RESULT_OK && data != null) {
                startServices(resultCode, data);
            } else {
                Toast.makeText(this, "截屏权限被拒绝", Toast.LENGTH_SHORT).show();
                tvStatus.setText(R.string.status_ready);
            }
        }
    }

    /**
     * 启动截屏服务和悬浮窗服务
     */
    private void startServices(int resultCode, Intent data) {
        // 启动截屏服务
        Intent captureIntent = new Intent(this, ScreenCaptureService.class);
        captureIntent.setAction(ScreenCaptureService.ACTION_START);
        captureIntent.putExtra(ScreenCaptureService.EXTRA_RESULT_CODE, resultCode);
        captureIntent.putExtra(ScreenCaptureService.EXTRA_DATA, data);

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            startForegroundService(captureIntent);
        } else {
            startService(captureIntent);
        }

        // 启动悬浮窗服务
        Intent floatingIntent = new Intent(this, FloatingViewService.class);
        startService(floatingIntent);

        isServiceRunning = true;
        tvStatus.setText(R.string.status_capturing);
        btnStartCapture.setVisibility(View.GONE);
        btnStopService.setVisibility(View.VISIBLE);

        Toast.makeText(this, "截屏检测已启动", Toast.LENGTH_SHORT).show();
    }

    /**
     * 停止所有服务
     */
    private void stopServices() {
        // 停止截屏服务
        Intent captureIntent = new Intent(this, ScreenCaptureService.class);
        captureIntent.setAction(ScreenCaptureService.ACTION_STOP);
        startService(captureIntent);

        // 停止悬浮窗服务
        Intent floatingIntent = new Intent(this, FloatingViewService.class);
        stopService(floatingIntent);

        isServiceRunning = false;
        tvStatus.setText(R.string.status_stopped);
        tvResult.setText("");
        btnStartCapture.setVisibility(View.VISIBLE);
        btnStopService.setVisibility(View.GONE);

        Toast.makeText(this, "服务已停止", Toast.LENGTH_SHORT).show();
    }

    /**
     * 更新检测结果
     */
    private void updateResult(int count) {
        tvResult.setText("检测到 " + count + " 张麻将牌");
    }

    @Override
    protected void onResume() {
        super.onResume();
        // 注册广播接收器
        IntentFilter filter = new IntentFilter(ScreenCaptureService.ACTION_DETECT_RESULT);
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            registerReceiver(resultReceiver, filter, Context.RECEIVER_NOT_EXPORTED);
        } else {
            registerReceiver(resultReceiver, filter);
        }
    }

    @Override
    protected void onPause() {
        super.onPause();
        try {
            unregisterReceiver(resultReceiver);
        } catch (IllegalArgumentException e) {
            Log.w(TAG, "广播接收器未注册");
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        try {
            unregisterReceiver(resultReceiver);
        } catch (IllegalArgumentException e) {
            Log.w(TAG, "广播接收器未注册");
        }
    }
}
