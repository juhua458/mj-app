package com.mahjong.detector;

import android.app.Service;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.PixelFormat;
import android.graphics.Typeface;
import android.os.Build;
import android.os.IBinder;
import android.util.DisplayMetrics;
import android.util.Log;
import android.view.Gravity;
import android.view.MotionEvent;
import android.view.View;
import android.view.WindowManager;
import android.widget.LinearLayout;
import android.widget.TextView;

import androidx.annotation.Nullable;

/**
 * 悬浮窗服务 - 显示麻将牌检测结果
 * 包含检测数量文字和可拖动的悬浮窗
 */
public class FloatingViewService extends Service {

    private static final String TAG = "FloatingViewService";

    private WindowManager windowManager;
    private WindowManager.LayoutParams params;

    private LinearLayout floatingLayout;
    private TextView tvCount;
    private DrawView drawView;

    private int screenWidth;
    private int screenHeight;

    // 拖动相关
    private float initialTouchX;
    private float initialTouchY;
    private float initialX;
    private float initialY;
    private boolean isDragging = false;

    private BroadcastReceiver resultReceiver = new BroadcastReceiver() {
        @Override
        public void onReceive(Context context, Intent intent) {
            if (ScreenCaptureService.ACTION_DETECT_RESULT.equals(intent.getAction())) {
                int count = intent.getIntExtra(ScreenCaptureService.EXTRA_DETECT_COUNT, 0);
                int width = intent.getIntExtra(ScreenCaptureService.EXTRA_DETECT_WIDTH, 0);
                int height = intent.getIntExtra(ScreenCaptureService.EXTRA_DETECT_HEIGHT, 0);
                updateDetectionCount(count);
            }
        }
    };

    @Override
    public void onCreate() {
        super.onCreate();

        windowManager = (WindowManager) getSystemService(Context.WINDOW_SERVICE);
        DisplayMetrics metrics = new DisplayMetrics();
        windowManager.getDefaultDisplay().getRealMetrics(metrics);
        screenWidth = metrics.widthPixels;
        screenHeight = metrics.heightPixels;

        createFloatingView();
        registerResultReceiver();
    }

    /**
     * 创建悬浮窗
     */
    private void createFloatingView() {
        // 主布局
        floatingLayout = new LinearLayout(this);
        floatingLayout.setOrientation(LinearLayout.VERTICAL);
        floatingLayout.setBackgroundColor(Color.parseColor("#CC333333"));
        floatingLayout.setPadding(16, 12, 16, 12);

        // 检测数量文字
        tvCount = new TextView(this);
        tvCount.setText("等待检测...");
        tvCount.setTextColor(Color.WHITE);
        tvCount.setTextSize(16);
        tvCount.setTypeface(null, Typeface.BOLD);
        tvCount.setGravity(Gravity.CENTER);

        LinearLayout.LayoutParams textParams = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.WRAP_CONTENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
        );
        textParams.gravity = Gravity.CENTER;
        floatingLayout.addView(tvCount, textParams);

        // 设置窗口参数
        int layoutType;
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            layoutType = WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY;
        } else {
            layoutType = WindowManager.LayoutParams.TYPE_PHONE;
        }

        params = new WindowManager.LayoutParams(
                WindowManager.LayoutParams.WRAP_CONTENT,
                WindowManager.LayoutParams.WRAP_CONTENT,
                layoutType,
                WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE,
                PixelFormat.TRANSLUCENT
        );

        params.gravity = Gravity.TOP | Gravity.END;
        params.x = 16;
        params.y = 100;

        // 添加触摸拖动支持
        floatingLayout.setOnTouchListener(new View.OnTouchListener() {
            @Override
            public boolean onTouch(View v, MotionEvent event) {
                switch (event.getAction()) {
                    case MotionEvent.ACTION_DOWN:
                        initialTouchX = event.getRawX();
                        initialTouchY = event.getRawY();
                        initialX = params.x;
                        initialY = params.y;
                        isDragging = false;
                        return true;

                    case MotionEvent.ACTION_MOVE:
                        float dx = event.getRawX() - initialTouchX;
                        float dy = event.getRawY() - initialTouchY;

                        if (Math.abs(dx) > 5 || Math.abs(dy) > 5) {
                            isDragging = true;
                        }

                        if (isDragging) {
                            params.x = (int) (initialX - dx);
                            params.y = (int) (initialY + dy);
                            windowManager.updateViewLayout(floatingLayout, params);
                        }
                        return true;

                    case MotionEvent.ACTION_UP:
                        return isDragging;

                    default:
                        return false;
                }
            }
        });

        windowManager.addView(floatingLayout, params);
        Log.i(TAG, "悬浮窗已创建");
    }

    /**
     * 注册广播接收器
     */
    private void registerResultReceiver() {
        IntentFilter filter = new IntentFilter(ScreenCaptureService.ACTION_DETECT_RESULT);
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            registerReceiver(resultReceiver, filter, Context.RECEIVER_NOT_EXPORTED);
        } else {
            registerReceiver(resultReceiver, filter);
        }
    }

    /**
     * 更新检测数量显示
     */
    private void updateDetectionCount(int count) {
        if (tvCount != null) {
            String text = getString(R.string.floating_count, count);
            tvCount.setText(text);

            // 根据数量改变背景色
            if (count > 0) {
                floatingLayout.setBackgroundColor(Color.parseColor("#CCFF5722"));
                tvCount.setTextColor(Color.WHITE);
            } else {
                floatingLayout.setBackgroundColor(Color.parseColor("#CC333333"));
                tvCount.setTextColor(Color.WHITE);
            }
        }
    }

    @Override
    public void onDestroy() {
        super.onDestroy();

        // 移除悬浮窗
        if (floatingLayout != null && windowManager != null) {
            try {
                windowManager.removeView(floatingLayout);
            } catch (IllegalArgumentException e) {
                Log.w(TAG, "悬浮窗已被移除");
            }
        }

        // 注销广播接收器
        try {
            unregisterReceiver(resultReceiver);
        } catch (IllegalArgumentException e) {
            Log.w(TAG, "广播接收器未注册");
        }
    }

    @Nullable
    @Override
    public IBinder onBind(Intent intent) {
        return null;
    }

    /**
     * 绘制检测框的自定义View（用于在全屏悬浮窗上绘制）
     */
    private static class DrawView extends View {
        private Paint boxPaint;
        private Paint textPaint;

        public DrawView(Context context) {
            super(context);
            initPaints();
        }

        private void initPaints() {
            boxPaint = new Paint();
            boxPaint.setColor(Color.parseColor("#FFFF5722"));
            boxPaint.setStyle(Paint.Style.STROKE);
            boxPaint.setStrokeWidth(4);

            textPaint = new Paint();
            textPaint.setColor(Color.WHITE);
            textPaint.setTextSize(32);
            textPaint.setTypeface(Typeface.DEFAULT_BOLD);
            textPaint.setShadowLayer(4, 2, 2, Color.BLACK);
        }

        @Override
        protected void onDraw(Canvas canvas) {
            super.onDraw(canvas);
            // 检测框绘制由外部调用
        }
    }
}
