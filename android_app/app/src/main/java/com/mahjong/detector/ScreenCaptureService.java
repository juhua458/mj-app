package com.mahjong.detector;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.Service;
import android.content.Context;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.PixelFormat;
import android.hardware.display.DisplayManager;
import android.hardware.display.VirtualDisplay;
import android.media.Image;
import android.media.ImageReader;
import android.media.projection.MediaProjection;
import android.media.projection.MediaProjectionManager;
import android.os.Build;
import android.os.Handler;
import android.os.IBinder;
import android.os.Looper;
import android.util.DisplayMetrics;
import android.util.Log;
import android.view.WindowManager;

import androidx.annotation.Nullable;
import androidx.core.app.NotificationCompat;

import java.nio.ByteBuffer;

/**
 * 截屏服务 - 使用MediaProjection API截取屏幕内容
 * 截屏后自动进行ONNX推理，并通过广播发送检测结果
 */
public class ScreenCaptureService extends Service {

    private static final String TAG = "ScreenCaptureService";
    private static final String CHANNEL_ID = "mahjong_capture_channel";
    private static final int NOTIFICATION_ID = 1001;

    public static final String ACTION_START = "com.mahjong.detector.ACTION_START";
    public static final String ACTION_STOP = "com.mahjong.detector.ACTION_STOP";
    public static final String EXTRA_RESULT_CODE = "RESULT_CODE";
    public static final String EXTRA_DATA = "DATA";

    public static final String ACTION_DETECT_RESULT = "com.mahjong.detector.DETECT_RESULT";
    public static final String EXTRA_DETECT_COUNT = "detect_count";
    public static final String EXTRA_DETECT_WIDTH = "screen_width";
    public static final String EXTRA_DETECT_HEIGHT = "screen_height";

    private MediaProjectionManager projectionManager;
    private MediaProjection mediaProjection;
    private VirtualDisplay virtualDisplay;
    private ImageReader imageReader;

    private MahjongDetector detector;
    private Handler handler;
    private boolean isRunning = false;

    private int screenWidth;
    private int screenHeight;
    private int screenDensity;

    // 截屏间隔（毫秒）
    private static final long CAPTURE_INTERVAL = 1000;

    private Runnable captureRunnable = new Runnable() {
        @Override
        public void run() {
            if (!isRunning) return;
            captureAndDetect();
            handler.postDelayed(this, CAPTURE_INTERVAL);
        }
    };

    @Override
    public void onCreate() {
        super.onCreate();
        handler = new Handler(Looper.getMainLooper());
        detector = new MahjongDetector();

        // 获取屏幕尺寸
        DisplayMetrics metrics = new DisplayMetrics();
        WindowManager windowManager = (WindowManager) getSystemService(Context.WINDOW_SERVICE);
        windowManager.getDefaultDisplay().getRealMetrics(metrics);
        screenWidth = metrics.widthPixels;
        screenHeight = metrics.heightPixels;
        screenDensity = metrics.densityDpi;

        projectionManager = (MediaProjectionManager) getSystemService(Context.MEDIA_PROJECTION_SERVICE);

        createNotificationChannel();
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        if (intent == null) {
            stopSelf();
            return START_NOT_STICKY;
        }

        String action = intent.getAction();
        if (ACTION_START.equals(action)) {
            int resultCode = intent.getIntExtra(EXTRA_RESULT_CODE, -1);
            Intent data = intent.getParcelableExtra(EXTRA_DATA);

            if (resultCode != -1 && data != null) {
                startForegroundNotification();
                startCapture(resultCode, data);
            }
        } else if (ACTION_STOP.equals(action)) {
            stopCapture();
            stopSelf();
        }

        return START_NOT_STICKY;
    }

    /**
     * 创建前台通知
     */
    private void startForegroundNotification() {
        Notification notification = new NotificationCompat.Builder(this, CHANNEL_ID)
                .setContentTitle("麻将检测服务运行中")
                .setContentText("正在截屏检测麻将牌...")
                .setSmallIcon(android.R.drawable.ic_menu_camera)
                .setPriority(NotificationCompat.PRIORITY_LOW)
                .build();

        startForeground(NOTIFICATION_ID, notification);
    }

    /**
     * 创建通知渠道
     */
    private void createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            NotificationChannel channel = new NotificationChannel(
                    CHANNEL_ID,
                    getString(R.string.notification_channel_name),
                    NotificationManager.IMPORTANCE_LOW
            );
            channel.setDescription(getString(R.string.notification_channel_desc));
            NotificationManager manager = getSystemService(NotificationManager.class);
            manager.createNotificationChannel(channel);
        }
    }

    /**
     * 开始截屏
     */
    private void startCapture(int resultCode, Intent data) {
        if (isRunning) return;

        // 初始化ONNX模型
        if (!detector.initialize(this)) {
            Log.e(TAG, "ONNX模型初始化失败");
            broadcastResult(0);
            return;
        }

        mediaProjection = projectionManager.getMediaProjection(resultCode, data);
        if (mediaProjection == null) {
            Log.e(TAG, "MediaProjection创建失败");
            return;
        }

        // 设置虚拟显示器回调
        mediaProjection.registerCallback(new MediaProjection.Callback() {
            @Override
            public void onStop() {
                super.onStop();
                Log.i(TAG, "MediaProjection已停止");
                isRunning = false;
                handler.removeCallbacks(captureRunnable);
            }
        }, null);

        // 创建ImageReader
        imageReader = ImageReader.newInstance(
                screenWidth, screenHeight,
                PixelFormat.RGBA_8888,
                2
        );

        imageReader.setOnImageAvailableListener(new ImageReader.OnImageAvailableListener() {
            @Override
            public void onImageAvailable(ImageReader reader) {
                // 截屏回调不在这里处理推理，由定时器控制
            }
        }, handler);

        // 创建虚拟显示器
        virtualDisplay = mediaProjection.createVirtualDisplay(
                "MahjongCapture",
                screenWidth, screenHeight, screenDensity,
                DisplayManager.VIRTUAL_DISPLAY_FLAG_AUTO_MIRROR,
                imageReader.getSurface(),
                null, handler
        );

        isRunning = true;
        Log.i(TAG, "截屏服务已启动，屏幕尺寸: " + screenWidth + "x" + screenHeight);

        // 开始定时截屏检测
        handler.postDelayed(captureRunnable, 500);
    }

    /**
     * 截屏并进行检测
     */
    private void captureAndDetect() {
        if (!isRunning || imageReader == null) return;

        Image image = null;
        try {
            image = imageReader.acquireLatestImage();
            if (image == null) return;

            // 将Image转换为Bitmap
            Image.Plane[] planes = image.getPlanes();
            ByteBuffer buffer = planes[0].getBuffer();
            int pixelStride = planes[0].getPixelStride();
            int rowStride = planes[0].getRowStride();
            int rowPadding = rowStride - pixelStride * screenWidth;

            Bitmap bitmap = Bitmap.createBitmap(
                    screenWidth + rowPadding / pixelStride,
                    screenHeight,
                    Bitmap.Config.ARGB_8888
            );
            bitmap.copyPixelsFromBuffer(buffer);

            // 裁剪掉padding
            if (rowPadding > 0) {
                bitmap = Bitmap.createBitmap(bitmap, 0, 0, screenWidth, screenHeight);
            }

            // 执行ONNX推理
            final java.util.List<MahjongDetector.Detection> detections = detector.detect(bitmap);

            // 回收Bitmap
            bitmap.recycle();

            // 广播检测结果
            broadcastResult(detections.size());

            Log.i(TAG, "检测完成，发现 " + detections.size() + " 张麻将牌");

        } catch (Exception e) {
            Log.e(TAG, "截屏检测失败: " + e.getMessage(), e);
        } finally {
            if (image != null) {
                image.close();
            }
        }
    }

    /**
     * 广播检测结果
     */
    private void broadcastResult(int count) {
        Intent intent = new Intent(ACTION_DETECT_RESULT);
        intent.putExtra(EXTRA_DETECT_COUNT, count);
        intent.putExtra(EXTRA_DETECT_WIDTH, screenWidth);
        intent.putExtra(EXTRA_DETECT_HEIGHT, screenHeight);
        sendBroadcast(intent);
    }

    /**
     * 停止截屏
     */
    private void stopCapture() {
        isRunning = false;
        handler.removeCallbacks(captureRunnable);

        if (virtualDisplay != null) {
            virtualDisplay.release();
            virtualDisplay = null;
        }

        if (imageReader != null) {
            imageReader.close();
            imageReader = null;
        }

        if (mediaProjection != null) {
            mediaProjection.stop();
            mediaProjection = null;
        }

        if (detector != null) {
            detector.release();
        }

        Log.i(TAG, "截屏服务已停止");
    }

    @Override
    public void onDestroy() {
        stopCapture();
        super.onDestroy();
    }

    @Nullable
    @Override
    public IBinder onBind(Intent intent) {
        return null;
    }
}
