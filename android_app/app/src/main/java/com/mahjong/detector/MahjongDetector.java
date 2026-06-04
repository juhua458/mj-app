package com.mahjong.detector;

import android.content.Context;
import android.graphics.Bitmap;
import android.util.Log;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

/**
 * 麻将牌面检测器 - 基于ONNX Runtime Mobile的YOLO推理引擎
 *
 * 模型输入: (1, 3, 640, 640) float32, NCHW格式, RGB
 * 模型输出: (1, 5, 8400) float32, 5=[cx, cy, w, h, conf]
 */
public class MahjongDetector {

    private static final String TAG = "MahjongDetector";

    // 模型参数
    private static final int INPUT_SIZE = 640;
    private static final float CONFIDENCE_THRESHOLD = 0.25f;
    private static final float NMS_IOU_THRESHOLD = 0.45f;
    private static final String MODEL_FILE = "mahjong_detect.onnx";
    private static final String INPUT_NAME = "images";
    private static final String OUTPUT_NAME = "output0";

    private OrtEnvironment ortEnv;
    private OrtSession ortSession;
    private boolean isInitialized = false;

    /**
     * 检测结果
     */
    public static class Detection {
        public float x1; // 左上角x（原始图像坐标）
        public float y1; // 左上角y（原始图像坐标）
        public float x2; // 右下角x（原始图像坐标）
        public float y2; // 右下角y（原始图像坐标）
        public float confidence; // 置信度

        public Detection(float x1, float y1, float x2, float y2, float confidence) {
            this.x1 = x1;
            this.y1 = y1;
            this.x2 = x2;
            this.y2 = y2;
            this.confidence = confidence;
        }

        public float getWidth() {
            return x2 - x1;
        }

        public float getHeight() {
            return y2 - y1;
        }

        public float getCenterX() {
            return (x1 + x2) / 2.0f;
        }

        public float getCenterY() {
            return (y1 + y2) / 2.0f;
        }
    }

    /**
     * 初始化ONNX模型
     */
    public boolean initialize(Context context) {
        try {
            ortEnv = OrtEnvironment.getEnvironment();

            // 从assets加载模型
            byte[] modelBytes = loadModelFromAssets(context, MODEL_FILE);
            OrtSession.SessionOptions sessionOptions = new OrtSession.SessionOptions();
            sessionOptions.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
            // 使用NNAPI加速（如果可用）
            try {
                sessionOptions.addNnapi();
            } catch (OrtException e) {
                Log.w(TAG, "NNAPI not available, using CPU: " + e.getMessage());
            }

            ortSession = ortEnv.createSession(modelBytes, sessionOptions);
            sessionOptions.close();
            isInitialized = true;
            Log.i(TAG, "ONNX模型加载成功");
            return true;
        } catch (OrtException e) {
            Log.e(TAG, "ONNX模型加载失败: " + e.getMessage(), e);
            return false;
        } catch (IOException e) {
            Log.e(TAG, "读取模型文件失败: " + e.getMessage(), e);
            return false;
        }
    }

    /**
     * 从assets加载模型文件
     */
    private byte[] loadModelFromAssets(Context context, String fileName) throws IOException {
        InputStream inputStream = context.getAssets().open(fileName);
        byte[] buffer = new byte[inputStream.available()];
        int bytesRead = inputStream.read(buffer);
        inputStream.close();
        if (bytesRead != buffer.length) {
            // 分块读取
            List<Byte> byteList = new ArrayList<>();
            inputStream = context.getAssets().open(fileName);
            byte[] tmpBuf = new byte[8192];
            while ((bytesRead = inputStream.read(tmpBuf)) != -1) {
                for (int i = 0; i < bytesRead; i++) {
                    byteList.add(tmpBuf[i]);
                }
            }
            inputStream.close();
            buffer = new byte[byteList.size()];
            for (int i = 0; i < buffer.length; i++) {
                buffer[i] = byteList.get(i);
            }
        }
        return buffer;
    }

    /**
     * 对Bitmap进行推理，返回检测结果
     *
     * @param bitmap 输入图像
     * @return 检测结果列表
     */
    public List<Detection> detect(Bitmap bitmap) {
        if (!isInitialized || bitmap == null) {
            return Collections.emptyList();
        }

        try {
            // 1. 预处理：letterbox到640x640
            float[] preprocessed = preprocess(bitmap);

            // 2. 创建输入tensor
            long[] inputShape = {1, 3, INPUT_SIZE, INPUT_SIZE};
            FloatBuffer inputBuffer = FloatBuffer.wrap(preprocessed);
            OnnxTensor inputTensor = OnnxTensor.createTensor(ortEnv, inputBuffer, inputShape);

            // 3. 推理
            OrtSession.Result result = ortSession.run(
                    Collections.singletonMap(INPUT_NAME, inputTensor)
            );

            // 4. 获取输出
            float[][][] outputArray = (float[][][]) result.get(0).getValue();
            result.close();
            inputTensor.close();

            // 5. 后处理
            List<Detection> detections = postprocess(outputArray[0], bitmap.getWidth(), bitmap.getHeight());

            return detections;

        } catch (OrtException e) {
            Log.e(TAG, "推理失败: " + e.getMessage(), e);
            return Collections.emptyList();
        }
    }

    /**
     * 预处理：letterbox缩放 + BGR转RGB + 归一化
     *
     * @param bitmap 输入图像
     * @return float[3*640*640] NCHW格式的数据
     */
    private float[] preprocess(Bitmap bitmap) {
        int srcWidth = bitmap.getWidth();
        int srcHeight = bitmap.getHeight();

        // 计算letterbox缩放比例（保持宽高比）
        float scale = Math.min(
                (float) INPUT_SIZE / srcWidth,
                (float) INPUT_SIZE / srcHeight
        );
        int newWidth = Math.round(srcWidth * scale);
        int newHeight = Math.round(srcHeight * scale);

        // 缩放图像
        Bitmap scaledBitmap = Bitmap.createScaledBitmap(bitmap, newWidth, newHeight, true);

        // 创建640x640的画布，填充灰色(128,128,128)作为letterbox padding
        Bitmap letterboxBitmap = Bitmap.createBitmap(INPUT_SIZE, INPUT_SIZE, Bitmap.Config.ARGB_8888);
        android.graphics.Canvas canvas = new android.graphics.Canvas(letterboxBitmap);
        canvas.drawColor(0xFF808080); // 灰色padding
        int padLeft = (INPUT_SIZE - newWidth) / 2;
        int padTop = (INPUT_SIZE - newHeight) / 2;
        canvas.drawBitmap(scaledBitmap, padLeft, padTop, null);

        // 提取像素数据并转换为NCHW RGB格式
        int[] pixels = new int[INPUT_SIZE * INPUT_SIZE];
        letterboxBitmap.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE);

        float[] result = new float[3 * INPUT_SIZE * INPUT_SIZE];
        int pixelCount = INPUT_SIZE * INPUT_SIZE;

        for (int i = 0; i < pixelCount; i++) {
            int pixel = pixels[i];
            // ARGB格式提取
            float r = ((pixel >> 16) & 0xFF) / 255.0f;
            float g = ((pixel >> 8) & 0xFF) / 255.0f;
            float b = (pixel & 0xFF) / 255.0f;

            // NCHW格式: 通道在前
            result[i] = r;                          // R通道
            result[pixelCount + i] = g;              // G通道
            result[2 * pixelCount + i] = b;         // B通道
        }

        // 回收临时Bitmap
        if (scaledBitmap != bitmap) {
            scaledBitmap.recycle();
        }
        letterboxBitmap.recycle();

        return result;
    }

    /**
     * 后处理：解析输出 + 置信度过滤 + NMS
     *
     * @param output 模型输出 [5][8400]
     * @param origWidth 原始图像宽度
     * @param origHeight 原始图像高度
     * @return 检测结果列表
     */
    private List<Detection> postprocess(float[][] output, int origWidth, int origHeight) {
        // output[5][8400]: 0=cx, 1=cy, 2=w, 3=h, 4=conf
        int numDetections = output[0].length; // 8400
        List<Detection> allDetections = new ArrayList<>();

        // 计算letterbox参数
        float scale = Math.min(
                (float) INPUT_SIZE / origWidth,
                (float) INPUT_SIZE / origHeight
        );
        int newWidth = Math.round(origWidth * scale);
        int newHeight = Math.round(origHeight * scale);
        int padLeft = (INPUT_SIZE - newWidth) / 2;
        int padTop = (INPUT_SIZE - newHeight) / 2;

        for (int i = 0; i < numDetections; i++) {
            float conf = output[4][i];

            // 置信度过滤
            if (conf < CONFIDENCE_THRESHOLD) {
                continue;
            }

            // 从640x640坐标转换到原始图像坐标
            float cx = (output[0][i] - padLeft) / scale;
            float cy = (output[1][i] - padTop) / scale;
            float w = output[2][i] / scale;
            float h = output[3][i] / scale;

            // 转换为x1,y1,x2,y2
            float x1 = cx - w / 2.0f;
            float y1 = cy - h / 2.0f;
            float x2 = cx + w / 2.0f;
            float y2 = cy + h / 2.0f;

            // 裁剪到图像边界
            x1 = Math.max(0, Math.min(x1, origWidth));
            y1 = Math.max(0, Math.min(y1, origHeight));
            x2 = Math.max(0, Math.min(x2, origWidth));
            y2 = Math.max(0, Math.min(y2, origHeight));

            allDetections.add(new Detection(x1, y1, x2, y2, conf));
        }

        // NMS去重
        return nms(allDetections, NMS_IOU_THRESHOLD);
    }

    /**
     * 非极大值抑制（NMS）
     */
    private List<Detection> nms(List<Detection> detections, float iouThreshold) {
        // 按置信度降序排序
        List<Detection> sorted = new ArrayList<>(detections);
        Collections.sort(sorted, new Comparator<Detection>() {
            @Override
            public int compare(Detection a, Detection b) {
                return Float.compare(b.confidence, a.confidence);
            }
        });

        List<Detection> kept = new ArrayList<>();
        boolean[] suppressed = new boolean[sorted.size()];

        for (int i = 0; i < sorted.size(); i++) {
            if (suppressed[i]) continue;
            kept.add(sorted.get(i]);

            for (int j = i + 1; j < sorted.size(); j++) {
                if (suppressed[j]) continue;
                if (computeIoU(sorted.get(i), sorted.get(j)) > iouThreshold) {
                    suppressed[j] = true;
                }
            }
        }

        return kept;
    }

    /**
     * 计算两个检测框的IoU
     */
    private float computeIoU(Detection a, Detection b) {
        float interX1 = Math.max(a.x1, b.x1);
        float interY1 = Math.max(a.y1, b.y1);
        float interX2 = Math.min(a.x2, b.x2);
        float interY2 = Math.min(a.y2, b.y2);

        float interArea = Math.max(0, interX2 - interX1) * Math.max(0, interY2 - interY1);
        float areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
        float areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
        float unionArea = areaA + areaB - interArea;

        if (unionArea <= 0) return 0;
        return interArea / unionArea;
    }

    /**
     * 释放资源
     */
    public void release() {
        try {
            if (ortSession != null) {
                ortSession.close();
                ortSession = null;
            }
        } catch (OrtException e) {
            Log.e(TAG, "关闭session失败: " + e.getMessage());
        }
        isInitialized = false;
    }

    public boolean isInitialized() {
        return isInitialized;
    }
}
