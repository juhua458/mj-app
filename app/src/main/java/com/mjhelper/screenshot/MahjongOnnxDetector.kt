package com.mjhelper.screenshot

import android.content.Context
import android.graphics.Bitmap
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.FloatBuffer
import java.io.InputStream
import kotlin.math.max
import kotlin.math.min

/**
 * ONNX Runtime 麻将牌识别引擎
 * 使用 YOLO11n nano模型，输入640x640，输出35类麻将牌
 * 
 * 类别映射: m=万子, p=筒子, s=条子, z=字牌
 * 0:1m 1:1p 2:1s 3:1z(东风) 4:2m 5:2p 6:2s 7:2z(南风) ...
 * 34:UNKNOWN
 */
class MahjongOnnxDetector(context: Context) {
    
    private var env: OrtEnvironment? = null
    private var session: OrtSession? = null
    private var isLoaded = false
    
    // 麻将牌类别名称
    private val classNames = mapOf(
        0 to "一万", 1 to "一筒", 2 to "一条", 3 to "东风",
        4 to "二万", 5 to "二筒", 6 to "二条", 7 to "南风",
        8 to "三万", 9 to "三筒", 10 to "三条", 11 to "西风",
        12 to "四万", 13 to "四筒", 14 to "四条", 15 to "北风",
        16 to "五万", 17 to "五筒", 18 to "五条", 19 to "中",
        20 to "六万", 21 to "六筒", 22 to "六条", 23 to "发",
        24 to "七万", 25 to "七筒", 26 to "七条", 27 to "白",
        28 to "八万", 29 to "八筒", 30 to "八条",
        31 to "九万", 32 to "九筒", 33 to "九条",
        34 to "未知"
    )
    
    // 简称映射(用于策略引擎)
    private val shortNames = mapOf(
        0 to "1m", 1 to "1p", 2 to "1s", 3 to "1z",
        4 to "2m", 5 to "2p", 6 to "2s", 7 to "2z",
        8 to "3m", 9 to "3p", 10 to "3s", 11 to "3z",
        12 to "4m", 13 to "4p", 14 to "4s", 15 to "4z",
        16 to "5m", 17 to "5p", 18 to "5s", 19 to "5z",
        20 to "6m", 21 to "6p", 22 to "6s", 23 to "6z",
        24 to "7m", 25 to "7p", 26 to "7s", 27 to "7z",
        28 to "8m", 29 to "8p", 30 to "8s",
        31 to "9m", 32 to "9p", 33 to "9s",
        34 to "X"
    )
    
    fun loadModel(inputStream: InputStream): Boolean {
        return try {
            env = OrtEnvironment.getEnvironment()
            val modelBytes = inputStream.readBytes()
            val sessionOptions = OrtSession.SessionOptions()
            sessionOptions.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            // 启用NNAPI加速(Android Neural Networks API)
            sessionOptions.addNnapi()  
            session = env?.createSession(modelBytes, sessionOptions)
            isLoaded = true
            true
        } catch (e: Exception) {
            isLoaded = false
            false
        }
    }
    
    data class Detection(
        val classId: Int,
        val className: String,     // 中文名
        val shortName: String,     // 策略引擎用简称
        val confidence: Float,
        val x1: Float, val y1: Float,
        val x2: Float, val y2: Float,
        val centerX: Float
    )
    
    data class RecognitionResult(
        val handTiles: List<Detection>,      // 手牌(按x坐标排序)
        val allDetections: List<Detection>,  // 所有检测
        val handTileNames: List<String>,     // 手牌简称列表
        val confidence: Float                // 平均置信度
    )
    
    /**
     * 识别截图中的麻将牌
     * @param bitmap 截图Bitmap
     * @param handRegionY1 手牌区域上边界(相对比例 0-1)
     * @param handRegionY2 手牌区域下边界(相对比例 0-1)
     * @param confThreshold 置信度阈值
     */
    fun recognize(
        bitmap: Bitmap, 
        handRegionY1: Float = 0.75f,
        handRegionY2: Float = 0.95f,
        confThreshold: Float = 0.4f
    ): RecognitionResult? {
        if (!isLoaded || session == null || env == null) return null
        
        return try {
            // 1. 预处理: resize到640x640, 归一化到0-1, CHW格式
            val resized = Bitmap.createScaledBitmap(bitmap, 640, 640, true)
            val input = preprocess(resized)
            resized.recycle()
            
            // 2. 推理
            val inputName = session?.inputNames?.iterator()?.next() ?: return null
            val inputBuffer = FloatBuffer.wrap(input)
            val inputTensor = OnnxTensor.createTensor(env, inputBuffer, longArrayOf(1, 3, 640, 640))
            val output = session?.run(mapOf(inputName to inputTensor))
            inputTensor.close()
            
            // 3. 后处理: 解析YOLO输出 (1, 39, 8400)
            val outputTensor = output?.get(0)?.value as? Array<FloatArray> ?: return null
            val detections = postprocess(outputTensor, bitmap.width, bitmap.height, confThreshold)
            
            // 4. 分离手牌区和其他区域
            val imgH = bitmap.height.toFloat()
            val handY1 = imgH * handRegionY1
            val handY2 = imgH * handRegionY2
            
            val handTiles = detections.filter { 
                it.y1 >= handY1 && it.y2 <= handY2 * 1.1f 
            }.sortedBy { it.centerX }
            
            val allNames = handTiles.map { it.shortName }
            val avgConf = if (handTiles.isNotEmpty()) handTiles.map { it.confidence }.average().toFloat() else 0f
            
            RecognitionResult(handTiles, detections, allNames, avgConf)
        } catch (e: Exception) {
            null
        }
    }
    
    private fun preprocess(bitmap: Bitmap): FloatArray {
        val floatArray = FloatArray(1 * 3 * 640 * 640)
        val pixels = IntArray(640 * 640)
        bitmap.getPixels(pixels, 0, 640, 0, 0, 640, 640)
        
        for (i in pixels.indices) {
            val pixel = pixels[i]
            // CHW格式, BGR顺序(与YOLO训练一致)
            floatArray[i] = ((pixel shr 16) and 0xFF) / 255.0f  // R
            floatArray[640 * 640 + i] = ((pixel shr 8) and 0xFF) / 255.0f  // G
            floatArray[2 * 640 * 640 + i] = (pixel and 0xFF) / 255.0f  // B
        }
        return floatArray
    }
    
    private fun postprocess(
        output: Array<FloatArray>, 
        origW: Int, origH: Int, 
        confThreshold: Float
    ): List<Detection> {
        val detections = mutableListOf<Detection>()
        // output shape: [1][39][8400] -> output[0] is the batch
        // 39 = 4(bbox) + 35(classes) for each of 8400 anchors
        val data = output[0]  // [39][8400]
        val numAnchors = 8400
        val numClasses = 35
        
        // Scale factors from 640x640 to original size
        val scaleX = origW / 640.0f
        val scaleY = origH / 640.0f
        
        for (i in 0 until numAnchors) {
            // Find max class score
            var maxClassScore = 0f
            var maxClassId = 0
            for (c in 0 until numClasses) {
                val score = data[4 + c + i * 39]  
                if (score > maxClassScore) {
                    maxClassScore = score
                    maxClassId = c
                }
            }
            
            if (maxClassScore < confThreshold) continue
            
            // Decode bbox (cx, cy, w, h)
            val cx = data[0 + i * 39] * scaleX
            val cy = data[1 + i * 39] * scaleY
            val w = data[2 + i * 39] * scaleX
            val h = data[3 + i * 39] * scaleY
            
            detections.add(Detection(
                classId = maxClassId,
                className = classNames[maxClassId] ?: "?",
                shortName = shortNames[maxClassId] ?: "?",
                confidence = maxClassScore,
                x1 = cx - w / 2, y1 = cy - h / 2,
                x2 = cx + w / 2, y2 = cy + h / 2,
                centerX = cx
            ))
        }
        
        // NMS (Non-Maximum Suppression)
        return nms(detections, 0.45f)
    }
    
    private fun nms(detections: List<Detection>, iouThreshold: Float): List<Detection> {
        val sorted = detections.sortedByDescending { it.confidence }
        val selected = mutableListOf<Detection>()
        val suppressed = BooleanArray(sorted.size)
        
        for (i in sorted.indices) {
            if (suppressed[i]) continue
            selected.add(sorted[i])
            for (j in i + 1 until sorted.size) {
                if (suppressed[j]) continue
                if (iou(sorted[i], sorted[j]) > iouThreshold) {
                    suppressed[j] = true
                }
            }
        }
        return selected
    }
    
    private fun iou(a: Detection, b: Detection): Float {
        val x1 = max(a.x1, b.x1)
        val y1 = max(a.y1, b.y1)
        val x2 = min(a.x2, b.x2)
        val y2 = min(a.y2, b.y2)
        val intersection = max(0f, x2 - x1) * max(0f, y2 - y1)
        val areaA = (a.x2 - a.x1) * (a.y2 - a.y1)
        val areaB = (b.x2 - b.x1) * (b.y2 - b.y1)
        return intersection / (areaA + areaB - intersection + 1e-6f)
    }
    
    fun close() {
        session?.close()
        env?.close()
        isLoaded = false
    }
}