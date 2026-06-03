package com.mjhelper.screenshot

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.OnnxValue
import java.nio.FloatBuffer
import java.io.InputStream
import kotlin.math.max
import kotlin.math.min

/**
 * ONNX Runtime 麻将牌识别引擎 v4
 * 
 * v4改进:
 * 1. Letterbox预处理(保持宽高比+灰色填充), 与YOLO训练一致
 * 2. FloatBuffer解析输出tensor, 不依赖类型转换
 * 3. 支持截图裁剪(去掉helper面板区域)
 * 4. 降低默认置信度阈值
 * 5. 暴露lastDebug调试信息
 */
class MahjongOnnxDetector(context: Context) {
    
    private var env: OrtEnvironment? = null
    private var session: OrtSession? = null
    private var isLoaded = false
    var lastError: String = ""
        private set
    var useNNAPI: Boolean = false
        private set
    var lastDebug: String = ""
        private set
    
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
        lastError = ""
        lastDebug = ""
        
        // 先关闭旧session
        try { session?.close() } catch (_: Exception) {}
        try { env?.close() } catch (_: Exception) {}
        session = null
        env = null
        isLoaded = false
        useNNAPI = false
        
        // 读取模型字节
        val modelBytes = try {
            inputStream.readBytes()
        } catch (e: Exception) {
            lastError = "读取模型文件失败: ${e.message}"
            return false
        }
        
        if (modelBytes.isEmpty()) {
            lastError = "模型文件为空(0字节)"
            return false
        }
        
        lastDebug = "modelBytes=${modelBytes.size}"
        
        // 策略1: 先用纯CPU加载（保证成功）
        try {
            env = OrtEnvironment.getEnvironment()
            val cpuOptions = OrtSession.SessionOptions()
            cpuOptions.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            session = env?.createSession(modelBytes, cpuOptions)
            isLoaded = true
            useNNAPI = false
            lastError = ""
            cpuOptions.close()
            return true
        } catch (e: Exception) {
            lastError = "CPU模式加载失败: ${e.message}"
        }
        
        // 策略2: 如果CPU也失败，尝试不同优化级别
        try {
            env = OrtEnvironment.getEnvironment()
            val basicOptions = OrtSession.SessionOptions()
            basicOptions.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.NO_OPT)
            session = env?.createSession(modelBytes, basicOptions)
            isLoaded = true
            useNNAPI = false
            lastError = ""
            basicOptions.close()
            return true
        } catch (e: Exception) {
            lastError = "CPU无优化模式也失败: ${e.message}"
        }
        
        return false
    }
    
    /**
     * 尝试切换到NNAPI加速（在CPU加载成功后调用）
     */
    fun tryEnableNNAPI(modelBytes: ByteArray): Boolean {
        try {
            val nnapiOptions = OrtSession.SessionOptions()
            nnapiOptions.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            nnapiOptions.addNnapi()
            val nnapiSession = env?.createSession(modelBytes, nnapiOptions)
            session?.close()
            session = nnapiSession
            useNNAPI = true
            nnapiOptions.close()
            return true
        } catch (e: Exception) {
            return false
        }
    }
    
    data class Detection(
        val classId: Int,
        val className: String,
        val shortName: String,
        val confidence: Float,
        val x1: Float, val y1: Float,
        val x2: Float, val y2: Float,
        val centerX: Float,
        val centerY: Float
    )
    
    data class RecognitionResult(
        val handTiles: List<Detection>,
        val allDetections: List<Detection>,
        val handTileNames: List<String>,
        val confidence: Float,
        val debugInfo: String
    )
    
    /**
     * Letterbox预处理: 保持宽高比缩放到640x640, 短边灰色填充
     * 与YOLO训练时的预处理一致
     */
    private fun letterboxResize(bitmap: Bitmap, targetSize: Int): Pair<Bitmap, FloatArray> {
        val w = bitmap.width
        val h = bitmap.height
        val scale = minOf(targetSize.toFloat() / w, targetSize.toFloat() / h)
        val newW = (w * scale).toInt()
        val newH = (h * scale).toInt()
        
        val resized = Bitmap.createScaledBitmap(bitmap, newW, newH, true)
        
        val result = Bitmap.createBitmap(targetSize, targetSize, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(result)
        canvas.drawColor(Color.rgb(114, 114, 114)) // YOLO标准灰色填充
        val dx = (targetSize - newW) / 2f
        val dy = (targetSize - newH) / 2f
        canvas.drawBitmap(resized, dx, dy, null)
        resized.recycle()
        
        // 返回letterbox参数: [scale, padX, padY]
        return Pair(result, floatArrayOf(scale, dx, dy))
    }
    
    fun recognize(
        bitmap: Bitmap, 
        handRegionY1: Float = 0.70f,
        handRegionY2: Float = 1.0f,
        confThreshold: Float = 0.15f,
        cropRight: Int = 0  // 从右侧裁掉的像素数(helper面板宽度)
    ): RecognitionResult? {
        if (!isLoaded || session == null || env == null) return null
        
        return try {
            // 1. 裁剪掉右侧helper面板区域
            val cropWidth = if (cropRight > 0 && cropRight < bitmap.width) {
                bitmap.width - cropRight
            } else {
                bitmap.width
            }
            val croppedBitmap = if (cropWidth < bitmap.width) {
                Bitmap.createBitmap(bitmap, 0, 0, cropWidth, bitmap.height)
            } else {
                bitmap
            }
            
            val origW = croppedBitmap.width.toFloat()
            val origH = croppedBitmap.height.toFloat()
            
            // 2. Letterbox预处理
            val (letterboxed, letterboxParams) = letterboxResize(croppedBitmap, 640)
            if (croppedBitmap !== bitmap) croppedBitmap.recycle()
            
            val scale = letterboxParams[0]
            val padX = letterboxParams[1]
            val padY = letterboxParams[2]
            
            // 3. 预处理像素数据
            val input = preprocess(letterboxed)
            letterboxed.recycle()
            
            // 4. ONNX推理
            val inputName = session?.inputNames?.iterator()?.next() ?: return null
            val inputBuffer = FloatBuffer.wrap(input)
            val inputTensor = OnnxTensor.createTensor(env, inputBuffer, longArrayOf(1, 3, 640, 640))
            val output = session?.run(mapOf(inputName to inputTensor))
            inputTensor.close()
            
            val outputValue = output?.get(0) ?: return null
            
            // 5. 后处理(FloatBuffer方式, 可靠解析)
            val rawDetections = postprocessFloatBuffer(outputValue, origW, origH, scale, padX, padY, confThreshold)
            
            lastDebug = "crop=${cropWidth}x${bitmap.height} letterbox_scale=${"%.3f".format(scale)} raw_dets=${rawDetections.size}"
            
            // 6. NMS
            val detections = nms(rawDetections, 0.45f)
            
            // 7. 手牌区域过滤(基于原始裁剪后图片坐标)
            val handY1 = origH * handRegionY1
            val handY2 = origH * handRegionY2
            
            val handTiles = detections.filter { 
                it.centerY >= handY1 && it.y1 >= handY1 * 0.9f
            }.sortedBy { it.centerX }
            
            val allNames = handTiles.map { it.shortName }
            val avgConf = if (handTiles.isNotEmpty()) handTiles.map { it.confidence }.average().toFloat() else 0f
            
            val debugStr = "$lastDebug | afterNMS=${detections.size} handY=${"%.0f".format(handY1)}-${"%.0f".format(handY2)} hand=${handTiles.size}"
            
            RecognitionResult(handTiles, detections, allNames, avgConf, debugStr)
        } catch (e: Exception) {
            lastError = "识别异常: ${e.message}"
            lastDebug = "exception: ${e.message}"
            null
        }
    }
    
    private fun preprocess(bitmap: Bitmap): FloatArray {
        val floatArray = FloatArray(1 * 3 * 640 * 640)
        val pixels = IntArray(640 * 640)
        bitmap.getPixels(pixels, 0, 640, 0, 0, 640, 640)
        
        for (i in pixels.indices) {
            val pixel = pixels[i]
            floatArray[i] = ((pixel shr 16) and 0xFF) / 255.0f
            floatArray[640 * 640 + i] = ((pixel shr 8) and 0xFF) / 255.0f
            floatArray[2 * 640 * 640 + i] = (pixel and 0xFF) / 255.0f
        }
        return floatArray
    }
    
    /**
     * 使用FloatBuffer解析ONNX输出, 不依赖类型转换
     * 支持 [1, 39, 8400] 和 [1, 8400, 39] 两种格式
     */
    private fun postprocessFloatBuffer(
        outputValue: OnnxValue,
        origW: Float, origH: Float,
        scale: Float, padX: Float, padY: Float,
        confThreshold: Float
    ): List<Detection> {
        val numClasses = 35
        val detections = mutableListOf<Detection>()
        
        // 尝试获取FloatBuffer
        val tensor = outputValue as? OnnxTensor
        if (tensor == null) {
            lastDebug = "output_not_OnnxTensor: ${outputValue.javaClass.simpleName}"
            return detections
        }
        
        val buffer = try {
            tensor.floatBuffer
        } catch (e: Exception) {
            lastDebug = "floatBuffer_failed: ${e.message}"
            return detections
        }
        
        buffer.rewind()
        val totalElements = buffer.remaining()
        
        // 获取shape信息(如果可用)
        val shapeStr = try {
            tensor.info.shape.joinToString(",")
        } catch (e: Exception) {
            "unknown"
        }
        
        lastDebug = "tensor_shape=[$shapeStr] elements=$totalElements"
        
        val data = FloatArray(totalElements)
        buffer.get(data)
        
        // 期望总元素数: 39 * 8400 = 327600
        val expectedSize = (4 + numClasses) * 8400
        if (totalElements != expectedSize) {
            lastDebug += " unexpected_size! expected=$expectedSize"
            // 尝试继续处理
        }
        
        val d1 = mutableListOf<Detection>()
        val d2 = mutableListOf<Detection>()
        
        // 格式1: [39, 8400] 按行优先
        // data[c * 8400 + a] = value at [c][a]
        try {
            parseFlat39x8400(data, 8400, origW, origH, scale, padX, padY, confThreshold, numClasses, d1)
        } catch (e: Exception) {
            lastDebug += " fmt1_err:${e.message}"
        }
        
        // 格式2: [8400, 39] 按行优先
        // data[a * 39 + v] = value at [a][v]
        try {
            parseFlat8400x39(data, 39, origW, origH, scale, padX, padY, confThreshold, numClasses, d2)
        } catch (e: Exception) {
            lastDebug += " fmt2_err:${e.message}"
        }
        
        lastDebug += " fmt1=${d1.size} fmt2=${d2.size}"
        
        detections.addAll(if (d1.size >= d2.size) d1 else d2)
        return detections
    }
    
    /**
     * 解析 [39, 8400] 格式
     * data[c * numAnchors + a]: 第c个通道的第a个anchor
     */
    private fun parseFlat39x8400(
        data: FloatArray, numAnchors: Int,
        origW: Float, origH: Float,
        scale: Float, padX: Float, padY: Float,
        confThreshold: Float, numClasses: Int,
        detections: MutableList<Detection>
    ) {
        for (a in 0 until numAnchors) {
            // 找最大类别分数
            var maxClassScore = 0f
            var maxClassId = 0
            for (c in 0 until numClasses) {
                val idx = (4 + c) * numAnchors + a
                if (idx >= data.size) break
                val score = data[idx]
                if (score > maxClassScore) {
                    maxClassScore = score
                    maxClassId = c
                }
            }
            if (maxClassScore < confThreshold) continue
            
            // bbox (在letterbox 640x640空间)
            val cxIdx = 0 * numAnchors + a
            val cyIdx = 1 * numAnchors + a
            val wIdx = 2 * numAnchors + a
            val hIdx = 3 * numAnchors + a
            if (cxIdx >= data.size || cyIdx >= data.size || wIdx >= data.size || hIdx >= data.size) continue
            
            val cx640 = data[cxIdx]
            val cy640 = data[cyIdx]
            val w640 = data[wIdx]
            val h640 = data[hIdx]
            
            // 从letterbox空间转换到原始图片空间
            val cxOrig = (cx640 - padX) / scale
            val cyOrig = (cy640 - padY) / scale
            val wOrig = w640 / scale
            val hOrig = h640 / scale
            
            // 过滤无效检测(超出图片范围)
            if (cxOrig < 0 || cyOrig < 0 || cxOrig > origW || cyOrig > origH) continue
            if (wOrig <= 0 || hOrig <= 0 || wOrig > origW || hOrig > origH) continue
            
            detections.add(Detection(
                classId = maxClassId,
                className = classNames[maxClassId] ?: "?",
                shortName = shortNames[maxClassId] ?: "?",
                confidence = maxClassScore,
                x1 = cxOrig - wOrig / 2, y1 = cyOrig - hOrig / 2,
                x2 = cxOrig + wOrig / 2, y2 = cyOrig + hOrig / 2,
                centerX = cxOrig,
                centerY = cyOrig
            ))
        }
    }
    
    /**
     * 解析 [8400, 39] 格式
     * data[a * 39 + v]: 第a个anchor的第v个值
     */
    private fun parseFlat8400x39(
        data: FloatArray, numValues: Int,
        origW: Float, origH: Float,
        scale: Float, padX: Float, padY: Float,
        confThreshold: Float, numClasses: Int,
        detections: MutableList<Detection>
    ) {
        val numAnchors = data.size / numValues
        for (a in 0 until numAnchors) {
            val base = a * numValues
            if (base + numValues > data.size) break
            
            // 找最大类别分数
            var maxClassScore = 0f
            var maxClassId = 0
            for (c in 0 until numClasses) {
                val score = data[base + 4 + c]
                if (score > maxClassScore) {
                    maxClassScore = score
                    maxClassId = c
                }
            }
            if (maxClassScore < confThreshold) continue
            
            // bbox (在letterbox 640x640空间)
            val cx640 = data[base + 0]
            val cy640 = data[base + 1]
            val w640 = data[base + 2]
            val h640 = data[base + 3]
            
            // 从letterbox空间转换到原始图片空间
            val cxOrig = (cx640 - padX) / scale
            val cyOrig = (cy640 - padY) / scale
            val wOrig = w640 / scale
            val hOrig = h640 / scale
            
            // 过滤无效检测
            if (cxOrig < 0 || cyOrig < 0 || cxOrig > origW || cyOrig > origH) continue
            if (wOrig <= 0 || hOrig <= 0 || wOrig > origW || hOrig > origH) continue
            
            detections.add(Detection(
                classId = maxClassId,
                className = classNames[maxClassId] ?: "?",
                shortName = shortNames[maxClassId] ?: "?",
                confidence = maxClassScore,
                x1 = cxOrig - wOrig / 2, y1 = cyOrig - hOrig / 2,
                x2 = cxOrig + wOrig / 2, y2 = cyOrig + hOrig / 2,
                centerX = cxOrig,
                centerY = cyOrig
            ))
        }
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
