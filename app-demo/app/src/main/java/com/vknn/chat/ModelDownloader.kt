package com.vknn.chat

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.net.HttpURLConnection
import java.net.URL

// Downloads the compiled Qwen .vxm from the katolikov/qwen-vknn HuggingFace repo into app storage, with
// progress. Only the model downloads; the tokenizer ships in the APK assets.
object ModelDownloader {
    const val MODEL_URL = "https://huggingface.co/katolikov/qwen-vknn/resolve/main/qwen2.5-coder-0.5b-decode-c256.vxm"
    const val FILE_NAME = "qwen2.5-coder-0.5b-decode-c256.vxm"
    const val EXPECTED_BYTES = 1261050516L

    fun modelFile(dir: File): File = File(dir, FILE_NAME)

    fun isPresent(dir: File): Boolean {
        val f = modelFile(dir)
        return f.exists() && f.length() == EXPECTED_BYTES
    }

    // onProgress(percent, bytesDone, bytesTotal). Follows HF's https redirect to the CDN.
    suspend fun download(dir: File, onProgress: (Int, Long, Long) -> Unit) = withContext(Dispatchers.IO) {
        val out = modelFile(dir)
        if (isPresent(dir)) {
            onProgress(100, out.length(), out.length())
            return@withContext
        }
        val tmp = File(dir, "$FILE_NAME.part")
        tmp.delete()
        var url = MODEL_URL
        var conn = open(url)
        var redirects = 0
        while (conn.responseCode in 300..399 && redirects < 5) {
            url = conn.getHeaderField("Location") ?: break
            conn.disconnect()
            conn = open(url)
            redirects++
        }
        val total = if (conn.contentLengthLong > 0) conn.contentLengthLong else EXPECTED_BYTES
        conn.inputStream.use { input ->
            tmp.outputStream().use { output ->
                val buf = ByteArray(1 shl 20)
                var done = 0L
                var lastPct = -1
                while (true) {
                    val n = input.read(buf)
                    if (n < 0) break
                    output.write(buf, 0, n)
                    done += n
                    val pct = ((done * 100) / total).toInt()
                    if (pct != lastPct) {
                        lastPct = pct
                        onProgress(pct, done, total)
                    }
                }
            }
        }
        conn.disconnect()
        if (!tmp.renameTo(out)) {
            tmp.copyTo(out, overwrite = true)
            tmp.delete()
        }
        onProgress(100, out.length(), out.length())
    }

    private fun open(url: String): HttpURLConnection =
        (URL(url).openConnection() as HttpURLConnection).apply {
            connectTimeout = 20000
            readTimeout = 60000
            instanceFollowRedirects = true
            setRequestProperty("User-Agent", "vknn-chat/1.0")
            connect()
        }
}
