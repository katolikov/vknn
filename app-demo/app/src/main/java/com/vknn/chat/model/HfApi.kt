package com.vknn.chat.model

import org.json.JSONObject
import java.net.HttpURLConnection
import java.net.URL

internal const val HF_USER_AGENT = "vknn-app/1.0"

// Exact byte length + sha256 of a repo file, as reported by the HF model API.
data class HfFileInfo(val size: Long, val sha256: String?)

// Queries https://huggingface.co/api/models/<repo>?blobs=true, which lists every repo file with its
// byte length and, for LFS files, the sha256 the finished download is verified against.
object HfApi {
    fun fetch(spec: ModelSpec): HfFileInfo? = try {
        val conn = URL("https://huggingface.co/api/models/${spec.repoId}?blobs=true")
            .openConnection() as HttpURLConnection
        conn.connectTimeout = 15_000
        conn.readTimeout = 30_000
        conn.setRequestProperty("User-Agent", HF_USER_AGENT)
        try {
            parse(conn.inputStream.bufferedReader().use { it.readText() }, spec.repoFile)
        } finally {
            conn.disconnect()
        }
    } catch (_: Exception) {
        null // offline or repo not (yet) published; the caller falls back to the catalogue values
    }

    // Pure JSON walk over the "siblings" file list; unit-tested on the JVM.
    fun parse(json: String, repoFile: String): HfFileInfo? {
        val siblings = JSONObject(json).optJSONArray("siblings") ?: return null
        for (i in 0 until siblings.length()) {
            val s = siblings.getJSONObject(i)
            if (s.optString("rfilename") != repoFile) continue
            val lfs = s.optJSONObject("lfs")
            val lfsSize = lfs?.optLong("size", -1L) ?: -1L
            val size = if (lfsSize > 0) lfsSize else s.optLong("size", -1L)
            if (size <= 0) return null
            val sha = lfs?.optString("sha256")
            return HfFileInfo(size, if (sha.isNullOrEmpty()) null else sha)
        }
        return null
    }
}
