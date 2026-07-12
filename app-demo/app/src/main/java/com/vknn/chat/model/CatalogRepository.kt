package com.vknn.chat.model

import android.app.Application
import android.util.Log
import com.vknn.chat.ChatPromptTemplate
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import java.io.File
import java.net.HttpURLConnection
import java.net.URL

// The effective model catalogue: the built-in [ModelCatalog.BUILTIN] list merged with a remote
// catalog.json the maintainer edits. Shipping a new model becomes a one-line manifest edit instead of
// an APK rebuild. The built-in list is the offline / first-launch seed; a cached copy of the last good
// manifest makes the effective list correct before the network fetch returns. A failed or malformed
// fetch never clobbers the current value.
class CatalogRepository(private val app: Application) {
    private val cacheFile = File(app.filesDir, CACHE_NAME)
    private val knownDialectIds = ChatPromptTemplate.DIALECTS.map { it.id }.toSet()
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)

    private val _catalog = MutableStateFlow(CatalogManifest.merge(ModelCatalog.BUILTIN, readCache()))
    val catalog: StateFlow<List<ModelSpec>> = _catalog.asStateFlow()

    /** Kick off the single background refresh from the remote manifest; call once at app start. */
    fun start() {
        scope.launch { refresh() }
    }

    private fun readCache(): List<ModelSpec> =
        if (cacheFile.exists()) {
            runCatching { CatalogManifest.parse(cacheFile.readText(), knownDialectIds) }.getOrDefault(emptyList())
        } else {
            emptyList()
        }

    private fun refresh() {
        val json = runCatching { fetch(CATALOG_URL) }.getOrElse {
            Log.i(TAG, "refresh: fetch failed (${it.javaClass.simpleName}); keeping the current list")
            return
        }
        val remote = CatalogManifest.parse(json, knownDialectIds)
        if (remote.isEmpty()) {
            Log.i(TAG, "refresh: manifest empty or malformed; keeping the current list")
            return // keep the current effective list
        }
        runCatching { cacheFile.writeText(json) }
        _catalog.value = CatalogManifest.merge(ModelCatalog.BUILTIN, remote)
        Log.i(TAG, "refresh: ${remote.size} model(s) from remote, effective catalogue ${_catalog.value.size}")
    }

    private fun fetch(url: String): String {
        val conn = (URL(url).openConnection() as HttpURLConnection).apply {
            connectTimeout = 15_000
            readTimeout = 20_000
            setRequestProperty("User-Agent", "vknn-chat")
        }
        try {
            if (conn.responseCode != HttpURLConnection.HTTP_OK) error("HTTP ${conn.responseCode}")
            return conn.inputStream.bufferedReader().use { it.readText() }
        } finally {
            conn.disconnect()
        }
    }

    companion object {
        const val CATALOG_URL = "https://raw.githubusercontent.com/katolikov/vknn/main/app-demo/catalog.json"
        private const val CACHE_NAME = "catalog-cache.json"
        private const val TAG = "CatalogRepository"
    }
}
