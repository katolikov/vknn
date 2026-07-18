// MessagePack (de)serialization of the warm-start model cache. The only file that touches the msgpack
// API; everything else sees the typed CacheDoc from cache_codec.h.
#include "core/cache_codec.h"
#include <cstring>
#include <msgpack.h>

namespace vknn {

    // ---------------------------------------------------------------- encode
    static void packStr(msgpack_packer *pk, const std::string &s) {
        msgpack_pack_str(pk, s.size());
        msgpack_pack_str_body(pk, s.data(), s.size());
    }
    static void packKey(msgpack_packer *pk, const char *k) {
        size_t n = std::strlen(k);
        msgpack_pack_str(pk, n);
        msgpack_pack_str_body(pk, k, n);
    }
    static void packBin(msgpack_packer *pk, const void *d, size_t n) {
        msgpack_pack_bin(pk, n);
        msgpack_pack_bin_body(pk, d, n);
    }

    std::vector<uint8_t> cacheEncode(const CacheDoc &doc) {
        msgpack_sbuffer sb;
        msgpack_sbuffer_init(&sb);
        msgpack_packer pk;
        msgpack_packer_init(&pk, &sb, msgpack_sbuffer_write);

        // The map header declares its entry count up front, so this literal must equal the number of
        // key/value pairs packed below; a mismatch produces a malformed stream. Decode is name-keyed
        // (mapGet), so field order here is not load-bearing, but the count is.
        msgpack_pack_map(&pk, 8);
        packKey(&pk, "format");
        msgpack_pack_uint32(&pk, doc.format);
        packKey(&pk, "kernelHash");
        packStr(&pk, doc.kernelHash);
        packKey(&pk, "vendorId");
        msgpack_pack_uint32(&pk, doc.vendorId);
        packKey(&pk, "deviceId");
        msgpack_pack_uint32(&pk, doc.deviceId);
        packKey(&pk, "driverVersion");
        msgpack_pack_uint32(&pk, doc.driverVersion);
        packKey(&pk, "pipelineCacheUUID");
        packBin(&pk, doc.pipelineCacheUUID.data(), doc.pipelineCacheUUID.size());
        packKey(&pk, "model");
        packStr(&pk, doc.model);
        packKey(&pk, "variants");
        msgpack_pack_array(&pk, doc.variants.size());
        for (const auto &v: doc.variants)
        {
            // Per-variant map: the count must match the key/value pairs packed below (the key fields
            // plus pipeline, weights, tune, tunelvl). Adding a field means bumping this literal in lockstep.
            msgpack_pack_map(&pk, 18);
            packKey(&pk, "precision");
            packStr(&pk, v.precision);
            packKey(&pk, "flatLayout");
            v.flatLayout ? msgpack_pack_true(&pk) : msgpack_pack_false(&pk);
            packKey(&pk, "gpuIslandFold");
            v.gpuIslandFold ? msgpack_pack_true(&pk) : msgpack_pack_false(&pk);
            packKey(&pk, "matmulViewFold");
            v.matmulViewFold ? msgpack_pack_true(&pk) : msgpack_pack_false(&pk);
            packKey(&pk, "ropeFusion");
            v.ropeFusion ? msgpack_pack_true(&pk) : msgpack_pack_false(&pk);
            packKey(&pk, "fusedAttention");
            v.fusedAttention ? msgpack_pack_true(&pk) : msgpack_pack_false(&pk);
            packKey(&pk, "kvConcatFold");
            v.kvConcatFold ? msgpack_pack_true(&pk) : msgpack_pack_false(&pk);
            packKey(&pk, "fp32Tensors");
            packStr(&pk, v.fp32Tensors);
            packKey(&pk, "winograd");
            msgpack_pack_int32(&pk, v.winograd);
            packKey(&pk, "winogradVariant");
            msgpack_pack_int32(&pk, v.winogradVariant);
            packKey(&pk, "winogradUnit");
            msgpack_pack_int32(&pk, v.winogradUnit);
            packKey(&pk, "directConv3x3");
            msgpack_pack_int32(&pk, v.directConv3x3);
            packKey(&pk, "splitKConv");
            msgpack_pack_int32(&pk, v.splitKConv);
            packKey(&pk, "coopmatGemm");
            msgpack_pack_int32(&pk, v.coopmatGemm);
            packKey(&pk, "pipeline");
            packBin(&pk, v.pipeline.data(), v.pipeline.size());
            packKey(&pk, "weights");
            msgpack_pack_map(&pk, v.weights.size());
            for (const auto &kv: v.weights)
            {
                // Each float vector is stored as its raw little-endian bytes; the decoder relies on the
                // blob length being a whole multiple of sizeof(float) to reconstruct the vector.
                packStr(&pk, kv.first);
                packBin(&pk, kv.second.data(), kv.second.size() * sizeof(float));
            }
            packKey(&pk, "tune");
            msgpack_pack_map(&pk, v.tune.size());
            for (const auto &kv: v.tune)
            {
                packStr(&pk, kv.first);
                msgpack_pack_int32(&pk, kv.second);
            }
            // Append-only companion to "tune": the Tuning level each entry was measured at. An older
            // reader ignores this key; an older cache omits it and the decoder leaves tuneLevel empty.
            packKey(&pk, "tunelvl");
            msgpack_pack_map(&pk, v.tuneLevel.size());
            for (const auto &kv: v.tuneLevel)
            {
                packStr(&pk, kv.first);
                msgpack_pack_int32(&pk, kv.second);
            }
        }

        std::vector<uint8_t> out((const uint8_t *) sb.data, (const uint8_t *) sb.data + sb.size);
        msgpack_sbuffer_destroy(&sb);
        return out;
    }

    // ---------------------------------------------------------------- decode
    static const msgpack_object *mapGet(const msgpack_object &m, const char *key) {
        if (m.type != MSGPACK_OBJECT_MAP)
        {
            return nullptr;
        }
        size_t klen = std::strlen(key);
        for (uint32_t i = 0; i < m.via.map.size; ++i)
        {
            const msgpack_object_kv &kv = m.via.map.ptr[i];
            if (kv.key.type == MSGPACK_OBJECT_STR && kv.key.via.str.size == klen && std::memcmp(kv.key.via.str.ptr, key, klen) == 0)
            {
                return &kv.val;
            }
        }
        return nullptr;
    }
    static std::string getStr(const msgpack_object &m, const char *key) {
        const msgpack_object *o = mapGet(m, key);
        if (o && o->type == MSGPACK_OBJECT_STR)
        {
            return std::string(o->via.str.ptr, o->via.str.size);
        }
        return {};
    }
    static uint32_t getU32(const msgpack_object &m, const char *key) {
        const msgpack_object *o = mapGet(m, key);
        return (o && o->type == MSGPACK_OBJECT_POSITIVE_INTEGER) ? (uint32_t) o->via.u64 : 0u;
    }
    // msgpack stores a non-negative int32 as a POSITIVE_INTEGER and only negative values as
    // NEGATIVE_INTEGER, so both object types must be accepted to round-trip a signed field.
    static int32_t asI32(const msgpack_object &o) {
        if (o.type == MSGPACK_OBJECT_POSITIVE_INTEGER)
        {
            return (int32_t) o.via.u64;
        }
        if (o.type == MSGPACK_OBJECT_NEGATIVE_INTEGER)
        {
            return (int32_t) o.via.i64;
        }
        return 0;
    }
    static int32_t getI32(const msgpack_object &m, const char *key) {
        const msgpack_object *o = mapGet(m, key);
        return o ? asI32(*o) : 0;
    }
    static bool getBool(const msgpack_object &m, const char *key, bool dflt) {
        const msgpack_object *o = mapGet(m, key);
        return (o && o->type == MSGPACK_OBJECT_BOOLEAN) ? o->via.boolean : dflt;
    }
    static std::vector<uint8_t> getBin(const msgpack_object &m, const char *key) {
        const msgpack_object *o = mapGet(m, key);
        if (o && o->type == MSGPACK_OBJECT_BIN && (o->via.bin.ptr || o->via.bin.size == 0))
        {
            return std::vector<uint8_t>((const uint8_t *) o->via.bin.ptr, (const uint8_t *) o->via.bin.ptr + o->via.bin.size);
        }
        return {};
    }

    bool cacheDecode(const uint8_t *data, size_t n, CacheDoc &out) {
        if (!data || n == 0)
        {
            return false;
        }
        msgpack_unpacked und;
        msgpack_unpacked_init(&und);
        msgpack_unpack_return ret = msgpack_unpack_next(&und, (const char *) data, n, nullptr);
        if (ret != MSGPACK_UNPACK_SUCCESS || und.data.type != MSGPACK_OBJECT_MAP)
        {
            msgpack_unpacked_destroy(&und);
            return false; // truncated, malformed, or a legacy (non-MessagePack) cache file
        }
        const msgpack_object &root = und.data;

        out.format            = getU32(root, "format");
        out.kernelHash        = getStr(root, "kernelHash");
        out.vendorId          = getU32(root, "vendorId");
        out.deviceId          = getU32(root, "deviceId");
        out.driverVersion     = getU32(root, "driverVersion");
        out.pipelineCacheUUID = getBin(root, "pipelineCacheUUID");
        out.model             = getStr(root, "model");
        out.variants.clear();

        const msgpack_object *va = mapGet(root, "variants");
        if (va && va->type == MSGPACK_OBJECT_ARRAY)
        {
            for (uint32_t i = 0; i < va->via.array.size; ++i)
            {
                const msgpack_object &vo = va->via.array.ptr[i];
                if (vo.type != MSGPACK_OBJECT_MAP)
                {
                    continue;
                }
                CacheVariant v;
                // The bool defaults mirror CacheVariant's struct defaults so a variant written by an
                // older codec that lacked these keys decodes to the same key() and stays interchangeable.
                v.precision       = getStr(vo, "precision");
                v.flatLayout      = getBool(vo, "flatLayout", true);
                v.gpuIslandFold   = getBool(vo, "gpuIslandFold", true);
                v.matmulViewFold  = getBool(vo, "matmulViewFold", true);
                v.ropeFusion      = getBool(vo, "ropeFusion", true);
                v.fusedAttention  = getBool(vo, "fusedAttention", true);
                v.kvConcatFold    = getBool(vo, "kvConcatFold", false);
                v.fp32Tensors     = getStr(vo, "fp32Tensors");
                v.winograd        = getI32(vo, "winograd");
                v.winogradVariant = getI32(vo, "winogradVariant");
                v.winogradUnit    = getI32(vo, "winogradUnit");
                v.directConv3x3   = getI32(vo, "directConv3x3");
                v.splitKConv      = getI32(vo, "splitKConv");
                v.coopmatGemm     = getI32(vo, "coopmatGemm");
                v.pipeline        = getBin(vo, "pipeline");

                if (const msgpack_object *w = mapGet(vo, "weights"); w && w->type == MSGPACK_OBJECT_MAP)
                {
                    for (uint32_t k = 0; k < w->via.map.size; ++k)
                    {
                        const msgpack_object_kv &kv = w->via.map.ptr[k];
                        if (kv.key.type != MSGPACK_OBJECT_STR || kv.val.type != MSGPACK_OBJECT_BIN)
                        {
                            continue;
                        }
                        size_t bytes = kv.val.via.bin.size;
                        // A well-formed cache always stores whole floats; skip a corrupt/misaligned blob
                        // (it is simply recomputed at load) rather than truncate it silently.
                        if (bytes % sizeof(float) != 0 || (bytes && !kv.val.via.bin.ptr))
                        {
                            continue;
                        }
                        std::string        name(kv.key.via.str.ptr, kv.key.via.str.size);
                        std::vector<float> f(bytes / sizeof(float));
                        if (bytes)
                        {
                            std::memcpy(f.data(), kv.val.via.bin.ptr, bytes);
                        }
                        v.weights.emplace(std::move(name), std::move(f));
                    }
                }
                if (const msgpack_object *t = mapGet(vo, "tune"); t && t->type == MSGPACK_OBJECT_MAP)
                {
                    for (uint32_t k = 0; k < t->via.map.size; ++k)
                    {
                        const msgpack_object_kv &kv = t->via.map.ptr[k];
                        if (kv.key.type != MSGPACK_OBJECT_STR)
                        {
                            continue;
                        }
                        v.tune.emplace(std::string(kv.key.via.str.ptr, kv.key.via.str.size), asI32(kv.val));
                    }
                }
                if (const msgpack_object *t = mapGet(vo, "tunelvl"); t && t->type == MSGPACK_OBJECT_MAP)
                {
                    for (uint32_t k = 0; k < t->via.map.size; ++k)
                    {
                        const msgpack_object_kv &kv = t->via.map.ptr[k];
                        if (kv.key.type != MSGPACK_OBJECT_STR)
                        {
                            continue;
                        }
                        v.tuneLevel.emplace(std::string(kv.key.via.str.ptr, kv.key.via.str.size), asI32(kv.val));
                    }
                }
                out.variants.push_back(std::move(v));
            }
        }
        msgpack_unpacked_destroy(&und);
        return true;
    }

} // namespace vknn
