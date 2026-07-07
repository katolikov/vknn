// Save/load the optimized vknn graph (post-import, post-passes) as a compact binary ".vxm" so a
// reload skips both ONNX protobuf parsing and all graph passes. Self-contained (embeds weights).
// Complements the weight/pipeline/tuning caches for fast warm starts.
//
// A ".vxm" holds either ONE graph (the "VXM3" container, a fixed-shape model) or SEVERAL shape
// buckets (the "VXM4" container). A bucket is a full pass+plan graph product for one declared
// input-shape set; buckets may differ in node identity (lowerConv/subpixel/layout-converts depend on
// shape) but share ONE content-deduped initializer pool because weights are shape-independent. A
// single-bucket save writes the VXM3 container verbatim, so a fixed-shape model's on-disk bytes are
// unchanged by the multi-bucket feature; the VXM4 container is written only for two or more buckets.
#include "vknn/graph.h"
#include "vknn/logging.h"
#include "vknn/op.h"
#include <cstdint>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

namespace vknn {

    namespace {
        // Format version guard, stored as the first word of every .vxm. A load whose leading word
        // does not match is rejected: the on-disk layout is field-order- and width-sensitive, so
        // bumping this is the only safe way to evolve the schema. The "3" revision widens the
        // pw_steps encoding from 4-int chain records to 8-int register-DAG records (+pw_outs);
        // older files fail the check rather than mis-parse their fused nodes.
        constexpr uint32_t kMagic = 0x334d5856; // "VXM3"
        // Multi-bucket container magic. A VXM4 file is: this word, the bucket count, a shared
        // initializer pool (content-deduped blobs), then one entry per bucket -- its name and its
        // graph body, whose initializer table references pool blobs by index instead of inlining
        // bytes. Single-bucket files stay VXM3 (byte-identical to legacy). A reader dispatches on the
        // leading word, so old files load as one bucket and new single-bucket files match legacy.
        constexpr uint32_t kMagic4 = 0x344d5856; // "VXM4"

        // Raw fixed-width serialization: every pod()/vec() writes host-endian bytes with no framing.
        // The Reader must consume fields in the exact same order the Writer emits them.
        struct Writer {
            FILE                      *f;
            template <typename T> void pod(const T &v) {
                fwrite(&v, sizeof(T), 1, f);
            }
            void u32(uint32_t v) {
                pod(v);
            }
            void i64(int64_t v) {
                pod(v);
            }
            void f32(float v) {
                pod(v);
            }
            void str(const std::string &s) {
                u32((uint32_t) s.size());
                if (!s.empty())
                {
                    fwrite(s.data(), 1, s.size(), f);
                }
            }
            template <typename T> void vec(const std::vector<T> &v) {
                u32((uint32_t) v.size());
                if (!v.empty())
                {
                    fwrite(v.data(), sizeof(T), v.size(), f);
                }
            }
        };
        struct Reader {
            FILE                   *f;
            // Sticky short-read flag. Once any fread returns fewer elements than requested it latches
            // false and every later read is skipped, so a truncated file yields zero/default-filled
            // fields instead of reading past EOF; callers gate on this after the whole graph is read.
            bool                    ok = true;
            template <typename T> T pod() {
                T v {};
                ok = ok && fread(&v, sizeof(T), 1, f) == 1;
                return v;
            }
            uint32_t u32() {
                return pod<uint32_t>();
            }
            int64_t i64() {
                return pod<int64_t>();
            }
            float f32() {
                return pod<float>();
            }
            std::string str() {
                uint32_t    n = u32();
                std::string s(n, 0);
                if (n)
                {
                    ok = ok && fread(&s[0], 1, n, f) == n;
                }
                return s;
            }
            template <typename T> std::vector<T> vec() {
                uint32_t       n = u32();
                std::vector<T> v(n);
                if (n)
                {
                    ok = ok && fread(v.data(), sizeof(T), n, f) == n;
                }
                return v;
            }
        };

        void writeAttr(Writer &w, const Attr &a) {
            w.u32((uint32_t) a.kind);
            w.i64(a.i);
            w.f32(a.f);
            w.vec(a.ints);
            w.vec(a.floats);
            w.str(a.str);
        }
        Attr readAttr(Reader &r) {
            Attr a;
            a.kind   = (Attr::Kind) r.u32();
            a.i      = r.i64();
            a.f      = r.f32();
            a.ints   = r.vec<int64_t>();
            a.floats = r.vec<float>();
            a.str    = r.str();
            return a;
        }

        // Write the tensor and node tables plus the graph I/O id lists -- everything in a graph body
        // except the initializers, which the two container formats serialize differently (VXM3 inlines
        // bytes per body; VXM4 references a shared pool). The byte layout is exactly the legacy one up
        // to the initializer section, so a VXM3 body written here matches the pre-bucket format.
        void writeGraphStructure(Writer &w, const Graph &g) {
            w.u32((uint32_t) g.tensors.size());
            for (const auto &t: g.tensors)
            {
                w.str(t.name);
                w.vec(t.shape);
                w.u32((uint32_t) t.dtype);
                w.u32((uint32_t) t.format);
                // Pack the three tensor roles into one word: bit0 input, bit1 output, bit2 initializer.
                w.u32((t.isInput ? 1u : 0) | (t.isOutput ? 2u : 0) | (t.isInitializer ? 4u : 0));
            }
            w.u32((uint32_t) g.nodes.size());
            for (const auto &n: g.nodes)
            {
                w.u32((uint32_t) n.type);
                w.str(n.name);
                w.vec(n.inputs);
                w.vec(n.outputs);
                w.u32((uint32_t) n.fusedAct);
                w.f32(n.actLo);
                w.f32(n.actHi);
                // subOp (int32) and the fused-tensor ids (TensorId == int32, kNoTensor when unset) are
                // stored as i64 so the on-disk width is independent of the in-memory type; the load
                // path narrows them back.
                w.i64(n.subOp);
                w.i64(n.fusedResidual);
                w.i64(n.fusedBias);
                w.u32((uint32_t) n.attr.map.size());
                for (const auto &kv: n.attr.map)
                {
                    w.str(kv.first);
                    writeAttr(w, kv.second);
                }
            }
            w.vec(g.inputs);
            w.vec(g.outputs);
        }

        // Read the tensor/node/I/O tables written by writeGraphStructure into a fresh graph body and
        // rebuild the derived name index. Initializers are read separately by the caller.
        void readGraphStructure(Reader &r, Graph &g) {
            uint32_t nt = r.u32();
            g.tensors.resize(nt);
            for (uint32_t i = 0; i < nt; ++i)
            {
                TensorDesc &t   = g.tensors[i];
                t.name          = r.str();
                t.shape         = r.vec<int64_t>();
                t.dtype         = (DType) r.u32();
                t.format        = (TensorFormat) r.u32();
                uint32_t flags  = r.u32();
                t.isInput       = flags & 1;
                t.isOutput      = flags & 2;
                t.isInitializer = flags & 4;
                // tensorByName is a derived index, not serialized: rebuild it from tensor position as
                // tensors are read. Unnamed tensors are addressed only by id and stay out of the map.
                if (!t.name.empty())
                {
                    g.tensorByName[t.name] = (TensorId) i;
                }
            }
            uint32_t nn = r.u32();
            g.nodes.resize(nn);
            for (uint32_t i = 0; i < nn; ++i)
            {
                Node &n         = g.nodes[i];
                n.type          = (OpType) r.u32();
                n.name          = r.str();
                n.inputs        = r.vec<TensorId>();
                n.outputs       = r.vec<TensorId>();
                n.fusedAct      = (ActType) r.u32();
                n.actLo         = r.f32();
                n.actHi         = r.f32();
                n.subOp         = (int32_t) r.i64();
                n.fusedResidual = (TensorId) r.i64();
                n.fusedBias     = (TensorId) r.i64();
                uint32_t na     = r.u32();
                for (uint32_t a = 0; a < na; ++a)
                {
                    std::string k = r.str();
                    n.attr.map[k] = readAttr(r);
                }
            }
            g.inputs  = r.vec<TensorId>();
            g.outputs = r.vec<TensorId>();
        }

        // Atomic publish: a writer opens a sibling ".tmp" path in the same directory and only a fully
        // written file is renamed onto the final path. A compile killed mid-write (Android's low-memory
        // killer can SIGKILL vknn_compile on a large model) leaves only the temp, so a later load never
        // sees a truncated final .vxm. std::rename between two names in one directory is atomic on the
        // same filesystem.
        FILE *openVxmTemp(const std::string &path, std::string &tmpPath) {
            tmpPath = path + ".tmp";
            return fopen(tmpPath.c_str(), "wb");
        }
        // Flush and close the temp, then rename it onto the final path. On any flush/write/close error
        // the temp is removed and the final path is left untouched; returns whether the final file was
        // published.
        bool commitVxmTemp(FILE *f, const std::string &tmpPath, const std::string &path) {
            bool bad = fflush(f) != 0 || ferror(f) != 0;
            if (fclose(f) != 0)
            {
                bad = true;
            }
            if (bad || std::rename(tmpPath.c_str(), path.c_str()) != 0)
            {
                std::remove(tmpPath.c_str());
                return false;
            }
            return true;
        }
    } // namespace

    bool saveGraphBin(const Graph &g, const std::string &path) {
        std::string tmpPath;
        FILE       *f = openVxmTemp(path, tmpPath);
        if (!f)
        {
            VKNN_WARN << "saveGraph: cannot write " << tmpPath;
            return false;
        }
        Writer w {f};
        w.u32(kMagic);
        writeGraphStructure(w, g);
        // initializers (inlined bytes, keyed by tensor id -- the VXM3 layout)
        w.u32((uint32_t) g.initializers.size());
        for (const auto &kv: g.initializers)
        {
            w.i64(kv.first);
            w.vec(kv.second.bytes);
        }
        if (!commitVxmTemp(f, tmpPath, path))
        {
            VKNN_WARN << "saveGraph: failed to finalize " << path;
            return false;
        }
        VKNN_INFO << "saved optimized model -> " << path << " (" << g.nodes.size() << " nodes, " << g.initializers.size() << " weights)";
        return true;
    }

    bool loadGraphBin(Graph &g, const std::string &path) {
        std::vector<Graph>       buckets;
        std::vector<std::string> names;
        if (!loadGraphBinBuckets(buckets, names, path) || buckets.empty())
        {
            return false;
        }
        // Single-graph callers take the first bucket. A VXM3 file has exactly one; a VXM4 file's
        // buckets are all the same model at different shapes, so bucket 0 is a valid representative.
        g = std::move(buckets.front());
        return true;
    }

    bool saveGraphBinBuckets(const std::vector<Graph> &buckets, const std::vector<std::string> &names, const std::string &path) {
        if (buckets.empty())
        {
            VKNN_WARN << "saveGraphBuckets: no buckets to write " << path;
            return false;
        }
        // A single bucket is a fixed-shape model: write the legacy VXM3 container so its on-disk bytes
        // are unchanged by the multi-bucket feature.
        if (buckets.size() == 1)
        {
            return saveGraphBin(buckets.front(), path);
        }
        std::string tmpPath;
        FILE       *f = openVxmTemp(path, tmpPath);
        if (!f)
        {
            VKNN_WARN << "saveGraphBuckets: cannot write " << tmpPath;
            return false;
        }
        Writer w {f};
        w.u32(kMagic4);
        w.u32((uint32_t) buckets.size());
        // Build the content-deduped initializer pool: identical payloads across buckets (the common
        // case -- weights are shape-independent) collapse to one blob. A blob is keyed by its raw
        // bytes; a bucket then references its initializers by pool index.
        std::vector<const std::vector<uint8_t> *> pool;
        std::map<std::string, uint32_t>           poolIndex; // payload bytes -> pool slot
        auto                                      internPayload = [&](const std::vector<uint8_t> &bytes) -> uint32_t {
            std::string key(bytes.begin(), bytes.end());
            auto        it = poolIndex.find(key);
            if (it != poolIndex.end())
            {
                return it->second;
            }
            uint32_t slot = (uint32_t) pool.size();
            pool.push_back(&bytes);
            poolIndex.emplace(std::move(key), slot);
            return slot;
        };
        // Reference-map every bucket's initializers into the pool first, so the pool blob table is
        // written once ahead of the bucket bodies.
        std::vector<std::vector<std::pair<int64_t, uint32_t>>> bucketInits(buckets.size());
        for (size_t b = 0; b < buckets.size(); ++b)
        {
            for (const auto &kv: buckets[b].initializers)
            {
                bucketInits[b].emplace_back(kv.first, internPayload(kv.second.bytes));
            }
        }
        // Shared pool: one blob per distinct payload.
        w.u32((uint32_t) pool.size());
        for (const auto *blob: pool)
        {
            w.vec(*blob);
        }
        // Per bucket: name, graph structure, then its (tensor id -> pool index) initializer table.
        int64_t totalWeights = 0;
        for (size_t b = 0; b < buckets.size(); ++b)
        {
            w.str(b < names.size() ? names[b] : std::string());
            writeGraphStructure(w, buckets[b]);
            w.u32((uint32_t) bucketInits[b].size());
            for (const auto &ref: bucketInits[b])
            {
                w.i64(ref.first);
                w.u32(ref.second);
            }
            totalWeights += (int64_t) bucketInits[b].size();
        }
        if (!commitVxmTemp(f, tmpPath, path))
        {
            VKNN_WARN << "saveGraphBuckets: failed to finalize " << path;
            return false;
        }
        VKNN_INFO << "saved optimized model -> " << path << " (" << buckets.size() << " buckets, " << pool.size() << " shared weights, " << totalWeights << " refs)";
        return true;
    }

    bool loadGraphBinBuckets(std::vector<Graph> &buckets, std::vector<std::string> &names, const std::string &path) {
        FILE *f = fopen(path.c_str(), "rb");
        if (!f)
        {
            return false;
        }
        Reader   r {f};
        uint32_t magic = r.u32();
        if (magic == kMagic)
        {
            // Legacy single-bucket container: one graph, initializers inlined.
            buckets.clear();
            names.clear();
            buckets.emplace_back();
            names.emplace_back();
            Graph &g = buckets.back();
            readGraphStructure(r, g);
            uint32_t ni = r.u32();
            for (uint32_t i = 0; i < ni; ++i)
            {
                TensorId   id = (TensorId) r.i64();
                HostBuffer hb;
                hb.bytes           = r.vec<uint8_t>();
                g.initializers[id] = std::move(hb);
            }
            fclose(f);
            if (!r.ok)
            {
                VKNN_WARN << "loadGraph: truncated " << path;
                return false;
            }
            return true;
        }
        if (magic != kMagic4)
        {
            fclose(f);
            // Every VXM magic is the ASCII "VXM<version>" little-endian, so the low three bytes spell
            // "VXM" and the top byte is the version digit. A recognizable "VXM<n>" with n outside {3,4}
            // was written by a different engine version (recompile from the .onnx); anything else is not
            // a .vxm at all (truncated, corrupt, or the wrong file). Both name the fix in the message so
            // a stale or foreign file explains itself instead of just "bad magic".
            constexpr uint32_t kVxmPrefix = 0x004d5856; // "VXM" -- low 3 bytes shared by every VXM<n>
            if ((magic & 0x00ffffffu) == kVxmPrefix)
            {
                VKNN_WARN << "loadGraph: " << path << " is a VXM" << (char) (magic >> 24)
                          << " container from an incompatible vknn version (this build reads VXM3/VXM4)"
                          << " -- reconvert the model from its .onnx with the current vknn_compile";
            } else
            {
                VKNN_WARN << "loadGraph: bad magic in " << path
                          << " -- not a valid .vxm (truncated, corrupt, or the wrong file)"
                          << " -- reconvert from the .onnx with the current vknn_compile";
            }
            return false;
        }
        // Multi-bucket container: read the shared pool, then each bucket's structure + pool refs.
        uint32_t nb = r.u32();
        uint32_t np = r.u32();
        std::vector<std::vector<uint8_t>> pool(np);
        for (uint32_t i = 0; i < np; ++i)
        {
            pool[i] = r.vec<uint8_t>();
        }
        buckets.clear();
        names.clear();
        buckets.resize(nb);
        names.resize(nb);
        for (uint32_t b = 0; b < nb; ++b)
        {
            names[b] = r.str();
            Graph &g = buckets[b];
            readGraphStructure(r, g);
            uint32_t ni = r.u32();
            for (uint32_t i = 0; i < ni; ++i)
            {
                TensorId id  = (TensorId) r.i64();
                uint32_t idx = r.u32();
                HostBuffer hb;
                if (idx < pool.size())
                {
                    hb.bytes = pool[idx]; // shared payload copied into this bucket's initializer map
                }
                g.initializers[id] = std::move(hb);
            }
        }
        fclose(f);
        if (!r.ok)
        {
            VKNN_WARN << "loadGraph: truncated " << path;
            return false;
        }
        return true;
    }

} // namespace vknn
