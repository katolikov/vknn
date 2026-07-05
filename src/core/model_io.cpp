// Save/load the optimized vknn graph (post-import, post-passes) as a compact binary ".vxm" so a
// reload skips both ONNX protobuf parsing and all graph passes. Self-contained (embeds weights).
// Complements the weight/pipeline/tuning caches for fast warm starts.
#include "vknn/graph.h"
#include "vknn/logging.h"
#include "vknn/op.h"
#include <cstdint>
#include <cstdio>
#include <string>

namespace vknn {

    namespace {
        // Format version guard, stored as the first word of every .vxm. A load whose leading word
        // does not match is rejected: the on-disk layout is field-order- and width-sensitive, so
        // bumping this is the only safe way to evolve the schema. The "3" revision widens the
        // pw_steps encoding from 4-int chain records to 8-int register-DAG records (+pw_outs);
        // older files fail the check rather than mis-parse their fused nodes.
        constexpr uint32_t kMagic = 0x334d5856; // "VXM3"

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
    } // namespace

    bool saveGraphBin(const Graph &g, const std::string &path) {
        FILE *f = fopen(path.c_str(), "wb");
        if (!f)
        {
            VKNN_WARN << "saveGraph: cannot write " << path;
            return false;
        }
        Writer w {f};
        w.u32(kMagic);
        // tensors
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
        // nodes
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
            // stored as i64 so the on-disk width is independent of the in-memory type; the load path
            // narrows them back.
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
        // initializers
        w.u32((uint32_t) g.initializers.size());
        for (const auto &kv: g.initializers)
        {
            w.i64(kv.first);
            w.vec(kv.second.bytes);
        }
        fclose(f);
        VKNN_INFO << "saved optimized model -> " << path << " (" << g.nodes.size() << " nodes, " << g.initializers.size() << " weights)";
        return true;
    }

    bool loadGraphBin(Graph &g, const std::string &path) {
        FILE *f = fopen(path.c_str(), "rb");
        if (!f)
        {
            return false;
        }
        Reader r {f};
        if (r.u32() != kMagic)
        {
            fclose(f);
            VKNN_WARN << "loadGraph: bad magic in " << path;
            return false;
        }
        g           = Graph {};
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
        g.inputs    = r.vec<TensorId>();
        g.outputs   = r.vec<TensorId>();
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

} // namespace vknn
