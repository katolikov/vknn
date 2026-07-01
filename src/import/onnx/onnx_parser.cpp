// minimal, dependency-free ONNX (protobuf) importer.
//
// Parses the protobuf wire format directly (varint / length-delimited / fixed32/64),
// reading only the fields vknn needs: GraphProto nodes, initializers (TensorProto),
// inputs/outputs (ValueInfoProto), and NodeProto attributes. Sufficient for MobileNetV2
// and CNNs in general. No protobuf library / no generated code required.
#include "vknn/graph.h"
#include "vknn/logging.h"
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

namespace vknn {
    namespace onnx {

        // ----------------------------- wire reader -----------------------------
        class Reader {
          public:
            Reader(const uint8_t *p, size_t n): p_(p), end_(p + n) {
            }
            bool eof() const {
                return p_ >= end_;
            }

            uint64_t varint() {
                uint64_t v     = 0;
                int      shift = 0;
                while (p_ < end_)
                {
                    uint8_t b = *p_++;
                    v |= (uint64_t) (b & 0x7F) << shift;
                    if (!(b & 0x80))
                    {
                        break;
                    }
                    shift += 7;
                }
                return v;
            }
            uint32_t fixed32() {
                uint32_t v = 0;
                std::memcpy(&v, p_, 4);
                p_ += 4;
                return v;
            }
            uint64_t fixed64() {
                uint64_t v = 0;
                std::memcpy(&v, p_, 8);
                p_ += 8;
                return v;
            }
            // returns (field number, wire type)
            bool tag(uint32_t &field, uint32_t &wire) {
                if (eof())
                {
                    return false;
                }
                uint64_t t = varint();
                field      = (uint32_t) (t >> 3);
                wire       = (uint32_t) (t & 7);
                return true;
            }
            // length-delimited region
            Reader sub() {
                uint64_t len = varint();
                Reader   r(p_, (size_t) len);
                p_ += len;
                return r;
            }
            std::string str() {
                uint64_t    len = varint();
                std::string s((const char *) p_, (size_t) len);
                p_ += len;
                return s;
            }
            std::vector<uint8_t> bytes() {
                uint64_t             len = varint();
                std::vector<uint8_t> b(p_, p_ + len);
                p_ += len;
                return b;
            }
            // skip a field of given wire type
            void skip(uint32_t wire) {
                switch (wire)
                {
                    case 0:
                        varint();
                        break;
                    case 1:
                        p_ += 8;
                        break;
                    case 5:
                        p_ += 4;
                        break;
                    case 2: {
                        uint64_t l = varint();
                        p_ += l;
                        break;
                    }
                    default:
                        break;
                }
            }
            const uint8_t *cur() const {
                return p_;
            }
            const uint8_t *end() const {
                return end_;
            }

          private:
            const uint8_t *p_;
            const uint8_t *end_;
        };

        // ----------------------------- TensorProto -----------------------------
        // fields: 1=dims(int64 repeated/packed), 2=data_type(int32), 4=float_data(packed),
        // 7=int64_data(packed), 8=name(string), 9=raw_data(bytes), 13=external_data
        // (repeated StringStringEntryProto{1=key,2=value}), 14=data_location(0=DEFAULT,1=EXTERNAL).
        // Large models (incl. anything torch's newer exporter emits) keep weights in a sibling .onnx.data
        // file and reference them via external_data; resolved against the model dir at materialize time.
        struct TensorProto {
            std::vector<int64_t> dims;
            int32_t              dataType = 1; // 1=FLOAT, 7=INT64
            std::string          name;
            std::vector<uint8_t> raw;
            std::vector<float>   floatData;
            std::vector<int64_t> int64Data;
            int32_t              dataLocation = 0; // 1 = EXTERNAL
            std::string          extLoc;           // external file (relative to the model dir)
            int64_t              extOffset = 0;
            int64_t              extLength = -1;
        };

        static TensorProto parseTensor(Reader r) {
            TensorProto t;
            uint32_t    f, w;
            while (r.tag(f, w))
            {
                switch (f)
                {
                    case 1: // dims
                        if (w == 2)
                        {
                            Reader s = r.sub();
                            while (!s.eof())
                            {
                                t.dims.push_back((int64_t) s.varint());
                            }
                        } else
                        {
                            t.dims.push_back((int64_t) r.varint());
                        }
                        break;
                    case 2:
                        t.dataType = (int32_t) r.varint();
                        break;
                    case 4: // float_data (packed or single)
                        if (w == 2)
                        {
                            Reader s = r.sub();
                            while (!s.eof())
                            {
                                uint32_t b = s.fixed32();
                                float    fv;
                                std::memcpy(&fv, &b, 4);
                                t.floatData.push_back(fv);
                            }
                        } else
                        {
                            uint32_t b = r.fixed32();
                            float    fv;
                            std::memcpy(&fv, &b, 4);
                            t.floatData.push_back(fv);
                        }
                        break;
                    case 7: // int64_data
                        if (w == 2)
                        {
                            Reader s = r.sub();
                            while (!s.eof())
                            {
                                t.int64Data.push_back((int64_t) s.varint());
                            }
                        } else
                        {
                            t.int64Data.push_back((int64_t) r.varint());
                        }
                        break;
                    case 8:
                        t.name = r.str();
                        break;
                    case 9:
                        t.raw = r.bytes();
                        break;
                    case 13: { // external_data: StringStringEntryProto { 1=key, 2=value }
                        Reader      s = r.sub();
                        std::string key, val;
                        uint32_t    ef, ew;
                        while (s.tag(ef, ew))
                        {
                            if (ef == 1 && ew == 2)
                            {
                                key = s.str();
                            } else if (ef == 2 && ew == 2)
                            {
                                val = s.str();
                            } else
                            {
                                s.skip(ew);
                            }
                        }
                        if (key == "location")
                        {
                            t.extLoc = val;
                        } else if (key == "offset")
                        {
                            t.extOffset = std::strtoll(val.c_str(), nullptr, 10);
                        } else if (key == "length")
                        { t.extLength = std::strtoll(val.c_str(), nullptr, 10); }
                        break;
                    }
                    case 14: // data_location
                        t.dataLocation = (int32_t) r.varint();
                        break;
                    default:
                        r.skip(w);
                        break;
                }
            }
            return t;
        }

        // Resolve an EXTERNAL tensor (data_location==1) by reading its byte range from the sibling data
        // file into t.raw, so the normal raw_data paths below handle it. The data file is read once and
        // cached (one .onnx.data backs every weight). No-op if the tensor isn't external or already inline.
        static void resolveExternal(const std::string &baseDir, TensorProto &t, std::map<std::string, std::vector<uint8_t>> &cache) {
            if (t.dataLocation != 1 || t.extLoc.empty() || !t.raw.empty())
            {
                return;
            }
            std::string path = baseDir.empty() ? t.extLoc : baseDir + "/" + t.extLoc;
            auto        it   = cache.find(path);
            if (it == cache.end())
            {
                std::ifstream        f(path, std::ios::binary);
                std::vector<uint8_t> buf;
                if (f)
                {
                    buf.assign((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
                } else
                {
                    VKNN_ERROR << "external data file not found: " << path << " (for tensor '" << t.name << "')";
                }
                it = cache.emplace(path, std::move(buf)).first;
            }
            const std::vector<uint8_t> &file = it->second;
            int64_t                     off  = t.extOffset;
            int64_t                     len  = t.extLength >= 0 ? t.extLength : (int64_t) file.size() - off;
            if (off < 0 || len < 0 || off + len > (int64_t) file.size())
            {
                VKNN_ERROR << "external data range [" << off << "," << off + len << ") out of bounds for '" << t.name << "' (file " << file.size() << " bytes)";
                return;
            }
            t.raw.assign(file.begin() + off, file.begin() + off + len);
        }

        // Materialize a TensorProto into a float32 HostBuffer (raw_data or typed data).
        static void fillHostFloat(const TensorProto &t, HostBuffer &hb, int64_t elems) {
            hb.resizeElems(elems, DType::Float32);
            float *dst = hb.f32();
            if (!t.raw.empty())
            {
                if (t.dataType == 1)
                { // FLOAT raw
                    std::memcpy(dst, t.raw.data(), std::min<size_t>(t.raw.size(), (size_t) elems * 4));
                } else if (t.dataType == 10)
                { // FLOAT16 raw -> decode to fp32 (2 bytes/elem)
                    const uint16_t *s     = reinterpret_cast<const uint16_t *>(t.raw.data());
                    int64_t         avail = (int64_t) (t.raw.size() / 2);
                    for (int64_t i = 0; i < elems && i < avail; ++i)
                    {
                        dst[i] = halfToFloat(s[i]);
                    }
                } else if (t.dataType == 7)
                { // INT64 raw
                    const int64_t *s     = reinterpret_cast<const int64_t *>(t.raw.data());
                    int64_t        avail = (int64_t) (t.raw.size() / 8);
                    for (int64_t i = 0; i < elems && i < avail; ++i)
                    {
                        dst[i] = (float) s[i];
                    }
                }
            } else if (!t.floatData.empty())
            {
                for (int64_t i = 0; i < elems && i < (int64_t) t.floatData.size(); ++i)
                {
                    dst[i] = t.floatData[i];
                }
            } else if (!t.int64Data.empty())
            {
                for (int64_t i = 0; i < elems && i < (int64_t) t.int64Data.size(); ++i)
                {
                    dst[i] = (float) t.int64Data[i];
                }
            }
        }

        // Materialize as int64 (for shape tensors).
        static void fillHostI64(const TensorProto &t, HostBuffer &hb, int64_t elems) {
            hb.resizeElems(elems, DType::Int64);
            int64_t *dst = hb.i64();
            if (!t.raw.empty() && t.dataType == 7)
            {
                std::memcpy(dst, t.raw.data(), std::min<size_t>(t.raw.size(), (size_t) elems * 8));
            } else if (!t.int64Data.empty())
            {
                for (int64_t i = 0; i < elems && i < (int64_t) t.int64Data.size(); ++i)
                {
                    dst[i] = t.int64Data[i];
                }
            }
        }

        // ----------------------------- AttributeProto -----------------------------
        // fields: 1=name, 2=f(float32), 3=i(int64), 4=s(bytes), 7=floats(packed),
        // 8=ints(packed), 5=t(TensorProto), 20=type(int32)
        static void parseAttr(Reader r, Node &node) {
            std::string name;
            Attr        a;
            uint32_t    f, w;
            TensorProto tp;
            bool        hasTp = false;
            while (r.tag(f, w))
            {
                switch (f)
                {
                    case 1:
                        name = r.str();
                        break;
                    case 2: {
                        uint32_t b = r.fixed32();
                        std::memcpy(&a.f, &b, 4);
                        a.kind = Attr::Float;
                        break;
                    }
                    case 3:
                        a.i    = (int64_t) r.varint();
                        a.kind = Attr::Int;
                        break;
                    case 4: {
                        auto b = r.bytes();
                        a.str.assign((const char *) b.data(), b.size());
                        a.kind = Attr::String;
                        break;
                    }
                    case 7: {
                        Reader s = r.sub();
                        while (!s.eof())
                        {
                            uint32_t b = s.fixed32();
                            float    fv;
                            std::memcpy(&fv, &b, 4);
                            a.floats.push_back(fv);
                        }
                        a.kind = Attr::Floats;
                        break;
                    }
                    case 8: {
                        if (w == 2)
                        {
                            Reader s = r.sub();
                            while (!s.eof())
                            {
                                a.ints.push_back((int64_t) s.varint());
                            }
                        } else
                        {
                            a.ints.push_back((int64_t) r.varint());
                        }
                        a.kind = Attr::Ints;
                        break;
                    }
                    case 5: {
                        tp    = parseTensor(r.sub());
                        hasTp = true;
                        break;
                    }
                    default:
                        r.skip(w);
                        break;
                }
            }
            if (hasTp)
            {
                // store as a float-ints attribute when it's a small shape/scalar constant
                a.kind    = Attr::Floats;
                a.shape   = tp.dims; // keep dims so a Constant node emits its true shape (e.g. anchor grids)
                int64_t n = 1;
                for (auto d: tp.dims)
                {
                    n *= d;
                }
                if (tp.dims.empty())
                {
                    n = std::max<int64_t>(1, (int64_t) std::max(tp.floatData.size(), tp.int64Data.size()));
                }
                if (!tp.int64Data.empty() || tp.dataType == 7)
                {
                    a.kind = Attr::Ints;
                    if (!tp.int64Data.empty())
                    {
                        a.ints = tp.int64Data;
                    } else if (!tp.raw.empty())
                    {
                        const int64_t *s     = (const int64_t *) tp.raw.data();
                        int64_t        avail = (int64_t) (tp.raw.size() / 8);
                        for (int64_t i = 0; i < n && i < avail; ++i)
                        {
                            a.ints.push_back(s[i]);
                        }
                    }
                } else
                {
                    if (!tp.floatData.empty())
                    {
                        a.floats = tp.floatData;
                    } else if (!tp.raw.empty())
                    {
                        // Decode by the tensor's dtype: FLOAT16 raw is 2 bytes/elem (reading it as fp32 would
                        // over-read 2x and fault); FLOAT raw is 4 bytes/elem. Bounds-checked either way.
                        if (tp.dataType == 10)
                        {
                            const uint16_t *s     = (const uint16_t *) tp.raw.data();
                            int64_t         avail = (int64_t) (tp.raw.size() / 2);
                            for (int64_t i = 0; i < n && i < avail; ++i)
                            {
                                a.floats.push_back(halfToFloat(s[i]));
                            }
                        } else
                        {
                            const float *s     = (const float *) tp.raw.data();
                            int64_t      avail = (int64_t) (tp.raw.size() / 4);
                            for (int64_t i = 0; i < n && i < avail; ++i)
                            {
                                a.floats.push_back(s[i]);
                            }
                        }
                    }
                }
            }
            node.attr.map[name] = a;
        }

        // ONNX TensorProto.DataType -> vknn DType. Unmapped/unknown types (complex, string, bf16) fall back
        // to Float32; the compute path is fp32/fp16, so integer I/O is carried as its own dtype and converted
        // at the graph boundary.
        static DType dtypeFromElem(int32_t el) {
            switch (el)
            {
                case 10:
                    return DType::Float16; // FLOAT16
                case 7:
                    return DType::Int64; // INT64
                case 6:
                    return DType::Int32; // INT32
                case 3:
                    return DType::Int8; // INT8
                case 2:
                case 9:
                    return DType::UInt8; // UINT8 / BOOL (0/1)
                case 1:
                default:
                    return DType::Float32; // FLOAT (and anything else)
            }
        }

        // ----------------------------- ValueInfoProto -----------------------------
        // field 1 = name, field 2 = type(TypeProto); TypeProto field1=tensor_type;
        // Tensor field1=elem_type(int32), field2=shape(TensorShapeProto);
        // TensorShapeProto field1=dim(repeated Dimension); Dimension field1=dim_value(int64).
        static void parseValueInfo(Reader r, std::string &name, Shape &shape, int32_t &elem) {
            uint32_t f, w;
            while (r.tag(f, w))
            {
                if (f == 1)
                {
                    name = r.str();
                } else if (f == 2)
                { // TypeProto
                    Reader   tp = r.sub();
                    uint32_t f2, w2;
                    while (tp.tag(f2, w2))
                    {
                        if (f2 == 1)
                        { // tensor_type
                            Reader   tt = tp.sub();
                            uint32_t f3, w3;
                            while (tt.tag(f3, w3))
                            {
                                if (f3 == 1)
                                {
                                    elem = (int32_t) tt.varint();
                                } else if (f3 == 2)
                                { // shape
                                    Reader   sh = tt.sub();
                                    uint32_t f4, w4;
                                    while (sh.tag(f4, w4))
                                    {
                                        if (f4 == 1)
                                        { // dim
                                            Reader   dim = sh.sub();
                                            uint32_t f5, w5;
                                            int64_t  val = -1;
                                            while (dim.tag(f5, w5))
                                            {
                                                if (f5 == 1)
                                                {
                                                    val = (int64_t) dim.varint(); // dim_value
                                                } else if (f5 == 2)
                                                {
                                                    dim.str();
                                                    val = -1;
                                                } // dim_param (dynamic)
                                                else
                                                {
                                                    dim.skip(w5);
                                                }
                                            }
                                            shape.push_back(val);
                                        } else
                                        {
                                            sh.skip(w4);
                                        }
                                    }
                                } else
                                {
                                    tt.skip(w3);
                                }
                            }
                        } else
                        {
                            tp.skip(w2);
                        }
                    }
                } else
                {
                    r.skip(w);
                }
            }
        }

        // ----------------------------- NodeProto -----------------------------
        // fields: 1=input(string repeated), 2=output(string repeated), 3=name, 4=op_type, 5=attribute,
        // 7=domain
        // Returns the raw input/output tensor NAMES (in `ins`/`outs`) instead of resolving them to ids
        // here; parseGraph resolves them in an SSA pass so a trace that REUSES a tensor name (two nodes
        // both writing "Cast_output_0" — common in un-deduped PyTorch exports) does not collapse onto one
        // TensorId. See the SSA-renaming block in parseGraph.
        static void parseNode(Reader r, Node &node, std::vector<std::string> &ins, std::vector<std::string> &outs) {
            uint32_t    f, w;
            std::string opType;
            while (r.tag(f, w))
            {
                switch (f)
                {
                    case 1:
                        ins.push_back(r.str());
                        break;
                    case 2:
                        outs.push_back(r.str());
                        break;
                    case 3:
                        node.name = r.str();
                        break;
                    case 4:
                        opType = r.str();
                        break;
                    case 5:
                        parseAttr(r.sub(), node);
                        break;
                    default:
                        r.skip(w);
                        break;
                }
            }
            node.type = opTypeFromOnnx(opType);
            if (node.type == OpType::Unary)
            {
                UnaryType u = unaryFromOnnx(opType);
                node.subOp  = (int32_t) u;
                // params (defaults per ONNX): LeakyRelu alpha=0.01, Elu alpha=1.0, HardSigmoid alpha,beta
                if (u == UnaryType::LeakyRelu)
                {
                    node.actLo = node.attr.getf("alpha", 0.01f);
                } else if (u == UnaryType::Elu)
                {
                    node.actLo = node.attr.getf("alpha", 1.0f);
                } else if (u == UnaryType::HardSigmoid)
                {
                    node.actLo = node.attr.getf("alpha", 0.2f);
                    node.actHi = node.attr.getf("beta", 0.5f);
                }
            } else if (node.type == OpType::Binary)
            {
                node.subOp = (int32_t) binaryFromOnnx(opType);
            } else if (node.type == OpType::Reduce)
            { node.subOp = (int32_t) reduceFromOnnx(opType); }
            if (node.type == OpType::Unknown)
            {
                VKNN_WARN << "unknown ONNX op '" << opType << "' (node " << node.name << ")";
            }
        }

        // ----------------------------- GraphProto -----------------------------
        // fields: 1=node, 2=name, 5=initializer, 11=input, 12=output, 13=value_info
        static void parseGraph(Reader r, Graph &g, const std::string &baseDir) {
            uint32_t f, w;
            struct PendingInit {
                std::string name;
                TensorProto tp;
            };
            std::vector<PendingInit>                    inits;
            std::map<std::string, std::vector<uint8_t>> extCache; // external .data files, read once each
            std::vector<Node>                           nodes;
            std::vector<std::vector<std::string>>       nodeIns, nodeOuts; // raw names, resolved in the SSA pass
            while (r.tag(f, w))
            {
                switch (f)
                {
                    case 1: {
                        Node                     n;
                        std::vector<std::string> ni, no;
                        parseNode(r.sub(), n, ni, no);
                        nodes.push_back(std::move(n));
                        nodeIns.push_back(std::move(ni));
                        nodeOuts.push_back(std::move(no));
                        break;
                    }
                    case 5: {
                        TensorProto tp = parseTensor(r.sub());
                        std::string nm = tp.name;
                        inits.push_back({nm, std::move(tp)});
                        break;
                    }
                    case 11: {
                        std::string nm;
                        Shape       sh;
                        int32_t     el = 1;
                        parseValueInfo(r.sub(), nm, sh, el);
                        TensorId id        = g.findOrAdd(nm);
                        g.desc(id).shape   = sh;
                        g.desc(id).dtype   = dtypeFromElem(el);
                        g.desc(id).isInput = true;
                        g.inputs.push_back(id);
                        break;
                    }
                    case 12: {
                        std::string nm;
                        Shape       sh;
                        int32_t     el = 1;
                        parseValueInfo(r.sub(), nm, sh, el);
                        TensorId id         = g.findOrAdd(nm);
                        g.desc(id).shape    = sh;
                        g.desc(id).dtype    = dtypeFromElem(el);
                        g.desc(id).isOutput = true;
                        g.outputs.push_back(id);
                        break;
                    }
                    case 13: {
                        std::string nm;
                        Shape       sh;
                        int32_t     el = 1;
                        parseValueInfo(r.sub(), nm, sh, el);
                        if (!nm.empty())
                        {
                            TensorId id = g.findOrAdd(nm);
                            if (g.desc(id).shape.empty())
                            {
                                g.desc(id).shape = sh;
                            }
                        }
                        break;
                    }
                    default:
                        r.skip(w);
                        break;
                }
            }

            // Materialize initializers.
            for (auto &pi: inits)
            {
                TensorId id     = g.findOrAdd(pi.name);
                auto    &d      = g.desc(id);
                d.isInitializer = true;
                d.shape         = pi.tp.dims;
                int64_t n       = 1;
                for (auto x: pi.tp.dims)
                {
                    n *= x;
                }
                if (pi.tp.dims.empty())
                {
                    n = 1;
                }
                resolveExternal(baseDir, pi.tp, extCache); // pull EXTERNAL weights from the sibling data file
                HostBuffer hb;
                if (pi.tp.dataType == 7)
                {
                    d.dtype = DType::Int64;
                    fillHostI64(pi.tp, hb, n);
                } else
                {
                    d.dtype = DType::Float32;
                    fillHostFloat(pi.tp, hb, n);
                }
                g.initializers[id] = std::move(hb);
            }

            // --- SSA-rename node I/O ---
            // ONNX requires unique tensor names, but un-deduped PyTorch traces reuse them (e.g. two Cast
            // nodes both output "Cast_output_0"). Binding by name (findOrAdd) would alias distinct tensors
            // onto ONE TensorId with ONE shape -> wrong static buffer sizes / shape mismatches on the GPU
            // path. Instead: bind each node input to the nearest PRECEDING producer of that name, and give
            // each node output a FRESH TensorId. Declared graph outputs re-point to their final producer.
            {
                std::unordered_map<std::string, TensorId> latest;
                for (TensorId id: g.inputs)
                {
                    latest[g.desc(id).name] = id;
                }
                for (auto &pi: inits)
                {
                    latest[pi.name] = g.find(pi.name);
                }
                for (size_t i = 0; i < nodes.size(); ++i)
                {
                    for (const std::string &s: nodeIns[i])
                    {
                        if (s.empty())
                        {
                            nodes[i].inputs.push_back(kNoTensor);
                            continue;
                        }
                        auto it = latest.find(s);
                        nodes[i].inputs.push_back(it != latest.end() ? it->second : g.findOrAdd(s));
                    }
                    for (const std::string &s: nodeOuts[i])
                    {
                        if (s.empty())
                        {
                            nodes[i].outputs.push_back(kNoTensor);
                            continue;
                        }
                        TensorDesc d;
                        d.name      = s;
                        TensorId id = g.addTensor(std::move(d)); // fresh id; tensorByName[s] -> id (last def wins)
                        latest[s]   = id;
                        nodes[i].outputs.push_back(id);
                    }
                    if (nodes[i].name.empty())
                    {
                        nodes[i].name = std::string(opTypeName(nodes[i].type)) + "_" + std::to_string(i);
                    }
                }
                // A declared output name may have been produced by several nodes; point g.outputs at the
                // final producer and carry the declared shape/dtype onto it.
                for (TensorId &oid: g.outputs)
                {
                    auto it = latest.find(g.desc(oid).name);
                    if (it != latest.end() && it->second != oid)
                    {
                        Shape declShape             = g.desc(oid).shape;
                        DType declDtype             = g.desc(oid).dtype;
                        g.desc(it->second).isOutput = true;
                        g.desc(it->second).shape    = declShape;
                        g.desc(it->second).dtype    = declDtype;
                        oid                         = it->second;
                    }
                }
            }
            g.nodes = std::move(nodes);

            // Remove initializer ids from the graph input list (ONNX lists them in both).
            std::vector<TensorId> realInputs;
            for (TensorId id: g.inputs)
            {
                if (!g.isInitializer(id))
                {
                    realInputs.push_back(id);
                }
            }
            g.inputs = realInputs;
        }

    } // namespace onnx

    // Public entry point.
    Graph importOnnx(const std::string &path) {
        std::ifstream f(path, std::ios::binary);
        if (!f)
        {
            throw Error(Status::IoError, "cannot open ONNX file: " + path);
        }
        std::vector<uint8_t> buf((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        // Directory the model lives in — external_data locations are relative to it.
        size_t      slash   = path.find_last_of("/\\");
        std::string baseDir = slash == std::string::npos ? std::string() : path.substr(0, slash);
        Graph       g;
        // ModelProto: field 7 = graph (GraphProto). Skip everything else.
        onnx::Reader r(buf.data(), buf.size());
        uint32_t     fld, wire;
        bool         foundGraph = false;
        while (r.tag(fld, wire))
        {
            if (fld == 7 && wire == 2)
            {
                onnx::parseGraph(r.sub(), g, baseDir);
                foundGraph = true;
            } else
            {
                r.skip(wire);
            }
        }
        if (!foundGraph)
        {
            throw Error(Status::InvalidArgument, "no GraphProto in ONNX model");
        }
        g.topoSort();
        VKNN_INFO << "Imported ONNX: " << g.nodes.size() << " nodes, " << g.initializers.size() << " initializers, " << g.inputs.size() << " inputs, "
                  << g.outputs.size() << " outputs";
        return g;
    }

} // namespace vknn
