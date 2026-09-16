// See onnx_graph_builder.h. GraphBuilder method bodies: the collect / materialize / SSA-resolve /
// integer-width phases.
#include "onnx_graph_builder.h"
#include "core/bitwise_attrs.h"
#include <set>

namespace vknn { namespace onnx {

    void GraphBuilder::build(Reader r) {
        collect(r);
        materializeInitializers();
        ssaResolveNodeIO();
        stampIntegerWidths();
        g.nodes = std::move(nodes);
        dropInitializerInputs();
    }

    bool GraphBuilder::fullyStatic(const Shape &sh) {
        for (int64_t d: sh)
        {
            if (d < 0)
            {
                return false;
            }
        }
        return true;
    }

    void GraphBuilder::collect(Reader r) {
        uint32_t f, w;
        while (r.tag(f, w))
        {
            switch (f)
            {
                case kGraphNode: {
                    Node                     n;
                    std::vector<std::string> ni, no;
                    NodeWireInfo             wire;
                    NodeParser::parseNode(r.sub(), n, ni, no, baseDir, &extCache, &wire);
                    nodes.push_back(std::move(n));
                    nodeWire.push_back(std::move(wire));
                    nodeIns.push_back(std::move(ni));
                    nodeOuts.push_back(std::move(no));
                    break;
                }
                case kGraphInitializer: {
                    TensorProto tp = TensorProtoParser::parse(r.sub());
                    std::string nm = tp.name;
                    inits.push_back({nm, std::move(tp)});
                    break;
                }
                case kGraphInput: {
                    std::string              nm;
                    Shape                    sh;
                    int32_t                  el = (int32_t) OnnxType::Undefined; // an untyped input computes as fp32
                    std::vector<std::string> params;                             // per-axis dim_param symbols (empty = concrete)
                    NodeParser::parseValueInfo(r.sub(), nm, sh, el, &params);
                    TensorId id          = g.findOrAdd(nm);
                    g.desc(id).shape     = sh;
                    g.desc(id).dimParams = std::move(params); // retained so inferShapes can bind symbols
                    g.desc(id).dtype     = dtypeFromElem(el);
                    g.desc(id).isInput   = true;
                    g.inputs.push_back(id);
                    if (!isType(el, OnnxType::Undefined))
                    {
                        declaredElemTypes[id] = el;
                    }
                    break;
                }
                case kGraphOutput: {
                    std::string nm;
                    Shape       sh;
                    int32_t     el = (int32_t) OnnxType::Undefined; // an untyped output computes as fp32
                    NodeParser::parseValueInfo(r.sub(), nm, sh, el);
                    TensorId id = g.findOrAdd(nm);
                    if (fullyStatic(sh))
                    {
                        g.desc(id).shape = sh;
                    }
                    g.desc(id).dtype    = dtypeFromElem(el);
                    g.desc(id).isOutput = true;
                    g.outputs.push_back(id);
                    if (!isType(el, OnnxType::Undefined))
                    {
                        declaredElemTypes[id] = el;
                    }
                    break;
                }
                case kGraphValueInfo: {
                    std::string nm;
                    Shape       sh;
                    int32_t     el = (int32_t) OnnxType::Undefined;
                    NodeParser::parseValueInfo(r.sub(), nm, sh, el);
                    if (!nm.empty() && !sh.empty())
                    {
                        valueInfoShapes[nm] = sh; // applied to the matching node output in ssaResolveNodeIO
                    }
                    if (!nm.empty() && !isType(el, OnnxType::Undefined))
                    {
                        valueInfoElemTypes[nm] = el; // applied to the matching node output in ssaResolveNodeIO
                    }
                    break;
                }
                default:
                    r.skip(w);
                    break;
            }
        }
    }

    void GraphBuilder::materializeInitializers() {
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
            TensorProtoParser::resolveExternal(baseDir, pi.tp, extCache); // pull EXTERNAL weights from the sibling data file
            declaredElemTypes[id] = pi.tp.dataType;
            HostBuffer hb;
            if (isType(pi.tp.dataType, OnnxType::Int64) || isType(pi.tp.dataType, OnnxType::Uint32) || isType(pi.tp.dataType, OnnxType::Uint64))
            {
                // INT64 stays exact; UINT32 and UINT64 exceed fp32's exact integer range, so they take the
                // same int64 storage (a UINT64 value at or above 2^63 keeps its two's-complement bits).
                d.dtype = DType::Int64;
                TensorProtoParser::fillHostI64(pi.tp, hb, n);
            } else if (isType(pi.tp.dataType, OnnxType::Int8) || isType(pi.tp.dataType, OnnxType::Uint8))
            {
                // A pre-quantized int8/uint8 initializer (a QDQ model's weights, a MatMulNBits packed
                // int4 payload) stays in NATIVE 1-byte host storage instead of widening to fp32 -- an
                // 8B int4 model's ~4.3 GB of packed weights would otherwise materialize as ~17 GB and
                // exhaust host RAM at import. The descriptor still records INT8/UINT8 so the dequantize
                // pass recovers the quantize dtype's saturation range ([-128,127] / [0,255]) from a
                // zero_point tensor; initFloats decodes the lanes to fp32 on demand, so every reader
                // still sees integer-valued fp32. (BOOL, which also decodes to UInt8 storage, is
                // normalized to 0/1 fp32 through fillHostFloat below and is never a quant parameter.)
                d.dtype = isType(pi.tp.dataType, OnnxType::Int8) ? DType::Int8 : DType::UInt8;
                TensorProtoParser::fillHostBytes(pi.tp, hb, n, d.dtype);
            } else
            {
                // FLOAT / FLOAT16 / DOUBLE (and INT32 / INT16 / UINT16 / BOOL) materialize to fp32 host
                // storage; a plain fp32 read through initFloats recovers the value.
                d.dtype = DType::Float32;
                TensorProtoParser::fillHostFloat(pi.tp, hb, n);
            }
            g.initializers[id] = std::move(hb);
        }
    }

    void GraphBuilder::ssaResolveNodeIO() {
        std::unordered_map<std::string, int> producerCount;
        for (const auto &outs: nodeOuts)
        {
            for (const std::string &s: outs)
            {
                if (!s.empty())
                {
                    ++producerCount[s];
                }
            }
        }
        size_t reused = 0;
        for (const auto &kv: producerCount)
        {
            if (kv.second > 1)
            {
                ++reused;
            }
        }
        if (reused > 0)
        {
            VKNN_WARN << "ONNX graph is not SSA: " << reused << " tensor name(s) have multiple producers " << "(un-deduped trace export). Inputs bind to the nearest preceding producer; " << "value_info shape hints for reused names are ignored.";
        }
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
                d.name   = s;
                auto vit = valueInfoShapes.find(s);
                if (vit != valueInfoShapes.end() && fullyStatic(vit->second) && producerCount[s] == 1)
                {
                    d.shape = vit->second; // carry the value_info shape hint onto this node output
                }
                TensorId id = g.addTensor(std::move(d)); // fresh id; tensorByName[s] -> id (last def wins)
                latest[s]   = id;
                nodes[i].outputs.push_back(id);
                auto eit = valueInfoElemTypes.find(s);
                if (eit != valueInfoElemTypes.end() && producerCount[s] == 1)
                {
                    declaredElemTypes[id] = eit->second; // attributable only to a single producer, like the shape hint
                }
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
                if (fullyStatic(declShape))
                {
                    g.desc(it->second).shape = declShape;
                }
                g.desc(it->second).dtype = declDtype;
                auto declaredElem        = declaredElemTypes.find(oid);
                if (declaredElem != declaredElemTypes.end())
                {
                    declaredElemTypes[it->second] = declaredElem->second;
                }
                oid = it->second;
            }
        }
    }

    namespace {

        // Integer width and signedness of an ONNX element type; false for a non-integer type.
        bool integerWidthOf(int32_t elemType, bitwise::IntegerWidth &width) {
            switch ((OnnxType) elemType)
            {
                case OnnxType::Uint8:
                case OnnxType::Bool:
                    width = {bitwise::kInt8Bits, false};
                    return true;
                case OnnxType::Int8:
                    width = {bitwise::kInt8Bits, true};
                    return true;
                case OnnxType::Uint16:
                    width = {bitwise::kInt16Bits, false};
                    return true;
                case OnnxType::Int16:
                    width = {bitwise::kInt16Bits, true};
                    return true;
                case OnnxType::Uint32:
                    width = {bitwise::kInt32Bits, false};
                    return true;
                case OnnxType::Int32:
                    width = {bitwise::kInt32Bits, true};
                    return true;
                case OnnxType::Uint64:
                    width = {bitwise::kInt64Bits, false};
                    return true;
                case OnnxType::Int64:
                    width = {bitwise::kInt64Bits, true};
                    return true;
                default:
                    return false;
            }
        }

        // ONNX ops whose output 0 has a fixed element type whatever their inputs are.
        const std::set<std::string> &int64ResultOps() {
            static const std::set<std::string> ops = {"Shape", "Size", "ArgMax", "ArgMin", "NonZero", "NonMaxSuppression"};
            return ops;
        }
        const std::set<std::string> &int32ResultOps() {
            static const std::set<std::string> ops = {"MatMulInteger", "ConvInteger"};
            return ops;
        }
        const std::set<std::string> &boolResultOps() {
            static const std::set<std::string> ops = {"Equal", "Greater", "GreaterOrEqual", "Less", "LessOrEqual", "And", "Or", "Xor", "Not", "IsNaN", "IsInf"};
            return ops;
        }

        // ONNX ops whose result takes the element type of an input other than input 0 (Where's condition
        // and a DequantizeLinear's quantized data are not the result type), keyed to that input's slot.
        const std::map<std::string, size_t> &typeCarryingInput() {
            static const std::map<std::string, size_t> slots = {{"Where", 1}, {"CastLike", 1}, {"DequantizeLinear", 1}, {"OneHot", 2}};
            return slots;
        }

        constexpr size_t kTopKIndicesOutput        = 1; // TopK's output slot holding the INT64 indices
        constexpr size_t kQuantizeZeroPointInput   = 2; // QuantizeLinear's zero point, which carries the result type
        constexpr size_t kDefaultTypeCarryingInput = 0; // every other op's result takes input 0's type

        Attr intAttr(int64_t value) {
            Attr attribute;
            attribute.kind = Attr::Int;
            attribute.i    = value;
            return attribute;
        }

    } // namespace

    int32_t GraphBuilder::producerElemType(const ProducerSlot &producer, TensorId &typeSource) const {
        const int32_t       undefined = (int32_t) OnnxType::Undefined;
        const Node         &node      = nodes[producer.node];
        const NodeWireInfo &wire      = nodeWire[producer.node];
        const std::string  &opType    = wire.opType;
        typeSource                    = kNoTensor;
        if (opType == "Cast")
        {
            return node.attr.has("to") ? (int32_t) node.attr.geti("to") : undefined;
        }
        if (int64ResultOps().count(opType) || (opType == "TopK" && producer.output == kTopKIndicesOutput))
        {
            return (int32_t) OnnxType::Int64;
        }
        if (int32ResultOps().count(opType))
        {
            return (int32_t) OnnxType::Int32;
        }
        if (boolResultOps().count(opType))
        {
            return (int32_t) OnnxType::Bool;
        }
        if (opType == "Constant")
        {
            // `value` is a tensor with its own data_type; value_int / value_ints are INT64.
            if (!isType(wire.tensorAttrElemType, OnnxType::Undefined))
            {
                return wire.tensorAttrElemType;
            }
            return (node.attr.has("value_int") || node.attr.has("value_ints")) ? (int32_t) OnnxType::Int64 : undefined;
        }
        if (opType == "ConstantOfShape")
        {
            // The fill tensor's data_type; the ONNX default fill is a FLOAT zero.
            return isType(wire.tensorAttrElemType, OnnxType::Undefined) ? (int32_t) OnnxType::Float : wire.tensorAttrElemType;
        }
        if (opType == "EyeLike" && node.attr.has("dtype"))
        {
            return (int32_t) node.attr.geti("dtype");
        }
        size_t typeInput = kDefaultTypeCarryingInput;
        if (opType == "QuantizeLinear")
        {
            // The zero point's type, or UINT8 when the zero point is absent.
            if (node.inputs.size() <= kQuantizeZeroPointInput || node.inputs[kQuantizeZeroPointInput] == kNoTensor)
            {
                return (int32_t) OnnxType::Uint8;
            }
            typeInput = kQuantizeZeroPointInput;
        } else if (auto carrying = typeCarryingInput().find(opType); carrying != typeCarryingInput().end())
        { typeInput = carrying->second; }
        if (typeInput < node.inputs.size())
        {
            typeSource = node.inputs[typeInput];
        }
        return undefined;
    }

    int32_t GraphBuilder::resolveElemType(TensorId t, const std::unordered_map<TensorId, ProducerSlot> &producers, std::unordered_map<TensorId, int32_t> &resolved) const {
        int32_t               type = (int32_t) OnnxType::Undefined;
        std::vector<TensorId> path; // tensors whose type is the one this walk ends at
        for (int hop = 0; hop < kElemTypeResolveMaxHops && t != kNoTensor; ++hop)
        {
            auto known = resolved.find(t);
            if (known != resolved.end())
            {
                type = known->second;
                break;
            }
            path.push_back(t);
            auto declared = declaredElemTypes.find(t);
            if (declared != declaredElemTypes.end())
            {
                type = declared->second;
                break;
            }
            auto producer = producers.find(t);
            if (producer == producers.end())
            {
                break;
            }
            type = producerElemType(producer->second, t);
            if (!isType(type, OnnxType::Undefined))
            {
                break;
            }
        }
        for (TensorId visited: path)
        {
            resolved[visited] = type;
        }
        return type;
    }

    void GraphBuilder::stampIntegerWidths() {
        std::unordered_map<TensorId, ProducerSlot> producers;
        for (size_t i = 0; i < nodes.size(); ++i)
        {
            for (size_t slot = 0; slot < nodes[i].outputs.size(); ++slot)
            {
                if (nodes[i].outputs[slot] != kNoTensor)
                {
                    producers[nodes[i].outputs[slot]] = {i, slot};
                }
            }
        }
        std::unordered_map<TensorId, int32_t> resolved;
        for (Node &node: nodes)
        {
            if (node.type != OpType::BitShift && node.type != OpType::BitwiseNot)
            {
                continue;
            }
            // BitShift's two operands share one type, so the shift count resolves the width when the
            // shifted operand does not.
            const size_t          operandsToTry = node.type == OpType::BitShift ? 2 : 1;
            bitwise::IntegerWidth width;
            bool                  isResolved = false;
            for (size_t operand = 0; operand < operandsToTry && operand < node.inputs.size() && !isResolved; ++operand)
            {
                isResolved = integerWidthOf(resolveElemType(node.inputs[operand], producers, resolved), width);
            }
            if (!isResolved)
            {
                VKNN_WARN << opTypeName(node.type) << " '" << node.name << "': the operand integer type is neither declared nor derivable; " << bitwise::kIntBitsAttr << " / " << bitwise::kIntSignedAttr << " default to " << bitwise::kDefaultIntBits << " / " << bitwise::kDefaultIntSigned;
                continue;
            }
            node.attr.map[bitwise::kIntBitsAttr]   = intAttr(width.bits);
            node.attr.map[bitwise::kIntSignedAttr] = intAttr(width.isSigned ? 1 : 0);
        }
    }

    void GraphBuilder::dropInitializerInputs() {
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

}} // namespace vknn::onnx
