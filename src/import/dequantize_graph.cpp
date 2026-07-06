#include "passes_internal.h"

namespace vknn {

    namespace {

        // Initializer payload widened to double. Integer payloads (int8/uint8/int32/bool) arrive
        // from the importer already widened into fp32 host storage (fillHostFloat), which initFloats
        // decodes correctly for any non-fp16 dtype; a Float16 payload is the fp16 .vxm form. The
        // stored dtype is thus a label, not a storage width -- Int8/UInt8 flag a quantized tensor
        // (see quantRange) while still reading through the fp32 path.
        // @returns False only for Int64 payloads (8-byte storage initFloats cannot widen).
        bool initAsDouble(const Graph &g, TensorId id, std::vector<double> &out) {
            if (g.desc(id).dtype == DType::Int64)
            {
                return false;
            }
            std::vector<float> f = initFloats(g, id);
            out.assign(f.begin(), f.end());
            return true;
        }

        // True when tensors `a` and `b` hold the same quantization parameter: the same id, or two
        // initializers with elementwise-identical payloads.
        bool sameParam(const Graph &g, TensorId a, TensorId b) {
            if (a == b)
            {
                return true;
            }
            if (a == kNoTensor || b == kNoTensor || !g.isInitializer(a) || !g.isInitializer(b))
            {
                return false;
            }
            std::vector<double> av, bv;
            if (!initAsDouble(g, a, av) || !initAsDouble(g, b, bv) || av.size() != bv.size())
            {
                return false;
            }
            for (size_t i = 0; i < av.size(); ++i)
            {
                if (av[i] != bv[i])
                {
                    return false;
                }
            }
            return true;
        }

        // Zero-point equality with the ONNX default: an absent zero_point input means 0, so absent
        // matches absent and an all-zero initializer payload.
        bool zeroPointsMatch(const Graph &g, TensorId a, TensorId b) {
            if (a == b)
            {
                return true;
            }
            if (a == kNoTensor || b == kNoTensor)
            {
                TensorId            present = a == kNoTensor ? b : a;
                std::vector<double> v;
                if (!g.isInitializer(present) || !initAsDouble(g, present, v))
                {
                    return false;
                }
                for (double x: v)
                {
                    if (x != 0.0)
                    {
                        return false;
                    }
                }
                return true;
            }
            return sameParam(g, a, b);
        }

        // True when any surviving node input, fused edge, or graph output reads tensor `t`.
        // `removed` holds the indices of nodes already marked for deletion in this run.
        bool tensorConsumed(const Graph &g, const std::set<size_t> &removed, TensorId t) {
            for (size_t i = 0; i < g.nodes.size(); ++i)
            {
                if (removed.count(i))
                {
                    continue;
                }
                const Node &n = g.nodes[i];
                for (TensorId x: n.inputs)
                {
                    if (x == t)
                    {
                        return true;
                    }
                }
                if (n.fusedResidual == t || n.fusedBias == t)
                {
                    return true;
                }
            }
            for (TensorId go: g.outputs)
            {
                if (go == t)
                {
                    return true;
                }
            }
            return false;
        }

        // Drops the nodes whose indices are in `remove`, preserving the surviving order.
        void compact(Graph &g, const std::set<size_t> &remove) {
            if (remove.empty())
            {
                return;
            }
            std::vector<Node> kept;
            kept.reserve(g.nodes.size() - remove.size());
            for (size_t i = 0; i < g.nodes.size(); ++i)
            {
                if (!remove.count(i))
                {
                    kept.push_back(std::move(g.nodes[i]));
                }
            }
            g.nodes = std::move(kept);
        }

        // Folds every DequantizeLinear whose inputs are all initializers into an fp32 initializer
        // holding (x - zero_point) * scale, computed per element in double and stored as float.
        // Per-tensor (single-element scale) and per-axis (1-D scale of length dims[axis], axis
        // attribute, negative values counted from the back) forms both fold; the zero_point
        // broadcasts as a scalar when the scale is per-axis. A node that cannot fold (blocked
        // quantization, shape/size mismatch, undecodable payload) stays in place with a WARN.
        // @returns The number of nodes folded.
        int foldWeightDequant(Graph &g) {
            std::set<size_t> remove;
            for (size_t i = 0; i < g.nodes.size(); ++i)
            {
                const Node &n = g.nodes[i];
                if (n.type != OpType::DequantizeLinear || n.inputs.size() < 2 || n.outputs.empty())
                {
                    continue;
                }
                TensorId x = n.inputs[0], s = n.inputs[1];
                TensorId z = n.inputs.size() > 2 ? n.inputs[2] : kNoTensor;
                TensorId y = n.outputs[0];
                if (x == kNoTensor || s == kNoTensor || y == kNoTensor)
                {
                    continue;
                }
                if (!g.isInitializer(x) || !g.isInitializer(s) || (z != kNoTensor && !g.isInitializer(z)))
                {
                    continue; // activation dequant: a later lowering stage owns it
                }
                if (n.attr.geti("block_size", 0) != 0)
                {
                    VKNN_WARN << "DequantizeLinear " << n.name << " kept: blocked quantization (block_size) is not folded";
                    continue;
                }
                std::vector<double> xv, sv, zv;
                if (!initAsDouble(g, x, xv) || !initAsDouble(g, s, sv) || (z != kNoTensor && !initAsDouble(g, z, zv)))
                {
                    VKNN_WARN << "DequantizeLinear " << n.name << " kept: undecodable operand payload dtype";
                    continue;
                }
                const Shape &shape = g.desc(x).shape;
                int64_t      inner = 1; // elements between consecutive channel steps (per-axis form)
                if (sv.empty())
                {
                    VKNN_WARN << "DequantizeLinear " << n.name << " kept: empty scale";
                    continue;
                }
                if (sv.size() > 1)
                {
                    int64_t rank = (int64_t) shape.size();
                    int64_t axis = n.attr.geti("axis", 1);
                    if (axis < 0)
                    {
                        axis += rank;
                    }
                    if (axis < 0 || axis >= rank || shape[axis] != (int64_t) sv.size())
                    {
                        VKNN_WARN << "DequantizeLinear " << n.name << " kept: per-axis scale of " << sv.size() << " elements does not match axis " << n.attr.geti("axis", 1) << " of " << shapeStr(shape);
                        continue;
                    }
                    for (int64_t d = axis + 1; d < rank; ++d)
                    {
                        inner *= shape[d];
                    }
                }
                if (!zv.empty() && zv.size() != sv.size() && zv.size() != 1)
                {
                    VKNN_WARN << "DequantizeLinear " << n.name << " kept: zero_point count " << zv.size() << " matches neither the scale count nor a scalar";
                    continue;
                }
                HostBuffer hb;
                hb.resizeElems((int64_t) xv.size(), DType::Float32);
                float *out = hb.f32();
                for (size_t e = 0; e < xv.size(); ++e)
                {
                    size_t c   = sv.size() == 1 ? 0 : (size_t) ((int64_t) e / inner) % sv.size();
                    double zp  = zv.empty() ? 0.0 : zv[zv.size() == 1 ? 0 : c];
                    out[e]     = (float) ((xv[e] - zp) * sv[c]);
                }
                TensorDesc &yd  = g.desc(y);
                yd.isInitializer = true;
                yd.dtype         = DType::Float32;
                yd.shape         = shape;
                g.initializers[y] = std::move(hb);
                remove.insert(i); // the orphaned x/scale/zp payloads fall to pruneDeadInitializers
            }
            compact(g, remove);
            return (int) remove.size();
        }

        // Integer range a QuantizeLinear saturates into, taken from the zero_point tensor's stored
        // dtype (Int8 -> [-128,127], UInt8 -> [0,255]). ONNX defaults an absent zero_point to uint8,
        // so kNoTensor takes the uint8 range. Any other dtype (a scale/zp built as plain fp32, an
        // int32-biased DequantizeLinear) is not an 8-bit quantize grid and yields no range.
        // @returns True when `zp` names an 8-bit quantize type; then qmin/qmax hold its range.
        bool quantRange(const Graph &g, TensorId zp, double &qmin, double &qmax) {
            DType dt = zp == kNoTensor ? DType::UInt8 : g.desc(zp).dtype;
            if (dt == DType::Int8)
            {
                qmin = -128.0;
                qmax = 127.0;
                return true;
            }
            if (dt == DType::UInt8)
            {
                qmin = 0.0;
                qmax = 255.0;
                return true;
            }
            return false;
        }

        // Registers a scalar Float32 initializer holding `v` and returns its id. Used for the Clip
        // bounds inserted in place of a collapsed quantize round-trip.
        TensorId addScalarFloatInit(Graph &g, const std::string &name, float v) {
            TensorDesc d;
            d.name          = name;
            d.shape         = {}; // rank-0 scalar; initFloats recovers the single element from the payload size
            d.isInitializer = true;
            TensorId   id   = g.addTensor(std::move(d));
            HostBuffer hb;
            hb.resizeElems(1, DType::Float32);
            hb.f32()[0]     = v;
            g.initializers[id] = std::move(hb);
            return id;
        }

        // Builds the saturation clamp a collapsed quantize hop leaves behind: a Clip(x, lo, hi) with
        // constant per-tensor bounds lo=(qmin-zp)*scale, hi=(qmax-zp)*scale, where qmin/qmax come from
        // the quantize dtype (quantRange). Dropping the round-trip drops only the 8-bit rounding; the
        // saturation is real and must survive, because ORT's QDQ quantizer folds a preceding ReLU into
        // the range (zp set so lo becomes 0) -- collapsing to raw float without the clamp silently
        // deletes that ReLU. Clip with constant bounds is pointwise-fusable and GPU-eligible; when the
        // bounds do not bind it is a harmless identity, so it is inserted for every activation hop.
        // The new node is appended to `added` (not `g.nodes`, which the caller iterates).
        // @returns The Clip output tensor id, or kNoTensor when the hop cannot take a per-tensor clamp
        //          (per-axis scale, or a non-8-bit dtype); the caller then leaves the sandwich intact.
        TensorId insertQuantClamp(Graph &g, TensorId scale, TensorId zp, TensorId x, std::vector<Node> &added) {
            double qmin, qmax;
            if (!quantRange(g, zp, qmin, qmax))
            {
                return kNoTensor;
            }
            std::vector<double> sv, zv;
            if (!initAsDouble(g, scale, sv) || sv.empty())
            {
                return kNoTensor;
            }
            if (sv.size() > 1)
            {
                return kNoTensor; // per-axis activation quant: a scalar Clip cannot carry per-channel bounds
            }
            double zpv = 0.0;
            if (zp != kNoTensor && (!initAsDouble(g, zp, zv) || zv.empty()))
            {
                return kNoTensor;
            }
            if (!zv.empty())
            {
                zpv = zv[0];
            }
            float    lo = (float) ((qmin - zpv) * sv[0]);
            float    hi = (float) ((qmax - zpv) * sv[0]);
            // Name off the clamped tensor id (not just its name, which may be empty on an anonymous
            // intermediate) so clamps on distinct hops never collide.
            std::string base = g.desc(x).name + "_qclamp" + std::to_string((long long) x);
            TensorId loId = addScalarFloatInit(g, base + "_lo", lo);
            TensorId hiId = addScalarFloatInit(g, base + "_hi", hi);
            TensorDesc yd;
            yd.name       = base;
            TensorId y    = g.addTensor(std::move(yd));
            Node clip;
            clip.type    = OpType::Clip;
            clip.name    = base;
            clip.inputs  = {x, loId, hiId};
            clip.outputs = {y};
            added.push_back(std::move(clip));
            return y;
        }

        // Removes each QuantizeLinear->DequantizeLinear activation sandwich whose scale/zero_point
        // match (same ids or equal payloads; per-axis pairs must also agree on the axis attribute):
        // the DequantizeLinear's consumers rewire past both nodes to a Clip on the QuantizeLinear's
        // float input, and both nodes drop. Dropping the round-trip drops only the 8-bit rounding; the
        // inserted Clip preserves the quantize dtype's saturation range (see insertQuantClamp -- the
        // range is where ORT hides a fused ReLU). A hop that cannot take a per-tensor clamp (per-axis
        // scale, or a non-8-bit quantize dtype) is left intact for a later lowering stage. A
        // QuantizeLinear still consumed after the rewiring (another non-matching consumer, a graph
        // output) survives.
        // @returns The number of sandwiches collapsed.
        int collapseQdqSandwiches(Graph &g) {
            std::map<TensorId, size_t> producer;
            for (size_t i = 0; i < g.nodes.size(); ++i)
            {
                for (TensorId o: g.nodes[i].outputs)
                {
                    if (o != kNoTensor)
                    {
                        producer[o] = i;
                    }
                }
            }
            std::set<size_t> remove;
            int              collapsed = 0;
            std::vector<Node> added;               // Clip nodes to append after the iteration
            std::map<size_t, TensorId> clampOf;    // Q producer index -> its shared clamp output (fan-out reuse)
            for (size_t i = 0; i < g.nodes.size(); ++i)
            {
                const Node &dq = g.nodes[i];
                if (dq.type != OpType::DequantizeLinear || dq.inputs.size() < 2 || dq.outputs.empty() || dq.outputs[0] == kNoTensor)
                {
                    continue;
                }
                auto it = producer.find(dq.inputs[0]);
                if (it == producer.end() || remove.count(it->second))
                {
                    continue;
                }
                const Node &q = g.nodes[it->second];
                if (q.type != OpType::QuantizeLinear || q.inputs.size() < 2 || q.inputs[0] == kNoTensor)
                {
                    continue;
                }
                if (!sameParam(g, q.inputs[1], dq.inputs[1]))
                {
                    continue; // requant edge: a later lowering stage owns it
                }
                TensorId qz = q.inputs.size() > 2 ? q.inputs[2] : kNoTensor;
                TensorId dz = dq.inputs.size() > 2 ? dq.inputs[2] : kNoTensor;
                if (!zeroPointsMatch(g, qz, dz))
                {
                    continue;
                }
                // A per-axis pair must quantize along the same axis for the params to be the same.
                if (g.isInitializer(dq.inputs[1]) && initFloats(g, dq.inputs[1]).size() > 1 && q.attr.geti("axis", 1) != dq.attr.geti("axis", 1))
                {
                    continue;
                }
                // Rewire past the sandwich to a saturation clamp on the Q float input, not the raw
                // input: dropping the round-trip drops the rounding, never the clamp (see
                // insertQuantClamp). One Q feeding several DQs shares a single clamp. A hop that
                // cannot take a per-tensor clamp is left standing for a later lowering stage.
                TensorId clamped;
                auto     ci = clampOf.find(it->second);
                if (ci != clampOf.end())
                {
                    clamped = ci->second;
                } else
                {
                    clamped = insertQuantClamp(g, q.inputs[1], qz, q.inputs[0], added);
                    if (clamped == kNoTensor)
                    {
                        VKNN_WARN << "QuantizeLinear " << q.name << " sandwich kept: per-axis or non-8-bit activation quant has no per-tensor clamp";
                        continue;
                    }
                    clampOf[it->second] = clamped;
                }
                rewireTensor(g, dq.outputs[0], clamped);
                remove.insert(i);
                ++collapsed;
            }
            for (Node &c: added)
            {
                g.nodes.push_back(std::move(c));
            }
            // Sweep the QuantizeLinears the rewiring left dead (their q/dq consumers are gone).
            for (size_t i = 0; i < g.nodes.size(); ++i)
            {
                const Node &q = g.nodes[i];
                if (q.type != OpType::QuantizeLinear || remove.count(i))
                {
                    continue;
                }
                bool consumed = false;
                for (TensorId o: q.outputs)
                {
                    consumed = consumed || (o != kNoTensor && tensorConsumed(g, remove, o));
                }
                if (!consumed)
                {
                    remove.insert(i);
                }
            }
            compact(g, remove);
            // The inserted Clips were appended at the tail, after the consumers the rewiring points at
            // them; restore producer-before-consumer order so the next inferShapes visits each Clip
            // before its readers.
            if (!added.empty())
            {
                g.topoSort();
            }
            return collapsed;
        }

        // Registers an fp32 initializer holding `vals` under `name` with shape `shape`. Used for the
        // dequantized weights and rescaled biases the QLinear decomposition folds.
        TensorId addFloatInit(Graph &g, const std::string &name, const Shape &shape, std::vector<float> vals) {
            TensorDesc d;
            d.name          = name;
            d.shape         = shape;
            d.isInitializer = true;
            TensorId   id   = g.addTensor(std::move(d));
            HostBuffer hb;
            hb.resizeElems((int64_t) vals.size(), DType::Float32);
            for (size_t i = 0; i < vals.size(); ++i)
            {
                hb.f32()[i] = vals[i];
            }
            g.initializers[id] = std::move(hb);
            return id;
        }

        // Dequantizes an int-payload initializer to an fp32 initializer: W_f = (W_q - zp) * scale,
        // per-tensor (scalar scale) or per-axis (1-D scale along `axis`, negatives from the back; the
        // zero_point broadcasts as a scalar or matches the scale count). This is the weight fold shared
        // by every decomposed QLinear op (Conv/MatMul/Gemm weights). Registers the result and returns
        // its id, or kNoTensor when the operand cannot be decoded (payload dtype, shape mismatch).
        TensorId foldDequantInit(Graph &g, TensorId w, TensorId scale, TensorId zp, int64_t axis, const std::string &name) {
            std::vector<double> wv, sv, zv;
            if (!g.isInitializer(w) || !g.isInitializer(scale) || (zp != kNoTensor && !g.isInitializer(zp)))
            {
                return kNoTensor;
            }
            if (!initAsDouble(g, w, wv) || !initAsDouble(g, scale, sv) || sv.empty() || (zp != kNoTensor && !initAsDouble(g, zp, zv)))
            {
                return kNoTensor;
            }
            if (!zv.empty() && zv.size() != sv.size() && zv.size() != 1)
            {
                return kNoTensor;
            }
            const Shape &shape = g.desc(w).shape;
            int64_t      inner = 1; // elements between consecutive channel steps (per-axis form)
            if (sv.size() > 1)
            {
                int64_t rank = (int64_t) shape.size();
                int64_t ax   = axis < 0 ? axis + rank : axis;
                if (ax < 0 || ax >= rank || shape[ax] != (int64_t) sv.size())
                {
                    return kNoTensor;
                }
                for (int64_t d = ax + 1; d < rank; ++d)
                {
                    inner *= shape[d];
                }
            }
            std::vector<float> out(wv.size());
            for (size_t e = 0; e < wv.size(); ++e)
            {
                size_t c = sv.size() == 1 ? 0 : (size_t) ((int64_t) e / inner) % sv.size();
                double zpv = zv.empty() ? 0.0 : zv[zv.size() == 1 ? 0 : c];
                out[e]     = (float) ((wv[e] - zpv) * sv[c]);
            }
            return addFloatInit(g, name, shape, std::move(out));
        }

        // Rescales an int32 bias to float: B_f = B_i32 * (x_scale * w_scale), per-tensor (a scalar
        // x_scale times a scalar w_scale) or per-axis (a per-output-channel w_scale times the scalar
        // x_scale; the bias is 1-D over the same channels). Returns the id, or kNoTensor when the
        // bias/scale operands cannot be decoded.
        TensorId foldBiasInit(Graph &g, TensorId bias, TensorId xScale, TensorId wScale, const std::string &name) {
            std::vector<double> bv, xs, ws;
            if (!g.isInitializer(bias) || !g.isInitializer(xScale) || !g.isInitializer(wScale))
            {
                return kNoTensor;
            }
            if (!initAsDouble(g, bias, bv) || !initAsDouble(g, xScale, xs) || xs.empty() || !initAsDouble(g, wScale, ws) || ws.empty())
            {
                return kNoTensor;
            }
            // w_scale is scalar (per-tensor) or per-channel matching the bias length; x_scale is scalar.
            if (ws.size() > 1 && ws.size() != bv.size())
            {
                return kNoTensor;
            }
            std::vector<float> out(bv.size());
            for (size_t i = 0; i < bv.size(); ++i)
            {
                out[i] = (float) (bv[i] * xs[0] * ws[ws.size() == 1 ? 0 : i]);
            }
            return addFloatInit(g, name, g.desc(bias).shape, std::move(out));
        }

        // Resolves a QLinear op's activation operand (x / A / the Add's second addend) to its real
        // float tensor. An operand produced upstream carries dequantized float already, so it is read
        // as-is; an operand that is an INITIALIZER is a genuine quantized constant (e.g. the classifier
        // bias ORT feeds a QLinearAdd as a quantized tensor rather than a fused int32 bias) and must be
        // dequantized here with its own (scale, zero_point) -- reading its raw int payload as float
        // scales every element by 1/scale and injects the zero_point offset, which detonates the graph
        // tail. @returns the float tensor id, or kNoTensor when an initializer operand cannot fold.
        TensorId activationOperand(Graph &g, TensorId x, TensorId scale, TensorId zp, const std::string &name) {
            if (!g.isInitializer(x))
            {
                return x; // an activation edge already carries real float from its producer
            }
            return foldDequantInit(g, x, scale, zp, /*axis=*/1, name);
        }

        // Copies the arithmetic attributes an ONNX QLinearConv/QGemm carries onto its plain float
        // counterpart (kernel geometry for Conv; the GEMM transposes/scale for Gemm). The quant-only
        // attrs (channels_last, the per-axis quant axis) are dropped -- they have no float-op meaning.
        Attributes convGemmAttrs(const Node &q) {
            Attributes a;
            for (const char *k: {"auto_pad", "dilations", "group", "kernel_shape", "pads", "strides", "transA", "transB", "alpha", "beta"})
            {
                if (q.attr.has(k))
                {
                    a.map[k] = q.attr.map.at(k);
                }
            }
            return a;
        }

        // Lowers each QLinear-family op to a plain float op plus a saturation Clip to the op's OUTPUT
        // quant range (y_scale, y_zero_point), reusing insertQuantClamp. The float op reads its
        // activation operands (x, A, B) directly -- their producers already emit dequantized float --
        // and folds its weight/bias initializers to fp32 via foldDequantInit/foldBiasInit. Every
        // "quantized" activation edge is thereby repurposed to carry real float: a decomposed op's
        // output tensor id becomes the Clip's output, so a downstream QLinear op or a boundary DQ
        // reads float without a re-quantize hop. The output tensor id of each decomposed op is
        // recorded in `floatValued`, which resolveBoundaryQuant then uses to drop the redundant
        // boundary Q/DQ nodes around the decomposed core.
        //
        // A y-quant range with zero_point placing its low bound at 0 (the range ORT's QDQ quantizer
        // assigns a ReLU output) makes the Clip recover that ReLU exactly -- the clamp is never
        // dropped, so no fused nonlinearity is silently deleted (the C2b rule).
        //
        // An op whose weight/bias fold fails or whose output range cannot take a per-tensor clamp is
        // left intact (it then surfaces unsupported at planning). @returns the number decomposed.
        int decomposeQLinearOps(Graph &g, std::set<TensorId> &floatValued) {
            std::vector<Node> added; // Clip nodes appended after the iteration
            int               count = 0;
            for (size_t i = 0; i < g.nodes.size(); ++i)
            {
                Node &n = g.nodes[i];
                if (n.outputs.empty() || n.outputs[0] == kNoTensor)
                {
                    continue;
                }
                OpType   ft;    // the float op the decomposition emits
                TensorId yScale = kNoTensor, yZp = kNoTensor; // output quant range
                Node     fn;    // the rewritten float node (reuses the original output id below)
                fn.name = n.name;
                switch (n.type)
                {
                    case OpType::QLinearConv: {
                        // ins: x, x_s, x_zp, w, w_s, w_zp, y_s, y_zp, [B_i32]
                        if (n.inputs.size() < 8)
                        {
                            continue;
                        }
                        ft            = OpType::Conv;
                        TensorId xs = n.inputs[1], xzp = n.inputs[2];
                        TensorId x    = activationOperand(g, n.inputs[0], xs, xzp, n.name + "_Xf");
                        TensorId w    = n.inputs[3], ws = n.inputs[4], wzp = n.inputs[5];
                        yScale        = n.inputs[6];
                        yZp           = n.inputs[7];
                        int64_t  axis = n.attr.geti("axis", 0); // per-axis weights default to axis 0
                        TensorId wf   = foldDequantInit(g, w, ws, wzp, axis, n.name + "_Wf");
                        if (x == kNoTensor || wf == kNoTensor)
                        {
                            VKNN_WARN << "QLinearConv " << n.name << " kept: weight/activation dequant fold failed";
                            continue;
                        }
                        fn.inputs = {x, wf};
                        if (n.inputs.size() > 8 && n.inputs[8] != kNoTensor)
                        {
                            TensorId bf = foldBiasInit(g, n.inputs[8], xs, ws, n.name + "_Bf");
                            if (bf == kNoTensor)
                            {
                                VKNN_WARN << "QLinearConv " << n.name << " kept: bias rescale fold failed";
                                continue;
                            }
                            fn.inputs.push_back(bf);
                        }
                        fn.attr = convGemmAttrs(n);
                        break;
                    }
                    case OpType::QLinearMatMul: {
                        // ins: a, a_s, a_zp, b, b_s, b_zp, y_s, y_zp
                        if (n.inputs.size() < 8)
                        {
                            continue;
                        }
                        ft            = OpType::MatMul;
                        TensorId a    = activationOperand(g, n.inputs[0], n.inputs[1], n.inputs[2], n.name + "_Af");
                        TensorId b    = n.inputs[3], bs = n.inputs[4], bzp = n.inputs[5];
                        yScale        = n.inputs[6];
                        yZp           = n.inputs[7];
                        int64_t  axis = n.attr.geti("axis", 1);
                        TensorId bf   = foldDequantInit(g, b, bs, bzp, axis, n.name + "_Bf");
                        if (a == kNoTensor || bf == kNoTensor)
                        {
                            VKNN_WARN << "QLinearMatMul " << n.name << " kept: weight/activation dequant fold failed";
                            continue;
                        }
                        fn.inputs = {a, bf};
                        break;
                    }
                    case OpType::QGemm: {
                        // ins: A, a_s, a_zp, B, b_s, b_zp, [C_i32], y_s, y_zp
                        if (n.inputs.size() < 8)
                        {
                            continue;
                        }
                        ft            = OpType::Gemm;
                        bool     hasC = n.inputs.size() >= 9; // C present when 9 inputs (8 = no bias)
                        TensorId as = n.inputs[1];
                        TensorId a    = activationOperand(g, n.inputs[0], as, n.inputs[2], n.name + "_Af");
                        TensorId b    = n.inputs[3], bs = n.inputs[4], bzp = n.inputs[5];
                        yScale        = n.inputs[hasC ? 7 : 6];
                        yZp           = n.inputs[hasC ? 8 : 7];
                        int64_t  axis = n.attr.geti("axis", 0);
                        TensorId bf   = foldDequantInit(g, b, bs, bzp, axis, n.name + "_Bf");
                        if (a == kNoTensor || bf == kNoTensor)
                        {
                            VKNN_WARN << "QGemm " << n.name << " kept: weight/activation dequant fold failed";
                            continue;
                        }
                        fn.inputs = {a, bf};
                        if (hasC && n.inputs[6] != kNoTensor)
                        {
                            TensorId cf = foldBiasInit(g, n.inputs[6], as, bs, n.name + "_Cf");
                            if (cf == kNoTensor)
                            {
                                VKNN_WARN << "QGemm " << n.name << " kept: bias rescale fold failed";
                                continue;
                            }
                            fn.inputs.push_back(cf);
                        }
                        fn.attr = convGemmAttrs(n);
                        break;
                    }
                    case OpType::QLinearAdd: {
                        // ins: A, a_s, a_zp, B, b_s, b_zp, y_s, y_zp
                        if (n.inputs.size() < 8)
                        {
                            continue;
                        }
                        ft            = OpType::Binary;
                        fn.subOp      = (int) BinaryType::Add;
                        // Either addend may be an activation (read as float) or a genuine quantized
                        // initializer (dequantized here) -- ORT feeds the classifier bias as a
                        // quantized tensor on the tail QLinearAdd, so both go through activationOperand.
                        TensorId aA   = activationOperand(g, n.inputs[0], n.inputs[1], n.inputs[2], n.name + "_Af");
                        TensorId aB   = activationOperand(g, n.inputs[3], n.inputs[4], n.inputs[5], n.name + "_Bf");
                        if (aA == kNoTensor || aB == kNoTensor)
                        {
                            VKNN_WARN << "QLinearAdd " << n.name << " kept: operand dequant fold failed";
                            continue;
                        }
                        fn.inputs = {aA, aB};
                        yScale    = n.inputs[6];
                        yZp       = n.inputs[7];
                        break;
                    }
                    case OpType::QLinearGlobalAveragePool: {
                        // ins: x, x_s, x_zp, y_s, y_zp
                        if (n.inputs.size() < 5)
                        {
                            continue;
                        }
                        ft          = OpType::GlobalAvgPool;
                        TensorId gx = activationOperand(g, n.inputs[0], n.inputs[1], n.inputs[2], n.name + "_Xf");
                        if (gx == kNoTensor)
                        {
                            VKNN_WARN << "QLinearGlobalAveragePool " << n.name << " kept: activation dequant fold failed";
                            continue;
                        }
                        fn.inputs = {gx};
                        yScale    = n.inputs[3];
                        yZp       = n.inputs[4];
                        break;
                    }
                    default:
                        continue;
                }
                // The float op writes an intermediate tensor; the Clip to the output quant range takes
                // the op's original output id, so consumers read the clamped real-valued float.
                TensorDesc mid;
                mid.name     = n.name + "_pre";
                TensorId m   = g.addTensor(std::move(mid));
                TensorId yid = n.outputs[0];
                TensorId clamped = insertQuantClamp(g, yScale, yZp, m, added);
                if (clamped == kNoTensor)
                {
                    VKNN_WARN << opTypeName(n.type) << " " << n.name << " kept: output quant range has no per-tensor clamp";
                    // roll back the intermediate: leaving it dangling is harmless (pruned), but the op
                    // stays quantized and unsupported.
                    continue;
                }
                fn.type    = ft;
                fn.outputs = {m};
                // insertQuantClamp appended the Clip as the last element of `added` with a fresh
                // output id; retarget it to the op's original output so consumers (which reference
                // `yid`) read the clamp directly. The fresh id it made falls to the initializer prune.
                added.back().outputs[0] = yid;
                (void) clamped;
                n = std::move(fn); // replace the QLinear node in place with the float op
                floatValued.insert(yid);
                ++count;
            }
            for (Node &c: added)
            {
                g.nodes.push_back(std::move(c));
            }
            // The Clips were appended at the tail, after the consumers that now read them; restore
            // producer-before-consumer order so the next inferShapes visits each Clip before its
            // readers, and each float op before its Clip.
            if (!added.empty())
            {
                g.topoSort();
            }
            return count;
        }

        // Drops the boundary quantize/dequantize nodes the decomposition leaves around the float core.
        // Every quantized activation edge now carries real float, so:
        //   - a DequantizeLinear whose data input is a decomposed-op float output is a re-dequant of
        //     already-real values: drop it and rewire consumers to the float input (the trailing
        //     int8-graph-output DQ, or an interior DQ the model placed between quantized ops);
        //   - a QuantizeLinear whose float input is real (a graph input, or a decomposed float output)
        //     but whose output feeds the decomposed float core becomes a saturation Clip to its own
        //     quant range (dropping the round-trip drops only the rounding, never the clamp -- the
        //     C2b rule), and its output joins the float-valued set.
        // A DequantizeLinear over a genuine int graph input (not float-valued, not an initializer)
        // stays as a kernel node -- the explicit graph-boundary dequant the CPU op runs.
        // @returns the number of boundary nodes resolved.
        int resolveBoundaryQuant(Graph &g, std::set<TensorId> &floatValued) {
            std::map<TensorId, size_t> producer;
            for (size_t i = 0; i < g.nodes.size(); ++i)
            {
                for (TensorId o: g.nodes[i].outputs)
                {
                    if (o != kNoTensor)
                    {
                        producer[o] = i;
                    }
                }
            }
            std::vector<Node> added;
            std::set<size_t>  remove;
            int               resolved = 0;
            // Pass 1: rewrite each activation QuantizeLinear feeding the float core to a Clip.
            for (size_t i = 0; i < g.nodes.size(); ++i)
            {
                Node &n = g.nodes[i];
                if (n.type != OpType::QuantizeLinear || n.inputs.size() < 2 || n.outputs.empty() || n.outputs[0] == kNoTensor)
                {
                    continue;
                }
                TensorId f = n.inputs[0];
                if (f == kNoTensor || g.isInitializer(f))
                {
                    continue; // a const-quantize (rare) is not an activation hop; leave for the CPU op
                }
                TensorId zp = n.inputs.size() > 2 ? n.inputs[2] : kNoTensor;
                TensorId clamped = insertQuantClamp(g, n.inputs[1], zp, f, added);
                if (clamped == kNoTensor)
                {
                    continue; // per-axis / non-8-bit: leave the CPU QuantizeLinear kernel in place
                }
                rewireTensor(g, n.outputs[0], clamped);
                floatValued.insert(n.outputs[0]); // the (rewired) quantized edge now carries float
                remove.insert(i);
                ++resolved;
            }
            // Pass 2: drop each DequantizeLinear reading a now-float edge (a decomposed op output, or a
            // Q-turned-Clip output). Its data input already holds real float; the dequant is redundant.
            for (size_t i = 0; i < g.nodes.size(); ++i)
            {
                Node &n = g.nodes[i];
                if (remove.count(i) || n.type != OpType::DequantizeLinear || n.inputs.empty() || n.outputs.empty())
                {
                    continue;
                }
                TensorId in = n.inputs[0];
                if (in == kNoTensor || !floatValued.count(in))
                {
                    continue; // genuine int input (graph boundary / initializer): keep the kernel node
                }
                rewireTensor(g, n.outputs[0], in);
                floatValued.insert(n.outputs[0]);
                remove.insert(i);
                ++resolved;
            }
            for (Node &c: added)
            {
                g.nodes.push_back(std::move(c));
            }
            compact(g, remove);
            if (!added.empty())
            {
                g.topoSort();
            }
            return resolved;
        }

    } // namespace

    // Collapses ONNX quantized graphs to plain float graphs, in both ONNX quant forms:
    //   QDQ form (QuantizeLinear/DequantizeLinear checkpoints around float ops): weight-arm dequants
    //     fold to fp32 initializers and matching Q->DQ activation sandwiches drop to a saturation
    //     Clip (foldWeightDequant / collapseQdqSandwiches).
    //   QLinear form (fused QLinearConv/QLinearMatMul/QGemm/QLinearAdd/QLinearGlobalAveragePool):
    //     each op lowers to its plain float op plus a Clip to the op's own output quant range, with
    //     weights/biases folded to fp32; the boundary quantize/dequantize nodes around the float core
    //     drop, leaving only a genuine int-graph-boundary DequantizeLinear as an explicit kernel node
    //     (decomposeQLinearOps / resolveBoundaryQuant).
    // Every drop preserves the saturation clamp, never just the rounding -- that clamp is where ORT's
    // QDQ quantizer hides a fused ReLU, so collapsing to raw float without it silently deletes the
    // nonlinearity (see insertQuantClamp / the C2b rule). Any residual q/dq or QLinear node the rules
    // could not lower (per-axis activation quant, a fold failure) stays in place: no backend kernel
    // decodes the fused QLinear ops, so they surface through the support report and fail planning; the
    // graph-boundary Quantize/DequantizeLinear nodes do have CPU kernels.
    //
    // Runs after the first inferShapes. The folds need no shapes (an initializer's shape is its
    // payload dims), but running here puts the collapse before every lowering/fusion pass, so those
    // passes only ever see the float graph; the pipeline's next inferShapes then resolves the shapes
    // the rewrite left unresolved. Each rewrite reaches a fixpoint in one application, so the pass is
    // idempotent. The orphaned quantized payloads (int weights, scales, zero points) are dropped by
    // pruneDeadInitializers at the end of the pipeline.
    void dequantizeGraph(Graph &g) {
        std::set<TensorId> floatValued; // activation edges the QLinear decomposition made real-float
        int                decomposed = decomposeQLinearOps(g, floatValued);
        int                boundary   = decomposed ? resolveBoundaryQuant(g, floatValued) : 0;
        int                folded     = foldWeightDequant(g);
        int                collapsed  = collapseQdqSandwiches(g);
        if (decomposed || boundary || folded || collapsed)
        {
            VKNN_INFO << "dequantizeGraph: decomposed " << decomposed << " QLinear op(s), resolved " << boundary << " boundary q/dq, folded " << folded << " weight dequant(s), collapsed " << collapsed << " q/dq sandwich(es)";
        }
        int residual = 0;
        for (const Node &n: g.nodes)
        {
            residual += n.type == OpType::QuantizeLinear || n.type == OpType::DequantizeLinear;
        }
        if (residual)
        {
            VKNN_WARN << "dequantizeGraph: " << residual << " activation quantize/dequantize node(s) remain (graph-boundary q/dq run on the CPU kernel; interior q/dq without a kernel fail at planning)";
        }
    }

} // namespace vknn
