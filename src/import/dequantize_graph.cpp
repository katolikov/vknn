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

    } // namespace

    // Collapses ONNX QDQ-format quantized graphs to plain float graphs: DequantizeLinear weight
    // arms fold to fp32 initializers and matching QuantizeLinear->DequantizeLinear activation
    // sandwiches drop (see the helpers above for the exact rules). Residual q/dq nodes on
    // activations -- mismatched sandwiches, graph-boundary q/dq -- stay in place: no backend
    // kernel exists for them, so they surface through the support report and fail planning.
    //
    // Runs after the first inferShapes. The fold itself needs no shapes (an initializer's shape is
    // its payload dims), but running here puts the collapse before every lowering/fusion pass, so
    // those passes only ever see the float graph; the pipeline's next inferShapes then resolves
    // the shapes the kernel-less q/dq nodes left unresolved. Both rewrites reach a fixpoint in one
    // application, so the pass is idempotent. The orphaned quantized payloads (int weights,
    // scales, zero points) are dropped by pruneDeadInitializers at the end of the pipeline.
    void dequantizeGraph(Graph &g) {
        int folded    = foldWeightDequant(g);
        int collapsed = collapseQdqSandwiches(g);
        if (folded || collapsed)
        {
            VKNN_INFO << "dequantizeGraph: folded " << folded << " weight dequant(s), collapsed " << collapsed << " q/dq sandwich(es)";
        }
        int residual = 0;
        for (const Node &n: g.nodes)
        {
            residual += n.type == OpType::QuantizeLinear || n.type == OpType::DequantizeLinear;
        }
        if (residual)
        {
            VKNN_WARN << "dequantizeGraph: " << residual << " activation quantize/dequantize node(s) remain (no kernel; reported unsupported at planning)";
        }
    }

} // namespace vknn
