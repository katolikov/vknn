#include "passes_internal.h"
#include <cmath>

namespace vknn {

    /// Fuse the decomposed RMSNormalization pattern a plain-opset export leaves behind into one
    /// OpType::RMSNorm node, so the wide sum of squares accumulates in fp32 inside a single kernel
    /// instead of overflowing/losing precision across an fp16-stored Pow/ReduceMean chain.
    ///
    /// The pattern, rooted at each last-axis ReduceMean whose input is Pow(x, 2) (HF exports every
    /// RMSNorm in fp32 via explicit Casts; eliminateFloatCast has already stripped those float->float
    /// no-ops, so the chain is Cast-free here):
    ///     p   = Pow(x, 2)                     [Binary, Pow; exponent a constant 2]
    ///     m   = ReduceMean(p, axis=-1)        [Reduce, Mean; keepdims]
    ///     a   = Add(m, eps)                   [Add; eps a constant]
    ///     s   = Sqrt(a)                       [Unary, Sqrt]
    ///     r   = Div(1, s) OR Reciprocal(s)    [Binary Div with a constant-1 numerator, or Unary Reciprocal]
    ///     nrm = Mul(x, r)                     [Binary Mul; the SAME x as the Pow input]
    ///     out = Mul(gamma, nrm)               [Binary Mul; gamma an initializer of shape [norm]]
    /// becomes  out = RMSNorm(x, gamma)  with an `epsilon` attribute, i.e.
    ///     out = x * rsqrt(mean(x^2, last-axis) + eps) * gamma.
    ///
    /// Each interior tensor (the Pow/ReduceMean/Add/Sqrt/reciprocal/normalize-Mul outputs) must have a
    /// single consumer that is the next node in the chain and must not be a graph output, so removing
    /// the chain cannot orphan another reader; x and gamma stay (x also feeds the residual add). Every
    /// constant (the exponent 2, eps, the Div numerator 1) must be a resolved initializer, so this runs
    /// after the const-fold/shape-inference fixpoint. Any chain that does not match exactly is left
    /// decomposed rather than mis-lowered. Rewrites stage in `added`/`remove` and apply in one rebuild
    /// (mirrors lowerEinsum), so the loop indexes g.nodes without the vector shifting underneath it.
    void lowerRMSNorm(Graph &g) {
        // Unique producer node index per tensor, and the consuming node indices per tensor.
        std::map<TensorId, int>              producer;
        std::map<TensorId, std::vector<int>> consumers;
        for (size_t i = 0; i < g.nodes.size(); ++i)
        {
            for (TensorId o: g.nodes[i].outputs)
            {
                if (o != kNoTensor)
                {
                    producer[o] = (int) i;
                }
            }
            for (TensorId in: g.nodes[i].inputs)
            {
                if (in != kNoTensor)
                {
                    consumers[in].push_back((int) i);
                }
            }
        }
        std::set<TensorId> graphOut(g.outputs.begin(), g.outputs.end());

        // The single consuming node index of `t`, or -1 when `t` has zero/multiple consumers or is a
        // graph output (either case makes the producing node unsafe to remove).
        auto uniqueConsumer = [&](TensorId t) -> int {
            if (graphOut.count(t))
            {
                return -1;
            }
            auto it = consumers.find(t);
            if (it == consumers.end() || it->second.size() != 1)
            {
                return -1;
            }
            return it->second[0];
        };
        // True when `t` is an initializer holding a single value, returning it in `v`.
        auto constScalar = [&](TensorId t, float &v) -> bool {
            if (t == kNoTensor || !g.isInitializer(t))
            {
                return false;
            }
            std::vector<float> f = initFloats(g, t);
            if (f.size() != 1)
            {
                return false;
            }
            v = f[0];
            return true;
        };

        std::vector<Node> added;
        std::set<int>     remove;
        int               fused = 0;
        for (size_t i = 0; i < g.nodes.size(); ++i)
        {
            const Node &rm = g.nodes[i];
            if (rm.type != OpType::Reduce || rm.subOp != (int) ReduceType::Mean || rm.inputs.empty() || rm.outputs.empty())
            {
                continue;
            }
            TensorId m = rm.outputs[0]; // mean(x^2)
            TensorId p = rm.inputs[0];  // Pow output

            // Backward to Pow(x, 2); the square must feed only this ReduceMean.
            auto pit = producer.find(p);
            if (pit == producer.end() || uniqueConsumer(p) != (int) i)
            {
                continue;
            }
            const Node &pow = g.nodes[pit->second];
            if (pow.type != OpType::Binary || pow.subOp != (int) BinaryType::Pow || pow.inputs.size() < 2)
            {
                continue;
            }
            float expv;
            if (!constScalar(pow.inputs[1], expv) || std::fabs(expv - 2.0f) > 1e-6f)
            {
                continue;
            }
            TensorId x = pow.inputs[0];

            // The ReduceMean must reduce exactly the last axis (the RMSNorm feature width). An
            // unresolved input rank or any other axis set leaves the chain decomposed.
            const Shape         &xs   = g.desc(x).shape;
            int                  rank = (int) xs.size();
            std::vector<int64_t> axes = readI64Param(g, rm, "axes", 1);
            if (rank == 0 || axes.size() != 1)
            {
                continue;
            }
            int64_t ax = axes[0] < 0 ? axes[0] + rank : axes[0];
            if (ax != rank - 1)
            {
                continue;
            }

            // Forward: Add(m, eps).
            int ai = uniqueConsumer(m);
            if (ai < 0)
            {
                continue;
            }
            const Node &add = g.nodes[ai];
            if (add.type != OpType::Add || add.inputs.size() != 2)
            {
                continue;
            }
            TensorId epsT = add.inputs[0] == m ? add.inputs[1] : (add.inputs[1] == m ? add.inputs[0] : kNoTensor);
            float    epsv;
            if (!constScalar(epsT, epsv))
            {
                continue;
            }
            TensorId a = add.outputs[0];

            // Sqrt(a).
            int si = uniqueConsumer(a);
            if (si < 0)
            {
                continue;
            }
            const Node &sq = g.nodes[si];
            if (sq.type != OpType::Unary || sq.subOp != (int) UnaryType::Sqrt || sq.inputs.empty() || sq.inputs[0] != a)
            {
                continue;
            }
            TensorId s = sq.outputs[0];

            // rsqrt tail: Reciprocal(s) or Div(1, s).
            int ri = uniqueConsumer(s);
            if (ri < 0)
            {
                continue;
            }
            const Node &rc = g.nodes[ri];
            if (rc.type == OpType::Unary && rc.subOp == (int) UnaryType::Reciprocal)
            {
                if (rc.inputs.empty() || rc.inputs[0] != s)
                {
                    continue;
                }
            } else if (rc.type == OpType::Binary && rc.subOp == (int) BinaryType::Div && rc.inputs.size() == 2)
            {
                float onev;
                if (rc.inputs[1] != s || !constScalar(rc.inputs[0], onev) || std::fabs(onev - 1.0f) > 1e-6f)
                {
                    continue;
                }
            } else
            {
                continue;
            }
            TensorId r = rc.outputs[0];

            // Mul(x, r) -> nrm (operands commutative; the multiplied tensor is the SAME x as Pow's).
            int ni = uniqueConsumer(r);
            if (ni < 0)
            {
                continue;
            }
            const Node &mulN = g.nodes[ni];
            if (mulN.type != OpType::Binary || mulN.subOp != (int) BinaryType::Mul || mulN.inputs.size() != 2)
            {
                continue;
            }
            if (!((mulN.inputs[0] == x && mulN.inputs[1] == r) || (mulN.inputs[1] == x && mulN.inputs[0] == r)))
            {
                continue;
            }
            TensorId nrm = mulN.outputs[0];

            // Mul(gamma, nrm) -> out (operands commutative; gamma an initializer of shape [norm]).
            int gi = uniqueConsumer(nrm);
            if (gi < 0)
            {
                continue;
            }
            const Node &mulG = g.nodes[gi];
            if (mulG.type != OpType::Binary || mulG.subOp != (int) BinaryType::Mul || mulG.inputs.size() != 2)
            {
                continue;
            }
            TensorId gamma = mulG.inputs[0] == nrm ? mulG.inputs[1] : (mulG.inputs[1] == nrm ? mulG.inputs[0] : kNoTensor);
            if (gamma == kNoTensor || !g.isInitializer(gamma))
            {
                continue;
            }
            TensorId out = mulG.outputs[0];

            // All seven nodes matched and are disjoint from any prior chain; emit one RMSNorm.
            std::vector<int> chain {pit->second, (int) i, ai, si, ri, ni, gi};
            bool             clash = false;
            for (int c: chain)
            {
                if (remove.count(c))
                {
                    clash = true;
                }
            }
            if (clash)
            {
                continue;
            }
            Node rn;
            rn.type    = OpType::RMSNorm;
            rn.name    = mulG.name.empty() ? rm.name + "#rmsnorm" : mulG.name + "#rmsnorm";
            rn.inputs  = {x, gamma};
            rn.outputs = {out};
            Attr eps;
            eps.kind                = Attr::Float;
            eps.f                   = epsv;
            rn.attr.map["epsilon"]  = eps;
            added.push_back(std::move(rn));
            for (int c: chain)
            {
                remove.insert(c);
            }
            fused++;
        }
        if (fused)
        {
            std::vector<Node> kept;
            for (size_t i = 0; i < g.nodes.size(); ++i)
            {
                if (!remove.count((int) i))
                {
                    kept.push_back(std::move(g.nodes[i]));
                }
            }
            for (auto &a: added)
            {
                kept.push_back(std::move(a));
            }
            g.nodes = std::move(kept);
            g.topoSort();
            VKNN_INFO << "lowerRMSNorm: fused " << fused << " RMSNorm chain(s)";
        }
    }

} // namespace vknn
