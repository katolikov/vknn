// Post-pass INT4 weight quantization (vknn_compile -Os): packs eligible MatMul/Gemm/Conv weights to
// 4-bit groups with fp16 scales, keeping the ~1% most activation-salient input columns in fp16
// (outlier preservation) and calibrating each group's step against the activation second moment
// (min-MSE with a diagonal-Hessian weighting — the output-MSE proxy for a linear layer). Weights the
// quantized error would distort past a relative bar stay fp16 entirely, so numerically sensitive
// layers opt themselves out (mixed precision). Runs after the standard passes, immediately before
// the fp16 initializer sweep and saveGraphBin.
//
// The packed layout, attribute names, and the dequantization contract live in core/quant_int4.h.
// The quantized weight keeps its LOGICAL TensorDesc (original shape, dtype Float16); only the
// payload bytes are packed. At load the session materializes fp16 bytes back for every consumer
// without a native int4 kernel, so quantization is transparent to all backends except the Vulkan
// MatMul, which reads the packed payload directly.
//
// Calibration runs the float graph on the CPU backend (linked into vknn_compile) over a small
// deterministic input set: caller-provided samples (--calib) or synthetic ones from a fixed-seed
// generator, so the same compile inputs always produce byte-identical .vxm output. A calibration
// failure degrades to weight-only quantization (uniform column weighting) rather than failing the
// compile.
#include "core/quant_int4.h"
#include "import/passes.h"
#include "vknn/config.h"
#include "vknn/error.h"
#include "vknn/logging.h"
#include "vknn/op.h"
#include "vknn/session.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace vknn {

    namespace {

        // One quantizable weight: the consumer node, the weight/activation tensors, and how the stored
        // tensor maps onto the logical [K, N] view (see quant_int4.h).
        struct QuantSite {
            size_t   nodeIdx = 0;
            TensorId weight  = kNoTensor;
            TensorId act     = kNoTensor;
            int64_t  K = 0, N = 0;
            int      layout = 0;  // 0 = tensor is [K,N] row-major, 1 = tensor is [N,K] row-major
            // Activation-channel granularity: a matmul/gemm column k IS activation channel k of the
            // last axis; a conv column k = (c, ky, kx) shares the per-input-channel statistic of c.
            bool    convChannels = false;
            int64_t actChannels  = 0; // K for matmul/gemm; Cin/group for conv
            int64_t convGroups   = 1; // conv group attr (activation channels span group*actChannels)
        };

        // Number of uses of tensor `t` across all node operands, fusion edges, and graph outputs.
        int64_t tensorUses(const Graph &g, TensorId t) {
            int64_t uses = 0;
            for (const Node &nd: g.nodes)
            {
                for (TensorId in: nd.inputs)
                {
                    uses += in == t;
                }
                uses += nd.fusedResidual == t;
                uses += nd.fusedBias == t;
            }
            for (TensorId out: g.outputs)
            {
                uses += out == t;
            }
            return uses;
        }

        // Collect the weights -Os quantizes: MatMul/Gemm/Conv with a single-consumer constant weight
        // large enough that 4-bit storage matters and a reduction axis deep enough for grouped scales.
        // Everything else (LayerNorm/Softmax/attention glue, small heads, depthwise convs — K = kH*kW
        // falls under minK) keeps 16-bit weights: the mixed-precision rule is structural, never
        // per-model.
        std::vector<QuantSite> collectSites(const Graph &g, const QuantOptions &opt) {
            std::vector<QuantSite> sites;
            for (size_t i = 0; i < g.nodes.size(); ++i)
            {
                const Node &nd = g.nodes[i];
                if (nd.type != OpType::MatMul && nd.type != OpType::Gemm && nd.type != OpType::Conv)
                {
                    continue;
                }
                if (pwCoreInputs(nd) < 2 || nd.attr.has(kWq))
                {
                    continue; // already quantized (a re-compiled -Os .vxm): the payload is packed bytes
                }
                const TensorId w = nd.inputs[1];
                if (!g.isInitializer(w) || tensorUses(g, w) != 1)
                {
                    continue;
                }
                const TensorDesc &wd = g.desc(w);
                if (wd.dtype != DType::Float32 && wd.dtype != DType::Float16)
                {
                    continue;
                }
                QuantSite s;
                s.nodeIdx = i;
                s.weight  = w;
                s.act     = nd.inputs[0];
                if (nd.type == OpType::MatMul)
                {
                    if (wd.shape.size() != 2)
                    {
                        continue; // the native kernel and the [K,N] view need a plain 2-D weight
                    }
                    s.K           = wd.shape[0];
                    s.N           = wd.shape[1];
                    s.layout      = 0;
                    s.actChannels = s.K;
                } else if (nd.type == OpType::Gemm)
                {
                    if (wd.shape.size() != 2 || nd.attr.geti("transA", 0) != 0)
                    {
                        continue;
                    }
                    const bool transB = nd.attr.geti("transB", 0) != 0;
                    s.K               = transB ? wd.shape[1] : wd.shape[0];
                    s.N               = transB ? wd.shape[0] : wd.shape[1];
                    s.layout          = transB ? 1 : 0;
                    s.actChannels     = s.K;
                } else // Conv
                {
                    if (wd.shape.size() != 4)
                    {
                        continue;
                    }
                    s.K            = wd.shape[1] * wd.shape[2] * wd.shape[3]; // Cin/group * kH * kW
                    s.N            = wd.shape[0];
                    s.layout       = 1; // tensor rows are output channels: [N, K] flattened
                    s.convChannels = true;
                    s.actChannels  = wd.shape[1];
                    s.convGroups   = std::max<int64_t>(nd.attr.geti("group", 1), 1);
                }
                if (s.K < opt.minK || s.K * s.N < opt.minElems || s.act == kNoTensor || g.isInitializer(s.act))
                {
                    continue;
                }
                sites.push_back(s);
            }
            return sites;
        }

        // Deterministic uniform doubles in [-1, 1) from raw generator words (the standard
        // distributions are implementation-defined, which would make the compile output depend on the
        // host's C++ library).
        struct DetRng {
            uint64_t state;
            explicit DetRng(uint64_t seed): state(seed) {}
            uint64_t next() {
                // splitmix64: a small, well-mixed generator with a portable, fixed sequence.
                uint64_t z = (state += 0x9e3779b97f4a7c15ull);
                z          = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
                z          = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
                return z ^ (z >> 31);
            }
            double uniform() {
                return (double) (next() >> 11) * (2.0 / 9007199254740992.0) - 1.0;
            }
            int64_t uniformInt(int64_t lo, int64_t hi) { // [lo, hi)
                return lo + (int64_t) (next() % (uint64_t) std::max<int64_t>(hi - lo, 1));
            }
        };

        // The extent an integer input's synthetic values must stay under: a Gather-index input is
        // bounded by its table's first extent (token ids by the vocabulary), anything else gets {0,1}.
        int64_t syntheticIntRange(const Graph &g, TensorId input) {
            for (const Node &nd: g.nodes)
            {
                if (nd.type == OpType::Gather && nd.inputs.size() >= 2 && nd.inputs[1] == input && g.isInitializer(nd.inputs[0]))
                {
                    const Shape &table = g.desc(nd.inputs[0]).shape;
                    if (!table.empty() && table[0] > 1)
                    {
                        return table[0];
                    }
                }
            }
            return 2;
        }

        // Per-activation-tensor accumulated column moments over the calibration runs. The second
        // moment weights the min-MSE step search; the first moment drives the bias correction
        // (E[Δy] = ΔW·E[x] is the systematic output shift quantization introduces).
        struct ActStats {
            std::vector<double> sum;   // per channel
            std::vector<double> sumSq; // per channel
            int64_t             count = 0;
        };

        // Fill one synthetic sample for every graph input, drawing from `rng` in graph-input order
        // (the draw order is part of the deterministic-compile contract).
        std::vector<IOTensor> syntheticSample(const Graph &g, DetRng &rng) {
            std::vector<IOTensor> inputs;
            for (TensorId id: g.inputs)
            {
                const TensorDesc &d = g.desc(id);
                IOTensor          t;
                t.name          = d.name;
                t.shape         = d.shape;
                const int64_t n = std::max<int64_t>(numElements(d.shape), 1);
                if (d.dtype == DType::Int64 || d.dtype == DType::Int32)
                {
                    const int64_t hi = syntheticIntRange(g, id);
                    t.dtype          = d.dtype;
                    t.data.resize((size_t) n * dtypeSize(d.dtype));
                    for (int64_t i = 0; i < n; ++i)
                    {
                        const int64_t v = rng.uniformInt(0, hi);
                        if (d.dtype == DType::Int64)
                        {
                            reinterpret_cast<int64_t *>(t.data.data())[i] = v;
                        } else
                        {
                            reinterpret_cast<int32_t *>(t.data.data())[i] = (int32_t) v;
                        }
                    }
                } else
                {
                    t.dtype = DType::Float32;
                    t.data.resize((size_t) n * 4);
                    float *v = reinterpret_cast<float *>(t.data.data());
                    for (int64_t i = 0; i < n; ++i)
                    {
                        v[i] = (float) rng.uniform();
                    }
                }
                inputs.push_back(std::move(t));
            }
            return inputs;
        }

        // Load one caller-provided calibration sample: raw little-endian files in graph-input order,
        // each sized exactly numElements * dtypeSize of its input (the vknn_run_io .bin convention).
        bool loadCalibSample(const Graph &g, const std::vector<std::string> &files, std::vector<IOTensor> &inputs) {
            if (files.size() != g.inputs.size())
            {
                printf("[compile] -Os calibration: sample has %zu file(s) but the model has %zu input(s)\n", files.size(), g.inputs.size());
                return false;
            }
            inputs.clear();
            for (size_t i = 0; i < files.size(); ++i)
            {
                const TensorDesc &d = g.desc(g.inputs[i]);
                FILE             *f = fopen(files[i].c_str(), "rb");
                if (!f)
                {
                    printf("[compile] -Os calibration: cannot read %s\n", files[i].c_str());
                    return false;
                }
                IOTensor t;
                t.name          = d.name;
                t.shape         = d.shape;
                t.dtype         = d.dtype == DType::Int64 || d.dtype == DType::Int32 ? d.dtype : DType::Float32;
                const int64_t n = std::max<int64_t>(numElements(d.shape), 1);
                t.data.resize((size_t) n * dtypeSize(t.dtype));
                const size_t got = fread(t.data.data(), 1, t.data.size(), f);
                fclose(f);
                if (got != t.data.size())
                {
                    printf("[compile] -Os calibration: %s holds %zu bytes, expected %zu for input %s\n", files[i].c_str(), got, t.data.size(), d.name.c_str());
                    return false;
                }
                inputs.push_back(std::move(t));
            }
            return true;
        }

        // Accumulate a captured activation's per-channel second moments. Matmul/gemm columns are the
        // last axis; conv channels are axis 1 of the NCHW activation.
        void accumulate(ActStats &st, const IOTensor &t, bool convChannels) {
            const int64_t total = std::max<int64_t>(numElements(t.shape), 0);
            if (total <= 0 || t.data.size() < (size_t) total * 4)
            {
                return;
            }
            const float *v = t.f32();
            int64_t      channels, inner;
            if (convChannels)
            {
                channels = t.shape.size() > 1 ? t.shape[1] : 1;
                inner    = 1;
                for (size_t ax = 2; ax < t.shape.size(); ++ax)
                {
                    inner *= t.shape[ax];
                }
            } else
            {
                channels = t.shape.empty() ? 1 : t.shape.back();
                inner    = 1;
            }
            if (channels <= 0)
            {
                return;
            }
            if (st.sumSq.empty())
            {
                st.sum.assign((size_t) channels, 0.0);
                st.sumSq.assign((size_t) channels, 0.0);
            }
            if ((int64_t) st.sumSq.size() != channels)
            {
                return; // shape drift across samples; keep the first-seen channel extent
            }
            const int64_t span = channels * inner;
            for (int64_t i = 0; i < total; ++i)
            {
                const int64_t c = convChannels ? (i % span) / inner : i % channels;
                st.sum[(size_t) c] += (double) v[i];
                st.sumSq[(size_t) c] += (double) v[i] * (double) v[i];
            }
            st.count += total / channels;
        }

        // Run the float graph on the CPU backend over the calibration samples, capturing each site's
        // activation input as a temporary graph output. Returns false when calibration cannot run
        // (the caller degrades to weight-only quantization).
        bool calibrate(const Graph &g, const std::vector<QuantSite> &sites, const QuantOptions &opt,
                       std::map<TensorId, ActStats> &stats) {
            // The capture set: distinct activation tensors, and whether any consumer reads them with
            // conv channel semantics (a tensor feeding both a conv and a matmul is accumulated per
            // conv channel; the matmul view then falls back to uniform weighting for safety).
            std::map<TensorId, bool> capture;
            for (const QuantSite &s: sites)
            {
                auto it = capture.find(s.act);
                if (it == capture.end())
                {
                    capture[s.act] = s.convChannels;
                } else if (it->second != s.convChannels)
                {
                    it->second = true;
                }
            }
            if (capture.empty())
            {
                return false;
            }
            Graph calib = g; // Session::create consumes its graph; quantization still needs `g`
            std::map<std::string, TensorId> nameOf;
            for (auto &kv: capture)
            {
                TensorDesc &d = calib.desc(kv.first);
                if (d.name.empty())
                {
                    d.name                     = "#calib" + std::to_string(kv.first);
                    calib.tensorByName[d.name] = kv.first;
                }
                nameOf[d.name] = kv.first;
                if (!d.isOutput)
                {
                    d.isOutput = true;
                    calib.outputs.push_back(kv.first);
                }
            }
            Config cfg;
            cfg.backend  = BackendKind::Cpu;
            cfg.fallback = {BackendKind::Cpu};
            cfg.noCache  = true;
            std::unique_ptr<Session> sess;
            try
            {
                sess = Session::create(std::move(calib), cfg);
            } catch (const Error &e)
            {
                printf("[compile] -Os calibration: CPU session failed (%s)\n", e.what());
                return false;
            }
            DetRng    rng(0x76786d34u); // fixed seed: identical compiles produce identical bytes
            const int samples = opt.calibFiles.empty() ? opt.calibSamples : (int) opt.calibFiles.size();
            int       ran     = 0;
            for (int sIdx = 0; sIdx < samples; ++sIdx)
            {
                std::vector<IOTensor> inputs;
                if (!opt.calibFiles.empty())
                {
                    if (!loadCalibSample(g, opt.calibFiles[(size_t) sIdx], inputs))
                    {
                        return false;
                    }
                } else
                {
                    inputs = syntheticSample(g, rng);
                }
                std::vector<IOTensor> outputs;
                if (sess->run(inputs, outputs) != Status::Ok)
                {
                    printf("[compile] -Os calibration: CPU run failed on sample %d\n", sIdx);
                    return false;
                }
                for (const IOTensor &out: outputs)
                {
                    auto it = nameOf.find(out.name);
                    if (it != nameOf.end())
                    {
                        accumulate(stats[it->second], out, capture[it->second]);
                    }
                }
                ++ran;
            }
            return ran > 0;
        }

        // The per-column weighting for a site: colScale[k]^2 = E[x_k^2] from calibration, replicated
        // over a conv's kH*kW taps (and averaged over conv groups, whose columns alias several input
        // channels). All-ones when calibration is unavailable or the captured extent mismatches.
        std::vector<double> columnWeights(const QuantSite &s, const std::map<TensorId, ActStats> &stats,
                                          const Graph &g) {
            std::vector<double> w((size_t) s.K, 1.0);
            auto                it = stats.find(s.act);
            if (it == stats.end() || it->second.count <= 0)
            {
                return w;
            }
            const ActStats &st = it->second;
            if (!s.convChannels)
            {
                if ((int64_t) st.sumSq.size() != s.K)
                {
                    return w;
                }
                for (int64_t k = 0; k < s.K; ++k)
                {
                    w[(size_t) k] = st.sumSq[(size_t) k] / (double) st.count;
                }
                return w;
            }
            // Conv: activation channels span group * actChannels; column (c, ky, kx) averages the
            // groups' channel moments (each output channel group reads its own slice, but the [K,N]
            // view shares one weighting per column).
            const int64_t     Cg   = s.actChannels;
            const int64_t     taps = s.K / std::max<int64_t>(Cg, 1);
            const TensorDesc &ad   = g.desc(s.act);
            (void) ad;
            if ((int64_t) st.sumSq.size() < Cg)
            {
                return w;
            }
            const int64_t groups = std::max<int64_t>((int64_t) st.sumSq.size() / Cg, 1);
            for (int64_t c = 0; c < Cg; ++c)
            {
                double m = 0;
                for (int64_t gp = 0; gp < std::min(groups, s.convGroups); ++gp)
                {
                    m += st.sumSq[(size_t) (gp * Cg + c)];
                }
                m /= (double) (std::min(groups, s.convGroups) * std::max<int64_t>(st.count, 1));
                for (int64_t t = 0; t < taps; ++t)
                {
                    w[(size_t) (c * taps + t)] = m;
                }
            }
            return w;
        }

        // Per-column activation means for a site, mirroring columnWeights' channel replication and
        // conv-group averaging. Empty when calibration is unavailable (no correction possible).
        std::vector<double> columnMeans(const QuantSite &s, const std::map<TensorId, ActStats> &stats) {
            auto it = stats.find(s.act);
            if (it == stats.end() || it->second.count <= 0)
            {
                return {};
            }
            const ActStats &st = it->second;
            std::vector<double> m((size_t) s.K, 0.0);
            if (!s.convChannels)
            {
                if ((int64_t) st.sum.size() != s.K)
                {
                    return {};
                }
                for (int64_t k = 0; k < s.K; ++k)
                {
                    m[(size_t) k] = st.sum[(size_t) k] / (double) st.count;
                }
                return m;
            }
            const int64_t Cg   = s.actChannels;
            const int64_t taps = s.K / std::max<int64_t>(Cg, 1);
            if ((int64_t) st.sum.size() < Cg)
            {
                return {};
            }
            const int64_t groups = std::max<int64_t>((int64_t) st.sum.size() / Cg, 1);
            for (int64_t c = 0; c < Cg; ++c)
            {
                double v = 0;
                for (int64_t gp = 0; gp < std::min(groups, s.convGroups); ++gp)
                {
                    v += st.sum[(size_t) (gp * Cg + c)];
                }
                v /= (double) (std::min(groups, s.convGroups) * std::max<int64_t>(st.count, 1));
                for (int64_t t = 0; t < taps; ++t)
                {
                    m[(size_t) (c * taps + t)] = v;
                }
            }
            return m;
        }

        // Subtract the systematic quantization shift ΔW·E[x] from the site's bias so every layer's
        // output keeps its calibrated mean (classic PTQ bias correction: round-to-nearest errors are
        // zero-mean per weight but NOT per output once weighted by E[x], and the shifts compound
        // through a deep stack). The correction lands in the existing bias when the node has one
        // (Conv/Gemm input[2], MatMul fusedBias); a bias-free node gets a fresh one only where that
        // cannot disturb fused-chain operand indexing.
        void applyBiasCorrection(Graph &g, Node &nd, const QuantSite &s, const std::vector<double> &delta,
                                 const std::string &weightName) {
            TensorId biasId = kNoTensor;
            if (nd.type == OpType::MatMul)
            {
                biasId = nd.fusedBias;
            } else if (pwCoreInputs(nd) >= 3)
            {
                biasId = nd.inputs[2];
            }
            if (biasId != kNoTensor)
            {
                if (!g.isInitializer(biasId) || tensorUses(g, biasId) != 1 ||
                    numElements(g.desc(biasId).shape) != s.N)
                {
                    return; // shared / runtime / oddly-shaped bias: skip rather than corrupt
                }
                const DType        bdt = g.desc(biasId).dtype;
                std::vector<float> b   = initFloats(g, biasId);
                for (int64_t n = 0; n < s.N; ++n)
                {
                    b[(size_t) n] -= (float) delta[(size_t) n];
                }
                HostBuffer hb;
                if (bdt == DType::Float16)
                {
                    std::vector<uint8_t> half((size_t) s.N * 2);
                    fp16_t              *h = reinterpret_cast<fp16_t *>(half.data());
                    for (int64_t n = 0; n < s.N; ++n)
                    {
                        h[n] = floatToHalfSat(b[(size_t) n]);
                    }
                    hb.bytes = std::move(half);
                } else if (bdt == DType::Float32)
                {
                    hb.resizeElems(s.N, DType::Float32);
                    std::memcpy(hb.f32(), b.data(), (size_t) s.N * 4);
                } else
                {
                    return;
                }
                g.initializers[biasId] = std::move(hb);
                return;
            }
            // No bias yet. A fused pointwise chain indexes its operands by input position, so only a
            // chain-free Conv/Gemm can grow an input[2]; a MatMul takes the fusedBias edge instead
            // (never an inputs change).
            TensorDesc bd;
            bd.name          = weightName + "#i4b";
            bd.shape         = {s.N};
            bd.dtype         = DType::Float32;
            bd.isInitializer = true;
            HostBuffer hb;
            hb.resizeElems(s.N, DType::Float32);
            for (int64_t n = 0; n < s.N; ++n)
            {
                hb.f32()[n] = (float) -delta[(size_t) n];
            }
            if (nd.type == OpType::MatMul)
            {
                TensorId id        = g.addTensor(bd);
                g.initializers[id] = std::move(hb);
                nd.fusedBias       = id;
            } else if (!nd.attr.has("pw_steps") && nd.inputs.size() == 2)
            {
                TensorId id        = g.addTensor(bd);
                g.initializers[id] = std::move(hb);
                nd.inputs.push_back(id);
            }
        }

    } // namespace

    QuantStats quantizeWeightsInt4(Graph &g, const QuantOptions &opt) {
        QuantStats stats;
        for (const auto &kv: g.initializers)
        {
            stats.bytesBefore += (int64_t) kv.second.bytes.size();
        }
        std::vector<QuantSite> sites = collectSites(g, opt);
        std::map<TensorId, ActStats> actStats;
        const bool calibrated = calibrate(g, sites, opt, actStats);
        if (!calibrated)
        {
            printf("[compile] -Os: no calibration statistics — quantizing weight-only (uniform column weights)\n");
        }
        for (const QuantSite &s: sites)
        {
            Node             &nd = g.nodes[s.nodeIdx];
            const TensorDesc  wdOriginal = g.desc(s.weight); // copied: addTensor below reallocates descs
            const int64_t     K = s.K, N = s.N;
            // The logical [K, N] fp32 view of the stored weight.
            std::vector<float> raw = initFloats(g, s.weight);
            std::vector<float> w((size_t) (K * N));
            if (s.layout == 0)
            {
                w = std::move(raw);
            } else
            {
                for (int64_t n = 0; n < N; ++n)
                {
                    for (int64_t k = 0; k < K; ++k)
                    {
                        w[(size_t) (k * N + n)] = raw[(size_t) (n * K + k)];
                    }
                }
            }

            const std::vector<double> colW = columnWeights(s, actStats, g);

            // Outlier preservation: the top outlierFrac of columns by (activation scale x weight
            // magnitude) stay fp16. Ties break on the lower k so the selection is deterministic.
            const double         outlierFrac = nd.type == OpType::MatMul ? opt.outlierFrac : opt.convOutlierFrac;
            const int64_t        nOut = (int64_t) ((double) K * outlierFrac);
            std::vector<int32_t> oidx;
            std::vector<char>    isOut((size_t) K, 0);
            if (nOut > 0)
            {
                std::vector<std::pair<double, int64_t>> salience((size_t) K);
                for (int64_t k = 0; k < K; ++k)
                {
                    double maxAbs = 0;
                    for (int64_t n = 0; n < N; ++n)
                    {
                        maxAbs = std::max(maxAbs, (double) std::fabs(w[(size_t) (k * N + n)]));
                    }
                    salience[(size_t) k] = {std::sqrt(std::max(colW[(size_t) k], 0.0)) * maxAbs, k};
                }
                std::stable_sort(salience.begin(), salience.end(), [](const auto &a, const auto &b) {
                    return a.first > b.first || (a.first == b.first && a.second < b.second);
                });
                oidx.reserve((size_t) nOut);
                for (int64_t j = 0; j < nOut; ++j)
                {
                    oidx.push_back((int32_t) salience[(size_t) j].second);
                }
                std::sort(oidx.begin(), oidx.end());
                for (int32_t k: oidx)
                {
                    isOut[(size_t) k] = 1;
                }
            }

            // Group-wise symmetric int4 with an activation-weighted min-MSE step search. The scale is
            // rounded to fp16 BEFORE requantization so q is optimal for the exact step the runtime
            // dequantizes with.
            const int64_t         G       = std::min<int64_t>(nd.type == OpType::MatMul ? opt.group : opt.convGroup, K);
            const int64_t         nGroups = int4GroupCount(K, G);
            std::vector<uint16_t> scales((size_t) (nGroups * N), floatToHalf(1.0f));
            std::vector<int8_t>   q((size_t) (K * N), 0);
            double                errNum = 0, errDen = 0;
            for (int64_t gp = 0; gp < nGroups; ++gp)
            {
                const int64_t k0 = gp * G, k1 = std::min(K, k0 + G);
                for (int64_t n = 0; n < N; ++n)
                {
                    double maxAbs = 0;
                    for (int64_t k = k0; k < k1; ++k)
                    {
                        if (!isOut[(size_t) k])
                        {
                            maxAbs = std::max(maxAbs, (double) std::fabs(w[(size_t) (k * N + n)]));
                        }
                    }
                    for (int64_t k = k0; k < k1; ++k)
                    {
                        if (!isOut[(size_t) k])
                        {
                            errDen += colW[(size_t) k] * (double) w[(size_t) (k * N + n)] * (double) w[(size_t) (k * N + n)];
                        }
                    }
                    if (maxAbs == 0)
                    {
                        continue; // all-zero (or all-outlier) group: q stays 0 at scale 1
                    }
                    // Candidate steps shrink the max-abs mapping from 1.0 down to 0.4: clipping a few
                    // large-magnitude weights often buys the rest of the group a finer step.
                    double bestCost = 1e300;
                    float  bestS    = 1.0f;
                    for (int cand = 0; cand < 21; ++cand)
                    {
                        const double p = 1.0 - 0.03 * (double) cand;
                        // The candidate is evaluated at its fp16-rounded value (saturated to the
                        // finite range) — the exact step the runtime dequantizes with.
                        const float s = (float) halfToFloat(floatToHalfSat((float) (maxAbs * p / 7.0)));
                        if (s <= 0)
                        {
                            continue;
                        }
                        double cost = 0;
                        for (int64_t k = k0; k < k1; ++k)
                        {
                            if (isOut[(size_t) k])
                            {
                                continue;
                            }
                            const double v  = w[(size_t) (k * N + n)];
                            double       qq = std::nearbyint(v / s);
                            qq              = std::min(7.0, std::max(-7.0, qq));
                            const double e  = v - qq * (double) s;
                            cost += colW[(size_t) k] * e * e;
                        }
                        if (cost < bestCost)
                        {
                            bestCost = cost;
                            bestS    = s;
                        }
                    }
                    scales[(size_t) (gp * N + n)] = floatToHalfSat(bestS);
                    for (int64_t k = k0; k < k1; ++k)
                    {
                        if (isOut[(size_t) k])
                        {
                            continue;
                        }
                        double qq = std::nearbyint((double) w[(size_t) (k * N + n)] / (double) bestS);
                        qq        = std::min(7.0, std::max(-7.0, qq));
                        q[(size_t) (k * N + n)] = (int8_t) qq;
                    }
                    errNum += bestCost;
                }
            }

            // Mixed-precision guard: a layer whose weighted relative error exceeds the bar keeps its
            // fp16 weight — quality is never traded silently on a sensitive layer.
            const double relErr = errDen > 0 ? std::sqrt(errNum / errDen) : 0.0;
            if (relErr > opt.maxLayerRelErr)
            {
                ++stats.guardKept;
                printf("[compile] -Os: kept %s fp16 (relative error %.4f > %.4f)\n",
                       wdOriginal.name.empty() ? nd.name.c_str() : wdOriginal.name.c_str(), relErr, opt.maxLayerRelErr);
                continue;
            }

            // Emit: packed payload into the weight tensor (logical desc unchanged apart from the fp16
            // dtype stamp), scales/outliers as side initializers referenced from the node attributes.
            std::vector<uint8_t> packed = int4Pack(q, K, N);
            std::vector<uint16_t> oval((size_t) (oidx.size() * (size_t) N));
            for (size_t j = 0; j < oidx.size(); ++j)
            {
                for (int64_t n = 0; n < N; ++n)
                {
                    oval[j * (size_t) N + (size_t) n] = floatToHalfSat(w[(size_t) ((int64_t) oidx[j] * N + n)]);
                }
            }

            auto addSide = [&](const char *suffix, DType dt, int64_t elems, const void *bytes, size_t byteCount) {
                TensorDesc d;
                d.name          = (wdOriginal.name.empty() ? ("#w" + std::to_string(s.weight)) : wdOriginal.name) + suffix;
                d.shape         = {elems};
                d.dtype         = dt;
                d.isInitializer = true;
                TensorId id     = g.addTensor(d);
                HostBuffer hb;
                std::vector<uint8_t> owned((const uint8_t *) bytes, (const uint8_t *) bytes + byteCount);
                hb.bytes = std::move(owned);
                g.initializers[id] = std::move(hb);
                return id;
            };
            const TensorId scaleId = addSide("#i4s", DType::Float16, nGroups * N, scales.data(), scales.size() * 2);
            TensorId       oidxId = kNoTensor, ovalId = kNoTensor;
            if (!oidx.empty())
            {
                oidxId = addSide("#i4oi", DType::Int32, (int64_t) oidx.size(), oidx.data(), oidx.size() * 4);
                ovalId = addSide("#i4ov", DType::Float16, (int64_t) oval.size(), oval.data(), oval.size() * 2);
            }

            {
                HostBuffer hb;
                hb.bytes = std::move(packed);
                g.initializers[s.weight] = std::move(hb);
            }
            // The desc keeps its logical shape so every shape-reading gate/planner is untouched; the
            // fp16 dtype matches what materialization reconstructs (and keeps the fp16 sweep off the
            // packed bytes).
            g.desc(s.weight).dtype = DType::Float16;

            auto seti = [&](const char *key, int64_t v) {
                Attr a;
                a.kind             = Attr::Int;
                a.i                = v;
                nd.attr.map[key]   = a;
            };
            seti(kWq, kWqVersion);
            seti(kWqK, K);
            seti(kWqN, N);
            seti(kWqGroup, G);
            seti(kWqNOut, (int64_t) oidx.size());
            seti(kWqLayout, s.layout);
            seti(kWqScales, scaleId);
            if (oidxId != kNoTensor)
            {
                seti(kWqOidx, oidxId);
                seti(kWqOval, ovalId);
            }

            // Bias correction from the calibration means: remove the systematic ΔW·E[x] output shift
            // this layer's rounding introduces (outlier columns are exact, so they contribute none).
            const std::vector<double> colMean = columnMeans(s, actStats);
            if (!colMean.empty())
            {
                std::vector<double> delta((size_t) N, 0.0);
                for (int64_t gp = 0; gp < nGroups; ++gp)
                {
                    const int64_t k0 = gp * G, k1 = std::min(K, k0 + G);
                    for (int64_t n = 0; n < N; ++n)
                    {
                        const double s16 = halfToFloat(scales[(size_t) (gp * N + n)]);
                        double       acc = 0;
                        for (int64_t k = k0; k < k1; ++k)
                        {
                            if (!isOut[(size_t) k])
                            {
                                acc += ((double) q[(size_t) (k * N + n)] * s16 - (double) w[(size_t) (k * N + n)]) * colMean[(size_t) k];
                            }
                        }
                        delta[(size_t) n] += acc;
                    }
                }
                applyBiasCorrection(g, nd, s, delta, wdOriginal.name.empty() ? ("#w" + std::to_string(s.weight)) : wdOriginal.name);
            }
            ++stats.quantized;
            stats.outlierCols += (int64_t) oidx.size();
        }
        for (const auto &kv: g.initializers)
        {
            stats.bytesAfter += (int64_t) kv.second.bytes.size();
        }
        stats.sites = (int64_t) sites.size();
        stats.calibrated = calibrated;
        return stats;
    }

} // namespace vknn
