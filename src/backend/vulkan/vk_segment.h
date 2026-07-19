// Pre-recorded Vulkan segment: liveness-pooled device buffers, static command stream, run loop.
#pragma once
#include "vk_common.h"
#include "vk_op_env.h"
#include "vknn/backend.h"
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace vknn {

    class BoundaryConvert;
    class VulkanBackend;

    /// A pre-recorded, statically-planned run of one contiguous GPU node range. The constructor does all
    /// the up-front work: allocate device activation buffers with a greedy liveness pool (internal tensors
    /// share buffers once their last use has passed; boundary/readback tensors get dedicated buffers),
    /// alias pure-copy outputs onto their input's buffer, prepare() every op (which prepacks + uploads
    /// weights), and pre-record the command buffer(s) with precise buffer-level barriers. run() then only
    /// (re)binds zero-copy dma-buf / staging boundaries, packs inputs, submits, and unpacks outputs — the
    /// static dispatch stream is reused across runs and re-recorded only when a boundary buffer changes.
    class VulkanSegment: public Segment {
      public:
        VulkanSegment(const std::vector<int> &idx, Graph &g, const Config &cfg, VulkanBackend *be);
        ~VulkanSegment() override;

        // Process resident-set size in MB (0 where /proc is unavailable). The vk buffer totals only
        // cover device allocations; on a UMA device the process RSS is what the OOM killer sees.
        static size_t hostRssMb();

        // A node runs in fp32 (selects its fp32 kernel + 4-byte buffers) when its output is storeFp32.
        bool nodeFp32(const Node &nd) const;

        void record();

        void run(ExecContext &ctx) override;

        // ---- device-resident output->input links (Session::linkOutputToInput) ----

        Status addResidentLink(TensorId sourceOutput, TensorId destInput, std::string &whyNot) override;
        void   setResidentLinkRangeSets(TensorId sourceOutput, TensorId destInput, const std::vector<std::vector<LinkRange>> &rangeSets) override;
        void   clearResidentLinks() override;
        bool   downloadResident(TensorId id, RtTensor &rt) override;
        Status setOutputArgMax(TensorId output, std::string &whyNot) override;
        Status setOutputRow(TensorId output, int64_t row, std::string &whyNot) override;
        bool   readOutputArgMax(TensorId output, int step, int64_t &index, float &value) override;
        Status configureDecodeChain(TensorId tokenInput, TensorId positionInput, TensorId maskInput, TensorId argMaxOutput, int steps, std::string &whyNot) override;
        Status setDecodeChainWindow(int64_t basePosition, int activeSteps) override;

      private:
        VulkanBackend                                  *be_;
        Graph                                          &g_;
        const Config                                   &cfg_;
        bool                                            useFp16_  = false;
        int                                             elemSize_ = 4;
        std::map<TensorId, std::shared_ptr<vk::Buffer>> buffers_;
        std::vector<std::unique_ptr<VulkanOp>>          ops_;
        VkOpEnv                                         env_;
        // Memo of the flat device buffer uploaded for each initializer of this segment's graph (weak:
        // the ops own the buffers). A weight feeding several nodes resolves through it instead of
        // re-digesting host bytes the first upload already released.
        std::map<TensorId, std::weak_ptr<vk::Buffer>> flatWeightByTensor_;
        VkCommandBuffer                               cmd_ = VK_NULL_HANDLE;
        std::vector<VkCommandBuffer> cmds_; // chunked submits (one entry unless the segment is split for the GPU watchdog; see Config::maxSubmitNodes)
        VkQueryPool                  queryPool_ = VK_NULL_HANDLE;
        bool                         recorded_  = false;
        // Config::timingSummary state: the chunk-timestamp pool, how many chunks carry a query
        // pair, and the lifetime accumulators printed once by the destructor.
        static constexpr uint32_t kMaxTimedChunks = 64;
        VkQueryPool               chunkPool_      = VK_NULL_HANDLE;
        uint32_t                  timedChunks_    = 0;
        struct {
            uint64_t runs = 0;
            double   packMs = 0, submitCallMs = 0, fenceWaitMs = 0, gpuBusyMs = 0, gpuGapMs = 0, unpackMs = 0;
        } stat_;
        std::vector<TensorId>        dumpTids_; // Config::dumpTensors debug: tensors to dump after the run
        // Zero-copy Concat/Split/Slice nodes whose EVERY slice the planner aliased as a sub-buffer
        // view: their record() emits nothing, so the barrier loop skips their hazard bookkeeping
        // (the data hazards ride the real producers/consumers through the shared arena ranges).
        std::set<int> fullyElided_;
        // Zero-copy Pad nodes: the data producer writes through a view into the padded buffer and
        // record() emits only vkCmdFillBuffer for the pad ranges — TRANSFER-stage writes, so the
        // barrier loop must treat these nodes like copies (a compute-only barrier does not order
        // them). Cleared together with fullyElided_ on any view-creation fallback.
        std::set<int> transferFillNodes_;
        // Zero-copy: each boundary tensor's pooled buffer (the fallback) and its imported dma-buf. The
        // import is kept per boundary tensor and refreshed when that tensor's dma-buf or required size
        // changes. Identity is the dma-buf's (device, inode) from fstat, not the fd: fd numbers are
        // recycled by the OS, so keying on the raw fd would alias a reused number to a stale buffer.
        std::map<TensorId, std::shared_ptr<vk::Buffer>> origBoundary_;
        struct Imported {
            uint64_t                    id    = 0; // dma-buf (dev,inode) hash (0 = unknown, fall back to fd)
            int                         fd    = -1;
            size_t                      bytes = 0;
            std::shared_ptr<vk::Buffer> buf;
        };
        std::map<TensorId, Imported> imported_;
        static uint64_t dmaBufId(int fd);
        // Declared-format zero-copy: boundary tensors whose declared dma-buf layout/dtype differs from the
        // device-native boundary, so the GPU converts between the imported buffer and the pooled boundary
        // buffer instead of binding the fd directly. `convert_` is rebuilt each run; `recordedConvert_` is
        // what the current command buffer encodes (a change re-records).
        struct ConvertBinding {
            std::shared_ptr<vk::Buffer> imported;
            bool                        isInput = true;
            NCHW                        shape;
            TensorFormat                declFmt   = TensorFormat::NCHW;
            DType                       declDtype = DType::Float32;
            TensorFormat                devFmt    = TensorFormat::NCHW;
            DType                       devDtype  = DType::Float32;
        };
        std::map<TensorId, ConvertBinding> convert_, recordedConvert_;
        std::unique_ptr<BoundaryConvert>   conv_;
        // Default-path (non-dma-buf) GPU image I/O: a persistent host-visible staging buffer per 8-bit graph
        // input. The caller's raw bytes are memcpy'd in each run and a recorded boundary_convert turns them
        // into the device-native boundary — no host uint8->fp32->fp16 pack. Allocated once, stable identity.
        std::map<TensorId, std::shared_ptr<vk::Buffer>> stagingIn_;
        std::set<TensorId>                              graphInputs_; // g_.inputs, for the staging-input gate
        // Device-resident output->input links (Session::linkOutputToInput). Each link's ranges live in a
        // small host-visible SSBO the recorded link_copy dispatch reads, so per-run range updates (the
        // moving destination slot of a KV fold) need no re-record — the copy runs at the head of the
        // first command chunk, reading the source buffer's PREVIOUS-run values (nothing has executed
        // yet) and writing the destination in place. Both buffers are dedicated boundary buffers whose
        // identity never changes, so the recording stays valid across runs.
        struct ResidentLink {
            TensorId                    src = kNoTensor, dst = kNoTensor;
            std::shared_ptr<vk::Buffer> rangesBuf;    // header {count,total} + 3 uints per range
            uint32_t                    capacity = 0; // ranges rangesBuf can hold
        };
        std::vector<ResidentLink>            residentLinks_;
        std::set<TensorId>                   linkedInputs_, linkedOutputs_;
        bool                                 linksChanged_ = false; // link set / ranges buffer identity changed -> re-record
        std::unique_ptr<vk::ComputePipeline> linkPipeFp16_, linkPipeFp32_;
        static constexpr uint32_t            kLinkRangeHeaderBytes     = 8;  // {rangeCount, totalElems}
        static constexpr uint32_t            kLinkInitialRangeCapacity = 16; // ranges per set; grows on demand
        static constexpr uint32_t            kLinkCopyGroups           = 4;  // fixed grid; the shader strides over totalElems
        // A link's ranges buffer holds one range set per chain iteration at a fixed stride; a
        // single-step segment (chainStepsMax_ == 1) holds exactly the original one-set layout.
        size_t linkRangesBufferBytes(uint32_t rangeCapacity) const;
        // Device-resident decode chain (Session::configureDecodeChain, ADR-0015): the segment
        // records chainSteps_ decode iterations into one command-buffer sequence. Between
        // iterations a chain_feedback dispatch writes the previous iteration's argmax index into
        // the token input, advances the position input, and marks the newly valid mask slot; the
        // link copies apply per-iteration range sets and the argmax epilogue lands in per-iteration
        // result slots. A run submits the chunk prefix covering chainActiveSteps_ iterations, so
        // one recording serves every chain length from 1 to chainSteps_ with no re-record.
        struct DecodeChain {
            TensorId tokenInput = kNoTensor, positionInput = kNoTensor, maskInput = kNoTensor;
            TensorId argMaxOutput = kNoTensor;
        };
        DecodeChain                          chain_;
        bool                                 chainConfigured_  = false;
        int                                  chainStepsMax_    = 1; // Config::decodeChainSteps; sizes the per-iteration buffers
        int                                  chainSteps_       = 1; // recorded iterations (<= chainStepsMax_)
        int                                  chainActiveSteps_ = 1; // iterations the next runs submit (prefix of the recording)
        std::shared_ptr<vk::Buffer>          chainStateBuf_;        // {uint basePosition}, host-written per chain
        std::vector<uint32_t>                iterationFirstChunk_;  // cmds_ index of each recorded iteration's first chunk
        std::unique_ptr<vk::ComputePipeline> chainFeedbackPipe_;
        bool                                 chainChanged_ = false; // chain configuration changed -> re-record
        // Command-buffer chunks a run submits: the whole recording, or the prefix covering the
        // active chain steps (each iteration past the prefix starts a fresh chunk by construction).
        uint32_t chunksForActiveSteps() const;
        // Device-side output argmax (Session::setOutputArgMax): one 8-byte host-visible result
        // buffer {uint index, float value} per registered output, written by the argmax_flat
        // epilogue dispatch the recording appends after the graph's nodes.
        std::set<TensorId>                              argMaxOutputs_;
        std::map<TensorId, std::shared_ptr<vk::Buffer>> argMaxResults_;
        bool                                            argMaxChanged_ = false; // registration set changed -> re-record
        std::map<TensorId, int64_t>                     rowSelectOutputs_;      // output -> the single flat row to read back (setOutputRow)
        std::unique_ptr<vk::ComputePipeline>            argMaxPipeFp16_, argMaxPipeFp32_;
        static constexpr uint32_t                       kArgMaxResultBytes = 8; // {index, value}
        // Device element width of a boundary tensor (2 for fp16 storage, 4 for fp32/storeFp32).
        int boundaryElemBytes(TensorId tid) const;
        static bool sameConvert(const std::map<TensorId, ConvertBinding> &a, const std::map<TensorId, ConvertBinding> &b);
    };

} // namespace vknn
