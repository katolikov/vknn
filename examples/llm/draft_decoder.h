// The DRAFT half of greedy speculative decoding (vknn/spec_decode.h): a second, much smaller
// with-past decoder that proposes the next few tokens for the target to verify.
//
// It is deliberately the simplest possible driver — the HOST KV cache flow, binding the whole cache
// per run and folding the produced rows back on the host — for three reasons. Its cache is small
// (that is what makes it a draft model). The host cache is the bridge between its prompt-catch-up
// bucket and its per-token bucket, so no resident-state materialization is needed when the window
// changes. And, decisively, NOTHING here can affect the emitted token stream: a proposal is only
// ever emitted after the target's own argmax has confirmed it, so a draft that is slow, wrong, or
// holding a stale cache costs throughput and nothing else. The engine-resident KV link is the
// obvious next optimization for this file, and it is a throughput change, not a correctness one.
//
// The rollback after a partially accepted round is `position = anchor`: rows past the committed
// position are stale, but a decoder only attends the slots its attention mask marks valid, and the
// mask is built from the position — so those rows are invisible and the next round's proposals
// overwrite them before they could become visible.
#pragma once
#include "vknn/runtime.h"
#include "vknn/session.h"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <exception>
#include <memory>
#include <string>
#include <vector>

namespace vknn {

    /// A draft decoder loaded from its own .vxm, driven on the host KV cache flow.
    class DraftDecoder {
    public:
        /// Load and describe a draft model. Returns null when the artifact does not load or is not a
        /// with-past decoder this driver can feed — speculation then simply does not engage, which is
        /// why every failure here is a notice rather than an error.
        static std::unique_ptr<DraftDecoder> open(const std::string &path, const Config &cfg) {
            auto draft = std::unique_ptr<DraftDecoder>(new DraftDecoder());
            // A malformed or non-model artifact reaches the loader as an exception, not a null
            // session. The primary model is allowed to end the process that way; the draft is not —
            // it is an optimization, so a bad path costs speculation and nothing else.
            try
            {
                draft->session_ = Runtime::load(path, cfg);
            } catch (const std::exception &e)
            {
                fprintf(stderr, "[chat] draft model %s did not load (%s); decoding without speculation\n", path.c_str(), e.what());
                return nullptr;
            } catch (...)
            {
                fprintf(stderr, "[chat] draft model %s did not load; decoding without speculation\n", path.c_str());
                return nullptr;
            }
            if (!draft->session_)
            {
                fprintf(stderr, "[chat] draft model %s failed to load; decoding without speculation\n", path.c_str());
                return nullptr;
            }
            if (!draft->describe())
            {
                fprintf(stderr, "[chat] draft model %s is not a with-past decoder this driver can feed; decoding without speculation\n", path.c_str());
                return nullptr;
            }
            fprintf(stderr, "[chat] draft model %s: layers=%d kv_heads=%d C=%d head_dim=%d vocab=%lld, catch-up window %d\n",
                    path.c_str(), layerCount(*draft), draft->kvHeads_, draft->cacheSlots_, draft->headDim_, (long long) draft->vocab_, draft->catchUpWindow_);
            return draft;
        }

        int64_t vocab() const noexcept {
            return vocab_;
        }
        /// Compiled context length: the draft can hold rows for positions 0..cacheSlots-1.
        int cacheSlots() const noexcept {
            return cacheSlots_;
        }
        /// Absolute position the next fed token occupies; rows 0..position-1 are live.
        int position() const noexcept {
            return position_;
        }

        /// Move the write cursor to `position`. Rows at or past it stay in the buffers but are
        /// masked out of every later forward and overwritten by the next tokens fed — the rollback
        /// after a partially accepted round, and the way a caller re-aligns the draft with the
        /// conversation prefix it is known to hold.
        void rewind(int position) noexcept {
            position_ = position;
        }

        /// Drop every cached row and restart at position 0 (a conversation reset).
        void reset() {
            for (int layer = 0; layer < (int) pastKey_.size(); ++layer)
            {
                std::fill(inputs_[(size_t) pastKey_[(size_t) layer]].data.begin(), inputs_[(size_t) pastKey_[(size_t) layer]].data.end(), (uint8_t) 0);
                std::fill(inputs_[(size_t) pastVal_[(size_t) layer]].data.begin(), inputs_[(size_t) pastVal_[(size_t) layer]].data.end(), (uint8_t) 0);
            }
            position_ = 0;
        }

        /// Consume `count` tokens starting at the current position, in windows of the widest bucket
        /// this model carries — the prompt catch-up that keeps the draft's cache aligned with the
        /// target's conversation. False on a run failure or a prompt that would cross the draft's
        /// compiled context edge (speculation then stops for the turn; the target decodes plainly).
        bool consume(const int64_t *tokens, int count) {
            int done = 0;
            while (done < count)
            {
                const int len = std::min(count - done, catchUpWindow_);
                if (position_ + len > cacheSlots_)
                {
                    return false;
                }
                if (!windowPass(catchUpBucket_, catchUpWindow_, tokens + done, len))
                {
                    return false;
                }
                done += len;
            }
            return true;
        }

        /// Propose `count` tokens extending `anchorToken`, which sits at absolute position `anchor`.
        /// Rewinds to `anchor` first (the rollback of the previous round's rejected proposals) and
        /// then runs `count` single-token forwards, feeding each proposal forward. False when the
        /// draft would run past its own context edge or a run fails.
        bool propose(int64_t anchorToken, int anchor, int64_t *proposals, int count) {
            if (anchor < 0 || anchor + count > cacheSlots_)
            {
                return false;
            }
            position_    = anchor;
            int64_t feed = anchorToken;
            for (int i = 0; i < count; ++i)
            {
                if (!windowPass(decodeBucket_, 1, &feed, 1))
                {
                    return false;
                }
                proposals[i] = argMaxRow();
                feed         = proposals[i];
            }
            return true;
        }

    private:
        DraftDecoder() = default;

        static int layerCount(const DraftDecoder &draft) noexcept {
            return (int) draft.pastKey_.size();
        }

        int findIn(const std::string &name) const {
            for (size_t i = 0; i < ins_.size(); ++i)
            {
                if (ins_[i].name == name)
                {
                    return (int) i;
                }
            }
            return -1;
        }
        int findOut(const std::string &name) const {
            for (size_t i = 0; i < outs_.size(); ++i)
            {
                if (outs_[i].name == name)
                {
                    return (int) i;
                }
            }
            return -1;
        }

        /// Locate the decode bucket, the widest catch-up bucket, and the whole cache geometry.
        bool describe() {
            decodeBucket_   = -1;
            catchUpBucket_  = -1;
            catchUpWindow_  = 0;
            for (size_t b = 0; b < session_->bucketCount(); ++b)
            {
                for (const IOInfo &in: session_->inputInfo(b))
                {
                    if (in.name != "input_ids" || in.shape.size() != 2 || in.shape[0] != 1)
                    {
                        continue;
                    }
                    if (in.shape[1] == 1 && decodeBucket_ < 0)
                    {
                        decodeBucket_ = (int) b;
                    } else if (in.shape[1] > catchUpWindow_)
                    {
                        catchUpBucket_ = (int) b;
                        catchUpWindow_ = (int) in.shape[1];
                    }
                }
            }
            if (decodeBucket_ < 0)
            {
                return false;
            }
            ins_  = session_->inputInfo((size_t) decodeBucket_);
            outs_ = session_->outputInfo((size_t) decodeBucket_);
            idIdx_     = findIn("input_ids");
            maskIdx_   = findIn("attention_mask");
            posIdx_    = findIn("position_ids");
            logitsIdx_ = findOut("logits");
            for (int layer = 0;; ++layer)
            {
                char keyName[64], valName[64], presKeyName[64], presValName[64];
                snprintf(keyName, sizeof keyName, "past_key_values.%d.key", layer);
                snprintf(valName, sizeof valName, "past_key_values.%d.value", layer);
                const int keyIdx = findIn(keyName), valIdx = findIn(valName);
                if (keyIdx < 0 || valIdx < 0)
                {
                    break;
                }
                snprintf(presKeyName, sizeof presKeyName, "present.%d.key", layer);
                snprintf(presValName, sizeof presValName, "present.%d.value", layer);
                const int presKeyIdx = findOut(presKeyName), presValIdx = findOut(presValName);
                if (presKeyIdx < 0 || presValIdx < 0)
                {
                    return false;
                }
                pastKey_.push_back(keyIdx);
                pastVal_.push_back(valIdx);
                presKey_.push_back(presKeyIdx);
                presVal_.push_back(presValIdx);
            }
            // position_ids is required: the catch-up and proposal passes feed absolute positions, so
            // a mask-derived-position export cannot be driven here.
            if (idIdx_ < 0 || maskIdx_ < 0 || posIdx_ < 0 || logitsIdx_ < 0 || pastKey_.empty())
            {
                return false;
            }
            const Shape &cacheShape = ins_[(size_t) pastKey_[0]].shape; // [1, kvHeads, C, headDim]
            if (cacheShape.size() != 4)
            {
                return false;
            }
            kvHeads_    = (int) cacheShape[1];
            cacheSlots_ = (int) cacheShape[2];
            headDim_    = (int) cacheShape[3];
            vocab_      = outs_[(size_t) logitsIdx_].shape.back();
            kvElemBytes_ = dtypeSize(ins_[(size_t) pastKey_[0]].dtype);
            logitsFp16_  = outs_[(size_t) logitsIdx_].dtype == DType::Float16;
            presRows_    = outs_[(size_t) presKey_[0]].shape.size() == 4 ? (int) outs_[(size_t) presKey_[0]].shape[2] : 0;
            if (presRows_ <= 0 || dtypeSize(outs_[(size_t) presKey_[0]].dtype) != kvElemBytes_)
            {
                return false; // the host fold copies present rows straight into the past buffers
            }
            for (int layer = 0; layer < (int) presKey_.size(); ++layer)
            {
                for (int part = 0; part < 2; ++part)
                {
                    const IOInfo &out = outs_[(size_t) (part ? presVal_[(size_t) layer] : presKey_[(size_t) layer])];
                    if (out.shape.size() != 4 || (int) out.shape[2] != presRows_)
                    {
                        return false; // one present-row convention must serve every layer's fold
                    }
                }
            }
            if (catchUpBucket_ < 0)
            {
                catchUpBucket_ = decodeBucket_; // no batched bucket: catch up token by token
                catchUpWindow_ = 1;
            } else if (!catchUpGeometryOk())
            {
                catchUpBucket_ = decodeBucket_;
                catchUpWindow_ = 1;
            }
            inputs_.assign(ins_.size(), IOTensor {});
            for (size_t i = 0; i < ins_.size(); ++i)
            {
                inputs_[i].name  = ins_[i].name;
                inputs_[i].shape = ins_[i].shape;
                inputs_[i].dtype = ins_[i].dtype;
                inputs_[i].data.assign((size_t) ins_[i].elems * dtypeSize(ins_[i].dtype), 0);
            }
            position_ = 0;
            return true;
        }

        /// The catch-up bucket must share the decode bucket's cache shapes and span past + window
        /// mask columns; anything else falls back to the token-by-token catch-up.
        bool catchUpGeometryOk() const {
            bool maskOk = false, cacheOk = true, presOk = false;
            for (const IOInfo &in: session_->inputInfo((size_t) catchUpBucket_))
            {
                if (in.name == ins_[(size_t) pastKey_[0]].name)
                {
                    cacheOk = cacheOk && in.shape == ins_[(size_t) pastKey_[0]].shape;
                }
                if (in.name == "attention_mask" && in.shape.size() == 2)
                {
                    maskOk = in.shape[1] == cacheSlots_ + catchUpWindow_;
                }
                if (in.name == "position_ids" && in.shape != Shape {1, (int64_t) catchUpWindow_})
                {
                    return false;
                }
            }
            for (const IOInfo &out: session_->outputInfo((size_t) catchUpBucket_))
            {
                if (out.name.rfind("present.", 0) != 0 || out.shape.size() != 4)
                {
                    continue;
                }
                if (out.name == outs_[(size_t) presKey_[0]].name)
                {
                    presOk = out.shape[2] >= catchUpWindow_;
                }
            }
            return maskOk && cacheOk && presOk;
        }

        /// One forward of `window` token columns carrying `len` real tokens at the current position:
        /// bind the cache, fold the produced rows back into slots position..position+len-1, keep the
        /// last real token's logits row, and advance the position by `len`. Pad columns stay masked,
        /// so their rows are neither attended nor folded.
        bool windowPass(int bucket, int window, const int64_t *tokens, int len) {
            const std::vector<IOInfo> bucketIns  = session_->inputInfo((size_t) bucket);
            const std::vector<IOInfo> bucketOuts = session_->outputInfo((size_t) bucket);
            int                       maskLen    = 0, presRows = 0;
            for (const IOInfo &in: bucketIns)
            {
                if (in.name == "attention_mask" && in.shape.size() == 2)
                {
                    maskLen = (int) in.shape[1];
                }
            }
            for (const IOInfo &out: bucketOuts)
            {
                if (out.name == outs_[(size_t) presKey_[0]].name && out.shape.size() == 4)
                {
                    presRows = (int) out.shape[2];
                }
            }
            if (maskLen != cacheSlots_ + window || presRows < window)
            {
                return false;
            }
            std::vector<IOTensor> bound;
            bound.reserve(ins_.size());
            for (size_t i = 0; i < ins_.size(); ++i)
            {
                IOTensor tensor;
                tensor.name  = ins_[i].name;
                tensor.dtype = ins_[i].dtype;
                if ((int) i == idIdx_ || (int) i == posIdx_)
                {
                    tensor.shape = {1, (int64_t) window};
                    std::vector<int64_t> values((size_t) window, 0);
                    for (int t = 0; t < window; ++t)
                    {
                        values[(size_t) t] = ((int) i == idIdx_) ? (t < len ? tokens[t] : 0) : (int64_t) (position_ + t);
                    }
                    tensor.data.resize((size_t) window * 8);
                    std::memcpy(tensor.data.data(), values.data(), tensor.data.size());
                } else if ((int) i == maskIdx_)
                {
                    tensor.shape = {1, (int64_t) maskLen};
                    std::vector<int64_t> values((size_t) maskLen, 0);
                    for (int j = 0; j < position_ && j < cacheSlots_; ++j)
                    {
                        values[(size_t) j] = 1; // valid past slots
                    }
                    for (int t = 0; t < len; ++t)
                    {
                        values[(size_t) (cacheSlots_ + t)] = 1; // the window's real tokens
                    }
                    tensor.data.resize((size_t) maskLen * 8);
                    std::memcpy(tensor.data.data(), values.data(), tensor.data.size());
                } else
                {
                    tensor.shape = ins_[i].shape;
                    tensor.data  = inputs_[i].data; // the host KV cache
                }
                bound.push_back(std::move(tensor));
            }
            if (session_->run(bound, outputs_) != Status::Ok)
            {
                return false;
            }
            const IOTensor *logits = nullptr;
            for (const IOTensor &out: outputs_)
            {
                if (out.name == "logits")
                {
                    logits = &out;
                }
            }
            if (!logits || logits->data.empty())
            {
                return false;
            }
            const int newRowsAt = presRows - window; // produced rows sit after any past block
            for (int layer = 0; layer < (int) pastKey_.size(); ++layer)
            {
                for (int part = 0; part < 2; ++part)
                {
                    const std::string &presName = outs_[(size_t) (part ? presVal_[(size_t) layer] : presKey_[(size_t) layer])].name;
                    const IOTensor    *present  = nullptr;
                    for (const IOTensor &out: outputs_)
                    {
                        if (out.name == presName)
                        {
                            present = &out;
                        }
                    }
                    if (!present || present->data.empty())
                    {
                        return false;
                    }
                    uint8_t *dst = inputs_[(size_t) (part ? pastVal_[(size_t) layer] : pastKey_[(size_t) layer])].data.data();
                    for (int head = 0; head < kvHeads_; ++head)
                    {
                        std::memcpy(dst + ((size_t) head * cacheSlots_ + position_) * headDim_ * kvElemBytes_,
                                    present->data.data() + ((size_t) head * presRows + newRowsAt) * headDim_ * kvElemBytes_,
                                    (size_t) len * headDim_ * kvElemBytes_);
                    }
                }
            }
            position_ += len;
            const size_t rowOffset = (size_t) (len - 1) * (size_t) vocab_;
            logitsRow_.resize((size_t) vocab_);
            if (logitsFp16_)
            {
                halfToFloatBulk(reinterpret_cast<const fp16_t *>(logits->data.data()) + rowOffset, logitsRow_.data(), vocab_);
            } else
            {
                std::memcpy(logitsRow_.data(), reinterpret_cast<const float *>(logits->data.data()) + rowOffset, (size_t) vocab_ * 4);
            }
            return true;
        }

        /// First-occurrence argmax of the last pass's logits row — the draft's own greedy choice.
        int64_t argMaxRow() const {
            int64_t best = 0;
            for (int64_t i = 1; i < vocab_; ++i)
            {
                if (logitsRow_[(size_t) i] > logitsRow_[(size_t) best])
                {
                    best = i;
                }
            }
            return best;
        }

        std::unique_ptr<Session> session_;
        std::vector<IOInfo>      ins_, outs_;
        std::vector<IOTensor>    inputs_;  ///< Persistent bound inputs; the past entries ARE the cache.
        std::vector<IOTensor>    outputs_; ///< Reused run() result vector.
        std::vector<float>       logitsRow_;
        std::vector<int>         pastKey_, pastVal_, presKey_, presVal_;
        int                      decodeBucket_ = -1, catchUpBucket_ = -1, catchUpWindow_ = 1;
        int                      idIdx_ = -1, maskIdx_ = -1, posIdx_ = -1, logitsIdx_ = -1;
        int                      kvHeads_ = 0, cacheSlots_ = 0, headDim_ = 0, presRows_ = 0;
        int                      position_    = 0;
        size_t                   kvElemBytes_ = 4;
        bool                     logitsFp16_  = false;
        int64_t                  vocab_       = 0;
    };

} // namespace vknn
