// Protobuf wire-format reader for the ONNX importer: varint / length-delimited / fixed32/64, with no
// protobuf library or generated code. Reads only the fields vknn needs.
#pragma once
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace vknn {
    namespace onnx {

        // Protobuf wire types: the low 3 bits of every field tag select how the payload is framed.
        // Wire types 3/4 (deprecated group start/end) never appear in ONNX.
        enum WireType : uint32_t {
            kWireVarint  = 0, // int32/int64/bool/enum payload
            kWireFixed64 = 1,
            kWireBytes   = 2, // length-delimited: strings, sub-messages, packed repeated fields
            kWireFixed32 = 5,
        };

        /// Forward-only cursor over a protobuf-encoded byte range (an ONNX ModelProto or one of its
        /// nested messages). Every accessor advances the internal cursor `p_` past the bytes it
        /// consumes; the caller drives the message by reading a tag() and then either decoding the
        /// value for a field it wants or skip()-ping it. `end_` is one past the last valid byte, so
        /// `p_ == end_` marks a fully consumed region. Every length- or width-consuming read clamps to
        /// the bytes that remain, so a corrupt or truncated field length reads only what is in range and
        /// leaves the cursor at `end_` rather than walking past the buffer.
        class Reader {
          public:
            /// Wrap the half-open byte range [p, p + n).
            Reader(const uint8_t *p, size_t n): p_(p), end_(p + n) {
            }
            /// True once the cursor has reached (or passed) the end of the wrapped range.
            bool eof() const {
                return p_ >= end_;
            }
            /// Bytes still readable ahead of the cursor. Clamped at 0 so a cursor already at `end_`
            /// never yields a wrapped (huge) count from `end_ - p_`.
            size_t remaining() const {
                return p_ < end_ ? (size_t) (end_ - p_) : 0;
            }

            /// Decode a base-128 varint (protobuf's variable-length integer): 7 payload bits per byte,
            /// little-endian, with the high bit of each byte set while more bytes follow. Loop stops at
            /// the first byte with a clear continuation bit, or at end-of-range as a safety backstop.
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
            /// Read a fixed 32-bit little-endian value (wire type 5). The memcpy avoids an
            /// unaligned-load UB trap since `p_` is not guaranteed 4-byte aligned within the buffer.
            uint32_t fixed32() {
                uint32_t v = 0;
                if (remaining() < 4)
                {
                    p_ = end_;
                    return 0;
                }
                std::memcpy(&v, p_, 4);
                p_ += 4;
                return v;
            }
            /// Read a fixed 64-bit little-endian value (wire type 1). Same unaligned-load care as
            /// fixed32().
            uint64_t fixed64() {
                uint64_t v = 0;
                if (remaining() < 8)
                {
                    p_ = end_;
                    return 0;
                }
                std::memcpy(&v, p_, 8);
                p_ += 8;
                return v;
            }
            /// Decode the next field's tag varint into its field number and wire type. Returns false
            /// (without touching the out-params) only at end-of-range, which is the loop-termination
            /// signal for a message walk. The tag packs the field number in the high bits and the
            /// 3-bit wire type in the low 3 bits.
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
            /// Enter a length-delimited field (wire type 2): read the length prefix, return a Reader
            /// bounded to just that sub-message, and advance this cursor past it. Used to descend into
            /// nested messages (graph, node, tensor) while the parent continues where it left off.
            Reader sub() {
                uint64_t len = varint();
                if (len > remaining())
                {
                    len = remaining();
                }
                Reader r(p_, (size_t) len);
                p_ += len;
                return r;
            }
            /// Read a length-delimited field as a UTF-8 string (names, op types, attribute strings).
            std::string str() {
                uint64_t len = varint();
                if (len > remaining())
                {
                    len = remaining();
                }
                std::string s((const char *) p_, (size_t) len);
                p_ += len;
                return s;
            }
            /// Read a length-delimited field as raw bytes (packed numeric arrays, raw tensor data).
            std::vector<uint8_t> bytes() {
                uint64_t len = varint();
                if (len > remaining())
                {
                    len = remaining();
                }
                std::vector<uint8_t> b(p_, p_ + len);
                p_ += len;
                return b;
            }
            /// Advance past a field the caller does not consume, given its wire type: varint (0),
            /// fixed64 (1), fixed32 (5), and length-delimited (2) each skip their exact byte width.
            /// Wire types 3/4 (deprecated group start/end) never appear in ONNX, so an unknown wire
            /// type falls through and consumes nothing.
            void skip(uint32_t wire) {
                switch (wire)
                {
                    case kWireVarint:
                        varint();
                        break;
                    case kWireFixed64:
                        p_ += remaining() < 8 ? remaining() : 8;
                        break;
                    case kWireFixed32:
                        p_ += remaining() < 4 ? remaining() : 4;
                        break;
                    case kWireBytes: {
                        uint64_t l = varint();
                        if (l > remaining())
                        {
                            l = remaining();
                        }
                        p_ += l;
                        break;
                    }
                    default:
                        break;
                }
            }
            /// Current cursor position; between cur() and end() lies the not-yet-consumed remainder.
            const uint8_t *cur() const {
                return p_;
            }
            /// One past the last byte of the wrapped range.
            const uint8_t *end() const {
                return end_;
            }

          private:
            const uint8_t *p_;   ///< Read cursor, advanced by every accessor.
            const uint8_t *end_; ///< One past the last valid byte of the wrapped range.
        };

    } // namespace onnx
} // namespace vknn
