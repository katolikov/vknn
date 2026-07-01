// Protobuf wire-format reader for the ONNX importer: varint / length-delimited / fixed32/64, with no
// protobuf library or generated code. Reads only the fields vknn needs.
#pragma once
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace vknn {
    namespace onnx {

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

    } // namespace onnx
} // namespace vknn
