/* libFLAC - Free Lossless Audio Codec library
 * Copyright (C) 2000-2009  Josh Coalson
 * Copyright (C) 2011-2025  Xiph.Org Foundation
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * - Redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 *
 * - Redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution.
 *
 * - Neither the name of the Xiph.org Foundation nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE FOUNDATION OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
// Adapted from xiph/flac 59fa3402cb19b6b7c85b916f138cfb03c7b27504:
// FLAC__bitwriter_write_rice_signed_block (32-bit output words) and
// bitreader_read_rice_signed_block.c (leading-zero unary decoding).
// Input is already unsigned codewords: omit audio signed folding/unfolding.
// Replace FLAC buffers/callbacks/CRC with bounded vectors and exact bit counts.
#pragma once
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace flac_rice {
using Bytes = std::vector<uint8_t>;
using Symbols = std::vector<uint16_t>;

inline Bytes encode(const Symbols &values, unsigned parameter, uint64_t bits) {
    Bytes out;
    out.reserve((bits + 7) / 8);
    uint64_t wide_accum = 0;
    unsigned bitpointer = 64;
    // FLAC's WIDE_ACCUM_TO_BW, with explicit portable big-endian output.
    auto flush_word = [&]() {
        uint32_t word = uint32_t(wide_accum >> 32);
        for (int shift = 24; shift >= 0; shift -= 8) out.push_back(uint8_t(word >> shift));
        wide_accum <<= 32;
        bitpointer += 32;
    };
    const uint32_t mask1 = UINT32_MAX << parameter;
    const uint32_t mask2 = UINT32_MAX >> (31 - parameter);
    const unsigned lsbits = 1 + parameter;
    for (uint32_t uval : values) {
        unsigned msbits = uval >> parameter;
        const unsigned total_bits = lsbits + msbits;
        uval |= mask1;  // set stop bit
        uval &= mask2; // keep stop bit and binary remainder
        if (total_bits <= bitpointer) {
            wide_accum |= uint64_t(uval) << (bitpointer - total_bits);
            bitpointer -= total_bits;
            if (bitpointer <= 32) flush_word();
        } else {
            if (msbits > bitpointer) {
                msbits -= bitpointer - 32;
                bitpointer = 32;
                flush_word();
                while (msbits > bitpointer) {
                    flush_word();
                    bitpointer -= 32;
                    msbits -= 32;
                }
                bitpointer -= msbits;
                if (bitpointer <= 32) flush_word();
            } else {
                bitpointer -= msbits;
                if (bitpointer <= 32) flush_word();
            }
            wide_accum |= uint64_t(uval) << (bitpointer - lsbits);
            bitpointer -= lsbits;
            if (bitpointer <= 32) flush_word();
        }
    }
    // Last partial word: byte padding only, never an uncharged whole word.
    const unsigned tail_bits = 64 - bitpointer;
    for (unsigned i = 0; i < (tail_bits + 7) / 8; ++i)
        out.push_back(uint8_t(wide_accum >> (56 - 8 * i)));
    if (out.size() != (bits + 7) / 8) throw std::runtime_error("Rice encoder bit count mismatch");
    return out;
}

inline Symbols decode(const Bytes &data, uint64_t bits, uint64_t count, unsigned parameter) {
    if (parameter > 16 || bits < count * (parameter + 1) || bits > count * 17 ||
        data.size() != (bits + 7) / 8)
        throw std::runtime_error("Invalid Rice parameter/bit count");
    if (bits % 8 && (data.back() & ((1u << (8 - bits % 8)) - 1)))
        throw std::runtime_error("Nonzero Rice padding");
    std::vector<uint32_t> words((data.size() + 3) / 4, 0);
    for (size_t i = 0; i < data.size(); ++i) words[i / 4] |= uint32_t(data[i]) << (24 - 8 * (i % 4));
    uint64_t position = 0;
    const uint32_t limit = 65535u >> parameter;
    Symbols values;
    values.reserve(count);
    for (uint64_t i = 0; i < count; ++i) {
        // FLAC's unary MSBs, using leading-zero count instead of per-bit loops.
        uint32_t msbs = 0;
        while (true) {
            if (position >= bits) throw std::runtime_error("Truncated Rice unary code");
            unsigned used = position % 32;
            uint32_t b = words[position / 32] << used;
            unsigned available = unsigned(std::min<uint64_t>(32 - used, bits - position));
            unsigned zeros = std::min(b ? unsigned(__builtin_clz(b)) : 32u, available);
            msbs += zeros;
            position += zeros;
            if (msbs > limit) throw std::runtime_error("Rice symbol exceeds uint16");
            if (zeros < available) { ++position; break; }
        }
        // The binary remainder crosses at most one word boundary (k <= 16).
        if (bits - position < parameter) throw std::runtime_error("Truncated Rice remainder");
        unsigned remaining = parameter;
        uint32_t lsbs = 0;
        while (remaining) {
            unsigned used = position % 32;
            unsigned take = std::min(remaining, 32 - used);
            lsbs = (lsbs << take) | ((words[position / 32] << used) >> (32 - take));
            position += take;
            remaining -= take;
        }
        values.push_back(uint16_t((msbs << parameter) | lsbs));
    }
    if (position != bits) throw std::runtime_error("Trailing Rice payload bits");
    return values;
}
} // namespace flac_rice
