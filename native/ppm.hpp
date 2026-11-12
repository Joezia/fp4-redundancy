// Complete Nayuki-style PPM adaptation with the existing A1 arithmetic coder.
// Context model provenance/license: third_party/Nayuki/PpmModel.hpp.
#pragma once
#include "third_party/Nayuki/PpmModel.hpp"

namespace ppm_codec {
using namespace nayuki_ppm;
inline Stream encode(const Symbols &symbols, uint32_t alphabet) {
    if (!alphabet || alphabet > 65536 || (alphabet & (alphabet - 1)))
        throw std::runtime_error("Invalid PPM alphabet");
    PpmModel model(alphabet);
    yaecl::ArithmeticCodingEncoder<uint64_t, uint32_t> coder(32);
    coder.bit_stream.reserve(symbols.size() * 2 + 16);
    for (uint32_t symbol : symbols) {
        if (symbol >= alphabet) throw std::runtime_error("PPM symbol outside complete codeword alphabet");
        auto contexts = model.contexts();
        bool found = false;
        for (auto it = contexts.rbegin(); it != contexts.rend(); ++it) {
            if (!*it) continue;
            auto interval = (*it)->frequencies.interval(symbol);
            found = interval.low != interval.high;
            if (!found) interval = (*it)->frequencies.interval(alphabet); // escape
            coder.encode_interval(interval.low, interval.high, CDF_BITS);
            if (found) break;
        }
        if (!found) {
            uint32_t unit = CDF_TOTAL / alphabet;
            coder.encode_interval(symbol * unit, (symbol + 1) * unit, CDF_BITS);
        }
        model.incrementContexts(symbol);
    }
    coder.flush();
    Stream result{symbols.size(), {}, coder.bit_stream.bytes()};
    for (uint32_t value : {1u, alphabet, ORDER, MAX_CONTEXTS, RESCALE, CDF_BITS}) put<uint32_t>(result.model, value);
    put<uint64_t>(result.model, coder.bit_stream.size());
    return result;
}
inline Symbols decode(const Stream &stream) {
    Reader reader{stream.model.data(), stream.model.size()};
    if (reader.get<uint32_t>() != 1) throw std::runtime_error("Invalid PPM version");
    uint32_t alphabet = reader.get<uint32_t>();
    if (!alphabet || alphabet > 65536 || (alphabet & (alphabet - 1)) ||
        reader.get<uint32_t>() != ORDER || reader.get<uint32_t>() != MAX_CONTEXTS ||
        reader.get<uint32_t>() != RESCALE || reader.get<uint32_t>() != CDF_BITS)
        throw std::runtime_error("Invalid PPM configuration");
    uint64_t bits = reader.get<uint64_t>();
    if (reader.position != reader.size || bits < 2 || bits > INT32_MAX || (bits + 7) / 8 != stream.data.size())
        throw std::runtime_error("Invalid PPM payload bit count");
    if (bits % 8 && (stream.data.back() & ((1u << (8 - bits % 8)) - 1)))
        throw std::runtime_error("Invalid PPM padding");
    yaecl::BitStream input;
    input.assign(stream.data.data(), stream.data.size(), int(bits));
    yaecl::ArithmeticCodingDecoder<uint64_t, uint32_t> coder(32, input);
    PpmModel model(alphabet);
    Symbols result;
    result.reserve(stream.count);
    for (uint64_t i = 0; i < stream.count; ++i) {
        auto contexts = model.contexts();
        bool found = false;
        uint32_t symbol = 0;
        for (auto it = contexts.rbegin(); it != contexts.rend(); ++it) {
            if (!*it) continue;
            auto interval = (*it)->frequencies.select(coder.cumulative(CDF_BITS));
            coder.decode_interval(interval.low, interval.high, CDF_BITS);
            if (interval.symbol != alphabet) { symbol = interval.symbol; found = true; break; }
        }
        if (!found) {
            uint32_t unit = CDF_TOTAL / alphabet;
            symbol = coder.cumulative(CDF_BITS) / unit;
            coder.decode_interval(symbol * unit, (symbol + 1) * unit, CDF_BITS);
        }
        if (symbol >= alphabet) throw std::runtime_error("Invalid PPM output symbol");
        result.push_back(uint16_t(symbol));
        model.incrementContexts(symbol);
    }
    return result;
}
} // namespace ppm_codec
