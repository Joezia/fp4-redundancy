// Buffer/layout glue around pinned upstream coders; no Python symbol/bit loops.
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
#include "SZ3/encoder/HuffmanEncoder.hpp"
#include "yaecl.hpp"
#include "rans_byte.h"
#include "third_party/FLAC/rice.hpp"

using Bytes = std::vector<uint8_t>;
using Symbols = std::vector<uint16_t>;
static constexpr uint64_t MAX_RAW = 16u << 20;
static constexpr uint32_t MAGIC = 0x31414e46; // FNA1; private C/Python buffer protocol.
static thread_local std::string error_message;

template<class T> void put(Bytes &b, T value) {
    for (unsigned i = 0; i < sizeof(T); ++i) b.push_back(uint8_t(value >> (8 * i)));
}
struct Reader {
    const uint8_t *data; size_t size, position = 0;
    template<class T> T get() {
        if (sizeof(T) > size - position) throw std::runtime_error("Truncated native header");
        T v = 0;
        for (unsigned i = 0; i < sizeof(T); ++i) v |= T(data[position++]) << (8 * i);
        return v;
    }
    const uint8_t *take(size_t n) {
        if (n > size - position) throw std::runtime_error("Truncated native data");
        const auto *p = data + position; position += n; return p;
    }
};
struct Stream { uint64_t count; Bytes model, data; };
#include "ppm.hpp"
#include "ctw.hpp"
#include "a2_tree.hpp"
#include "context_models.hpp"
#include "neural.hpp"
#include "cavlc.hpp"

// The alphabet stays complete codewords (uint16_t), with an order-preserving
// dense rank only for CDF indexing. Serialize the inverse mapping and all frequencies.
struct Distribution {
    unsigned precision;
    Symbols alphabet;
    std::vector<uint32_t> frequency, cumulative, rank;
};
Distribution distribution(const Symbols &symbols, bool arithmetic) {
    std::vector<uint64_t> counts(65536, 0);
    for (auto s : symbols) ++counts[s];
    Distribution d;
    d.rank.resize(65536);
    for (unsigned s = 0; s < counts.size(); ++s) if (counts[s]) d.alphabet.push_back(uint16_t(s));
    d.precision = arithmetic ? 24 : 12;
    while ((1u << d.precision) < d.alphabet.size()) ++d.precision;
    const uint32_t total = 1u << d.precision;
    const uint64_t remaining = total - d.alphabet.size();
    d.frequency.resize(d.alphabet.size());
    std::vector<uint64_t> remainders(d.alphabet.size());
    uint32_t sum = 0;
    for (size_t i = 0; i < d.alphabet.size(); ++i) {
        const uint64_t scaled = counts[d.alphabet[i]] * remaining;
        d.frequency[i] = 1 + scaled / symbols.size();
        remainders[i] = scaled % symbols.size(); sum += d.frequency[i];
        d.rank[d.alphabet[i]] = uint32_t(i);
    }
    std::vector<size_t> order(d.alphabet.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return remainders[a] != remainders[b] ? remainders[a] > remainders[b] : a < b;
    });
    for (uint32_t i = 0; i < total - sum; ++i) ++d.frequency[order[i]];
    d.cumulative.push_back(0);
    for (auto f : d.frequency) d.cumulative.push_back(d.cumulative.back() + f);
    return d;
}
void save_distribution(Bytes &out, const Distribution &d) {
    put<uint32_t>(out, d.precision); put<uint32_t>(out, uint32_t(d.alphabet.size()));
    for (size_t i = 0; i < d.alphabet.size(); ++i) {
        put<uint16_t>(out, d.alphabet[i]); put<uint32_t>(out, d.frequency[i]);
    }
}
Distribution load_distribution(Reader &r, bool arithmetic) {
    Distribution d; d.precision = r.get<uint32_t>(); const auto n = r.get<uint32_t>();
    if (!n || n > 65536 || (arithmetic ? d.precision != 24 : d.precision < 12 || d.precision > 16))
        throw std::runtime_error("Invalid native distribution");
    d.cumulative.push_back(0);
    for (uint32_t i = 0; i < n; ++i) {
        auto s = r.get<uint16_t>(); auto f = r.get<uint32_t>();
        if ((i && s <= d.alphabet.back()) || !f || f > (1u << d.precision) - d.cumulative.back())
            throw std::runtime_error("Invalid native alphabet/frequency");
        d.alphabet.push_back(s); d.frequency.push_back(f); d.cumulative.push_back(d.cumulative.back() + f);
    }
    if (d.cumulative.back() != (1u << d.precision)) throw std::runtime_error("Invalid CDF total");
    return d;
}

Stream encode_stream(unsigned codec, const Symbols &symbols, unsigned alphabet = 0) {
    if (codec == 4) return ppm_codec::encode(symbols, alphabet);
    if (codec == 5) return ctw_codec::encode(symbols, alphabet);
    if (codec == 6) return a2_tree::encode<context_models::Paq>(symbols, alphabet);
    if (codec == 7) return a2_tree::encode<context_models::Cabac>(symbols, alphabet);
    if (codec == 8) return a2_tree::encode<neural_codec::Model>(symbols, alphabet);
    if (codec == 9) return cavlc_codec::encode(symbols, alphabet);
    Stream result{symbols.size(), {}, {}};
    if (codec == 3) {
        // Exact Rice length: sum((symbol >> k) + 1 + k). Pairwise histogram
        // folding evaluates every k in O(n + alphabet), with smaller-k tie break.
        // This only selects k; symbol values and their sequence are unchanged.
        auto maximum = *std::max_element(symbols.begin(), symbols.end());
        std::vector<uint64_t> histogram(size_t(maximum) + 1, 0);
        uint64_t quotients = 0;
        for (auto s : symbols) { ++histogram[s]; quotients += s; }
        unsigned best_k = 0;
        uint64_t best_bits = UINT64_MAX;
        for (unsigned k = 0; k <= 16; ++k) {
            uint64_t bits = quotients + symbols.size() * uint64_t(k + 1);
            if (bits < best_bits) { best_bits = bits; best_k = k; }
            quotients = 0;
            const auto n = histogram.size();
            for (size_t i = 0; i < n; i += 2) {
                auto count = histogram[i] + (i + 1 < n ? histogram[i + 1] : 0);
                histogram[i / 2] = count;
                quotients += (i / 2) * count;
            }
            histogram.resize((n + 1) / 2);
        }
        put<uint8_t>(result.model, best_k);
        put<uint64_t>(result.model, best_bits);
        result.data = flac_rice::encode(symbols, best_k, best_bits);
        return result;
    }
    if (codec == 0) {
        SZ3::HuffmanEncoder<uint16_t> encoder;
        encoder.preprocess_encode(symbols, 0);
        result.model.resize(encoder.size_est());
        auto *m = result.model.data(); encoder.save(m); result.model.resize(m - result.model.data());
        result.data.resize(symbols.size() * 4 + 16);
        auto *p = result.data.data(); encoder.encode(symbols, p); result.data.resize(p - result.data.data());
        return result;
    }
    auto d = distribution(symbols, codec == 1);
    save_distribution(result.model, d);
    if (codec == 1) {
        yaecl::ArithmeticCodingEncoder<uint64_t, uint32_t> encoder(32);
        encoder.bit_stream.reserve(symbols.size() * 2 + 16);
        for (auto s : symbols) encoder.encode(d.rank[s], d.cumulative.data(), d.precision);
        encoder.flush();
        put<uint64_t>(result.model, encoder.bit_stream.size());
        result.data = encoder.bit_stream.bytes();
    } else {
        std::vector<RansEncSymbol> entries(d.alphabet.size());
        for (size_t i = 0; i < entries.size(); ++i)
            RansEncSymbolInit(&entries[i], d.cumulative[i], d.frequency[i], d.precision);
        result.data.resize(symbols.size() * 2 + 16);
        auto *end = result.data.data() + result.data.size(); auto *p = end;
        RansState state; RansEncInit(&state);
        for (auto it = symbols.rbegin(); it != symbols.rend(); ++it)
            RansEncPutSymbol(&state, &p, &entries[d.rank[*it]]);
        RansEncFlush(&state, &p);
        result.data = Bytes(p, end);
    }
    return result;
}
Symbols decode_stream(unsigned codec, const Stream &stream) {
    if (codec == 4) return ppm_codec::decode(stream);
    if (codec == 5) return ctw_codec::decode(stream);
    if (codec == 6) return a2_tree::decode<context_models::Paq>(stream);
    if (codec == 7) return a2_tree::decode<context_models::Cabac>(stream);
    if (codec == 8) return a2_tree::decode<neural_codec::Model>(stream);
    if (codec == 9) return cavlc_codec::decode(stream);
    if (codec == 3) {
        Reader reader{stream.model.data(), stream.model.size()};
        auto k = reader.get<uint8_t>();
        auto bits = reader.get<uint64_t>();
        if (reader.position != reader.size) throw std::runtime_error("Trailing Rice model bytes");
        return flac_rice::decode(stream.data, bits, stream.count, k);
    }
    if (codec == 0) {
        SZ3::HuffmanEncoder<uint16_t> decoder;
        auto *m = stream.model.data(); size_t remaining = stream.model.size(); decoder.load(m, remaining);
        if (remaining) throw std::runtime_error("Trailing Huffman table bytes");
        auto *p = stream.data.data(); remaining = stream.data.size();
        auto result = decoder.decode(p, stream.count, remaining);
        if (remaining) throw std::runtime_error("Trailing Huffman payload bytes");
        return result;
    }
    Reader reader{stream.model.data(), stream.model.size()};
    auto d = load_distribution(reader, codec == 1);
    Symbols result; result.reserve(stream.count);
    if (codec == 1) {
        const uint64_t bits = reader.get<uint64_t>();
        if (bits > INT32_MAX || (bits + 7) / 8 != stream.data.size()) throw std::runtime_error("Invalid AC bit count");
        yaecl::BitStream input; input.assign(stream.data.data(), stream.data.size(), int(bits));
        yaecl::ArithmeticCodingDecoder<uint64_t, uint32_t> decoder(32, input);
        for (uint64_t i = 0; i < stream.count; ++i)
            result.push_back(d.alphabet[decoder.decode(int(d.alphabet.size()), d.cumulative.data(), d.precision)]);
    } else {
        if (stream.data.size() < 4) throw std::runtime_error("Truncated rANS initial state");
        Reader payload{stream.data.data(), stream.data.size()};
        uint32_t state = payload.get<uint32_t>();
        if (state < RANS_BYTE_L || state >= uint64_t(RANS_BYTE_L) * 256) throw std::runtime_error("Invalid rANS state");
        std::vector<uint16_t> inverse(1u << d.precision);
        for (size_t i = 0; i < d.alphabet.size(); ++i)
            std::fill(inverse.begin() + d.cumulative[i], inverse.begin() + d.cumulative[i + 1], uint16_t(i));
        for (uint64_t j = 0; j < stream.count; ++j) {
            const auto slot = RansDecGet(&state, d.precision); const auto i = inverse[slot];
            result.push_back(d.alphabet[i]);
            RansDecAdvanceStep(&state, d.cumulative[i], d.frequency[i], d.precision);
            while (state < RANS_BYTE_L) state = (state << 8) | payload.get<uint8_t>();
        }
        if (payload.position != payload.size || state != RANS_BYTE_L) throw std::runtime_error("Invalid rANS ending");
    }
    if (reader.position != reader.size) throw std::runtime_error("Trailing CDF bytes");
    return result;
}

size_t symbol_count(unsigned fmt, size_t bytes) {
    if (!bytes || bytes > MAX_RAW || fmt > 6 || (fmt == 5 && bytes % 2) || (fmt == 6 && bytes % 4))
        throw std::runtime_error("Unsupported native format/length (max 16 MiB per component)");
    return fmt < 2 ? bytes * 2 : fmt == 5 ? bytes / 2 : bytes;
}
std::pair<unsigned, unsigned> fields(unsigned fmt) {
    if (fmt == 2) return {4, 3};
    if (fmt == 3) return {5, 2};
    if (fmt == 5) return {8, 7};
    return {0, 0};
}
Bytes encode_component(unsigned codec, unsigned fmt, bool split, const uint8_t *raw, size_t size) {
    if (codec > 9) throw std::runtime_error("Unknown native codec");
    if (codec >= 4 && split) throw std::runtime_error("A2 retains complete symbols without exponent splitting");
    const auto count = symbol_count(fmt, size);
    Symbols whole(count);
    for (size_t i = 0; i < count; ++i) {
        if (fmt < 2) whole[i] = (raw[i / 2] >> ((i % 2 == (fmt == 0 ? 0 : 1)) ? 0 : 4)) & 15;
        else if (fmt == 5) whole[i] = uint16_t(raw[2 * i]) | (uint16_t(raw[2 * i + 1]) << 8);
        else whole[i] = raw[i];
    }
    auto [exponent, mantissa] = fields(fmt);
    split = split && exponent;
    std::vector<Symbols> parts;
    if (split) {
        parts.resize(2); parts[0].resize(count); parts[1].resize(count);
        for (size_t i = 0; i < count; ++i) {
            parts[0][i] = (whole[i] >> mantissa) & ((1u << exponent) - 1);
            parts[1][i] = ((whole[i] >> (exponent + mantissa)) << mantissa) | (whole[i] & ((1u << mantissa) - 1));
        }
    } else parts.push_back(std::move(whole));
    Bytes frame; put<uint32_t>(frame, MAGIC); put<uint64_t>(frame, size);
    put<uint32_t>(frame, codec); put<uint32_t>(frame, fmt); put<uint32_t>(frame, split);
    put<uint32_t>(frame, parts.size());
    for (size_t i = 0; i < parts.size(); ++i) {
        unsigned width = split ? (i == 0 ? exponent : mantissa + 1) : fmt < 2 ? 4 : fmt == 5 ? 16 : 8;
        auto stream = encode_stream(codec, parts[i], 1u << width);
        put<uint64_t>(frame, stream.count); put<uint64_t>(frame, stream.model.size()); put<uint64_t>(frame, stream.data.size());
        frame.insert(frame.end(), stream.model.begin(), stream.model.end());
        frame.insert(frame.end(), stream.data.begin(), stream.data.end());
    }
    return frame;
}
Bytes decode_component(const uint8_t *frame, size_t size) {
    Reader reader{frame, size};
    if (reader.get<uint32_t>() != MAGIC) throw std::runtime_error("Unknown native frame");
    const auto raw_size = reader.get<uint64_t>();
    const auto codec = reader.get<uint32_t>(), fmt = reader.get<uint32_t>(), split = reader.get<uint32_t>();
    if (codec > 9 || split > 1) throw std::runtime_error("Invalid native codec/flags");
    if (codec >= 4 && split) throw std::runtime_error("A2 retains complete symbols without exponent splitting");
    const auto count = symbol_count(fmt, raw_size);
    auto [exponent, mantissa] = fields(fmt);
    const auto streams = reader.get<uint32_t>();
    if ((split && !exponent) || streams != (split ? 2u : 1u)) throw std::runtime_error("Invalid native field layout");
    std::vector<Symbols> parts;
    for (unsigned i = 0; i < streams; ++i) {
        Stream stream; stream.count = reader.get<uint64_t>();
        const auto model_size = reader.get<uint64_t>(), data_size = reader.get<uint64_t>();
        if (stream.count != count) throw std::runtime_error("Invalid native symbol count");
        auto *p = reader.take(model_size); stream.model.assign(p, p + model_size);
        p = reader.take(data_size); stream.data.assign(p, p + data_size);
        if (codec >= 4) {
            Reader configuration{stream.model.data(), stream.model.size()};
            configuration.get<uint32_t>(); // version is checked by the method.
            unsigned width = fmt < 2 ? 4 : fmt == 5 ? 16 : 8;
            if (configuration.get<uint32_t>() != (1u << width))
                throw std::runtime_error("A2 alphabet does not match complete native codewords");
        }
        parts.push_back(decode_stream(codec, stream));
    }
    if (reader.position != reader.size) throw std::runtime_error("Trailing native frame bytes");
    Bytes raw(raw_size, 0);
    for (size_t i = 0; i < count; ++i) {
        uint32_t value = parts[0][i];
        if (split) {
            const uint32_t remainder = parts[1][i];
            if (value >= (1u << exponent) || remainder >= (1u << (mantissa + 1))) throw std::runtime_error("Invalid field value");
            value = ((remainder >> mantissa) << (exponent + mantissa)) | (value << mantissa) | (remainder & ((1u << mantissa) - 1));
        }
        if (fmt < 2) {
            if (value > 15) throw std::runtime_error("Invalid FP4 codeword");
            raw[i / 2] |= uint8_t(value << ((i % 2 == (fmt == 0 ? 0 : 1)) ? 0 : 4));
        } else if (fmt == 5) { raw[2 * i] = uint8_t(value); raw[2 * i + 1] = uint8_t(value >> 8); }
        else {
            if (value > 255) throw std::runtime_error("Invalid byte symbol");
            raw[i] = uint8_t(value);
        }
    }
    return raw;
}

#define API extern "C" __attribute__((visibility("default")))
static int copy_result(const Bytes &data, uint8_t **out, size_t *length) {
    *out = static_cast<uint8_t *>(std::malloc(data.size()));
    if (!*out) throw std::bad_alloc();
    std::memcpy(*out, data.data(), data.size()); *length = data.size(); return 0;
}
API int fp4_native_encode(unsigned codec, unsigned fmt, unsigned split, const uint8_t *data, size_t size,
                          uint8_t **out, size_t *length) {
    try { return copy_result(encode_component(codec, fmt, split, data, size), out, length); }
    catch (const std::exception &e) { error_message = e.what(); return -1; }
}
API int fp4_native_decode(const uint8_t *frame, size_t size, uint8_t **out, size_t *length) {
    try { return copy_result(decode_component(frame, size), out, length); }
    catch (const std::exception &e) { error_message = e.what(); return -1; }
}
API const char *fp4_native_error() { return error_message.c_str(); }
API void fp4_native_free(void *p) { std::free(p); }
