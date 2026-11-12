/*
 * PPM model adapted from Project Nayuki's Reference arithmetic coding
 * Copyright (c) Project Nayuki
 * MIT License. See Readme.markdown.
 * Upstream ab6ee50afec04d235a4b82d17f407f0fd2b42e9a, PpmModel.cpp/.hpp,
 * PpmCompress.cpp and PpmDecompress.cpp.
 * Retained: order-3 suffix contexts, escape frequency 1, no exclusion,
 * uniform order-minus-one, increment all suffix contexts after each symbol.
 * Changes: sparse children/frequencies, Fenwick root, deterministic bounded
 * reset/rescale; complete uint16 symbols, known length instead of EOF.
 */
#pragma once
#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace nayuki_ppm {
constexpr uint32_t ORDER = 3, MAX_CONTEXTS = 65536, RESCALE = 1u << 20;
constexpr uint32_t CDF_BITS = 24, CDF_TOTAL = 1u << CDF_BITS;
struct Interval { uint32_t symbol, low, high; };

// Same integer counts as Nayuki's SimpleFrequencyTable, stored sparsely in
// suffix contexts and with a Fenwick index at the large order-0 alphabet.
class Frequencies {
    bool dense;
    uint32_t total = 1;
    std::vector<uint32_t> counts, tree;
    std::vector<std::pair<uint32_t, uint32_t>> entries;
    void add_tree(uint32_t symbol, uint32_t delta) {
        for (size_t i = size_t(symbol) + 1; i < tree.size(); i += i & -i) tree[i] += delta;
    }
    uint32_t prefix(uint32_t symbol) const {
        uint32_t sum = 0;
        for (size_t i = symbol; i; i -= i & -i) sum += tree[i];
        return sum;
    }
    void rescale() {
        total = 0;
        if (dense) {
            tree.assign(counts.size() + 1, 0);
            for (size_t i = 0; i < counts.size(); ++i) {
                counts[i] = (counts[i] + 1) / 2;
                total += counts[i]; tree[i + 1] += counts[i];
                size_t parent = i + 1 + ((i + 1) & -(i + 1));
                if (parent < tree.size()) tree[parent] += tree[i + 1];
            }
        } else for (auto &entry : entries) { entry.second = (entry.second + 1) / 2; total += entry.second; }
    }
    Interval normalized(uint32_t symbol, uint32_t low, uint32_t freq) const {
        return {symbol, uint32_t(uint64_t(low) * CDF_TOTAL / total),
                        uint32_t(uint64_t(low + freq) * CDF_TOTAL / total)};
    }
public:
    explicit Frequencies(uint32_t alphabet, bool is_root) : dense(is_root) {
        if (dense) { counts.resize(alphabet + 1); tree.resize(alphabet + 2); counts[alphabet] = 1; add_tree(alphabet, 1); }
        else entries.push_back({alphabet, 1}); // escape
    }
    // Raw-count access for the m-ary CTW adapter (escape is excluded).
    uint32_t observations() const { return total - 1; }
    uint32_t before(uint32_t symbol) const {
        if (dense) return prefix(symbol);
        uint32_t sum = 0;
        for (const auto &entry : entries) { if (entry.first >= symbol) break; sum += entry.second; }
        return sum;
    }
    Interval interval(uint32_t symbol) const {
        if (dense) return normalized(symbol, prefix(symbol), counts.at(symbol));
        uint32_t sum = 0;
        for (const auto &entry : entries) {
            if (entry.first == symbol) return normalized(symbol, sum, entry.second);
            if (entry.first > symbol) break;
            sum += entry.second;
        }
        return {symbol, 0, 0};
    }
    Interval select(uint32_t scaled) const {
        uint32_t target = uint32_t(((uint64_t(scaled) + 1) * total - 1) / CDF_TOTAL);
        if (target >= total) throw std::runtime_error("Invalid PPM CDF value");
        if (dense) {
            uint32_t index = 0, sum = 0, step = 1;
            while (step < counts.size()) step <<= 1;
            for (; step; step >>= 1) {
                uint32_t next = index + step;
                if (next < tree.size() && sum + tree[next] <= target) { index = next; sum += tree[next]; }
            }
            if (index >= counts.size() || !counts[index]) throw std::runtime_error("Invalid PPM symbol");
            return normalized(index, sum, counts[index]);
        }
        uint32_t sum = 0;
        for (const auto &entry : entries) {
            if (target < sum + entry.second) return normalized(entry.first, sum, entry.second);
            sum += entry.second;
        }
        throw std::runtime_error("Invalid PPM sparse CDF");
    }
    void increment(uint32_t symbol) {
        if (total >= RESCALE) rescale();
        if (dense) { ++counts.at(symbol); add_tree(symbol, 1); }
        else {
            auto it = std::lower_bound(entries.begin(), entries.end(), symbol,
                [](const auto &entry, uint32_t s) { return entry.first < s; });
            if (it == entries.end() || it->first != symbol) entries.insert(it, {symbol, 1});
            else ++it->second;
        }
        ++total;
    }
};

class PpmModel {
public:
    struct Context {
        Frequencies frequencies;
        std::unordered_map<uint32_t, std::unique_ptr<Context>> subcontexts;
        Context(uint32_t alphabet, bool root) : frequencies(alphabet, root) {}
    };
private:
    uint32_t alphabet, context_count = 1;
    std::unique_ptr<Context> root;
    std::vector<uint32_t> history;
public:
    explicit PpmModel(uint32_t symbols) : alphabet(symbols), root(new Context(symbols, true)) {}
    std::array<Context *, ORDER + 1> contexts() const {
        std::array<Context *, ORDER + 1> result{};
        result[0] = root.get();
        size_t next = 1;
        Context *ctx = root.get();
        for (uint32_t sym : history) {
            auto it = ctx->subcontexts.find(sym);
            if (it == ctx->subcontexts.end()) break;
            ctx = it->second.get(); result[next++] = ctx;
        }
        return result;
    }
    void incrementContexts(uint32_t symbol) {
        // Adapted from PpmModel::incrementContexts; reset only when the next
        // update would exceed the context budget. Encoder/decoder do this after
        // the same decoded symbol, then both clear their history identically.
        Context *ctx = root.get();
        size_t existing = 0;
        for (; existing < history.size(); ++existing) {
            auto it = ctx->subcontexts.find(history[existing]);
            if (it == ctx->subcontexts.end()) break;
            ctx = it->second.get();
        }
        if (context_count + history.size() - existing > MAX_CONTEXTS) {
            root.reset(new Context(alphabet, true)); history.clear(); context_count = 1;
        }
        ctx = root.get();
        ctx->frequencies.increment(symbol);
        for (uint32_t sym : history) {
            auto &subctx = ctx->subcontexts[sym];
            if (!subctx) { subctx.reset(new Context(alphabet, false)); ++context_count; }
            ctx = subctx.get(); ctx->frequencies.increment(symbol);
        }
        if (history.size() == ORDER) history.pop_back();
        history.insert(history.begin(), symbol);
    }
};
} // namespace nayuki_ppm
