/* Adaptive context rANS coder for S3-b (15 symbols).
 *
 * Model: for every context a Dirichlet(alpha = 1/2) adaptive count (the 008 / txcore.dirichlet model):
 *   p(s | ctx) = (c_s + 1/2) / (n + 15/2) = (2 c_s + 1) / (2 n + 15).
 * The probabilities are quantised to PREC bits before coding: f_s = 1 + floor((2 c_s + 1)(M - 15) / (2 n + 15)),
 * the remainder going to the most probable symbol, so every f_s >= 1 and sum f_s = M = 2^PREC.
 * Coder: 64-bit rANS with 32-bit renormalisation (as ryg_rans rans64.h). The encoder computes all (start, freq)
 * forward, then encodes backwards; the decoder runs forward and updates the counts as it goes.
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define NS 15
#define PREC 16
#define M (1u << PREC)
#define RANS_L (1ull << 31)

static void table(const uint32_t *cnt, uint32_t n, uint32_t *f) {
    uint64_t tot = 2ull * n + NS;
    uint32_t sum = 0, best = 0, bi = 0;
    for (int s = 0; s < NS; s++) {
        f[s] = 1 + (uint32_t)(((2ull * cnt[s] + 1) * (uint64_t)(M - NS)) / tot);
        sum += f[s];
        if (cnt[s] > best) { best = cnt[s]; bi = s; }
    }
    f[bi] += M - sum;
}

/* returns number of 32-bit words written to out (out must hold at least n + 2 words) */
int64_t rac_encode(const int32_t *sym, const int32_t *ctx, int64_t n, int32_t nctx, uint32_t *out) {
    uint32_t *cnt = calloc((size_t)nctx * NS, 4), *tot = calloc((size_t)nctx, 4);
    uint32_t *st = malloc((size_t)n * 4), *fr = malloc((size_t)n * 4);
    uint32_t f[NS];
    for (int64_t i = 0; i < n; i++) {
        int32_t c = ctx[i];
        table(cnt + (size_t)c * NS, tot[c], f);
        uint32_t start = 0;
        for (int s = 0; s < sym[i]; s++) start += f[s];
        st[i] = start; fr[i] = f[sym[i]];
        cnt[(size_t)c * NS + sym[i]]++; tot[c]++;
    }
    /* encode backwards; words are emitted into a stack that is reversed at the end */
    uint64_t x = RANS_L;
    int64_t w = 0;
    for (int64_t i = n - 1; i >= 0; i--) {
        uint64_t xmax = ((RANS_L >> PREC) << 32) * fr[i];
        if (x >= xmax) { out[w++] = (uint32_t)x; x >>= 32; }
        x = ((x / fr[i]) << PREC) + (x % fr[i]) + st[i];
    }
    out[w++] = (uint32_t)x; out[w++] = (uint32_t)(x >> 32);
    for (int64_t a = 0, b = w - 1; a < b; a++, b--) { uint32_t t = out[a]; out[a] = out[b]; out[b] = t; }
    free(cnt); free(tot); free(st); free(fr);
    return w;
}

int rac_decode(const uint32_t *in, int64_t nw, const int32_t *ctx, int64_t n, int32_t nctx, int32_t *sym) {
    uint32_t *cnt = calloc((size_t)nctx * NS, 4), *tot = calloc((size_t)nctx, 4);
    uint32_t f[NS];
    int64_t r = 0;
    uint64_t x = ((uint64_t)in[0] << 32) | in[1];
    r = 2;
    for (int64_t i = 0; i < n; i++) {
        int32_t c = ctx[i];
        table(cnt + (size_t)c * NS, tot[c], f);
        uint32_t slot = (uint32_t)(x & (M - 1)), start = 0;
        int s = 0;
        while (start + f[s] <= slot) { start += f[s]; s++; }
        sym[i] = s;
        x = f[s] * (x >> PREC) + slot - start;
        if (x < RANS_L) { if (r >= nw) { free(cnt); free(tot); return -1; } x = (x << 32) | in[r++]; }
        cnt[(size_t)c * NS + s]++; tot[c]++;
    }
    free(cnt); free(tot);
    return (r == nw && x == RANS_L) ? 0 : -2;
}
