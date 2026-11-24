/* Spec 015 (ANS, R1 + R2): static-table rANS for one MXFP4 tensor.
 *
 * Coder: ryg_rans `rans_byte.h` (public domain, native/third_party/ryg_rans, unmodified): 32-bit state,
 * normalisation interval [2^23, 2^31), byte-wise renormalisation, 4-byte flush per stream; probabilities
 * quantised to SCALE_BITS = 12 (frequencies sum to 4096).
 *
 * Symbols. Scale stream: the E8M0 byte of every 32-code block (256-symbol table). Code stream: the 16 FP4
 * codes with -0 (code 8) mapped to +0 (value-exact), i.e. 15 symbols m = c (c < 8), c - 1 (c > 8); each code
 * uses the table of its block's scale value. Streams are cut into independent chunks (CHUNK codes / CHUNK
 * scales each); each chunk is a separate rANS stream with its own flush.
 *
 * All tables are static (built before coding, never updated). Tables are given as frequencies `freq`
 * (0 = absent) and cumulative starts `cum`; code tables are [256][15] (only rows of scale values present).
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "../../native/third_party/ryg_rans/rans_byte.h"

#define SCALE_BITS 12
#define M (1u << SCALE_BITS)
#define NS 15

static inline int merged(uint8_t c) { return c == 8 ? 0 : (c > 8 ? c - 1 : c); }

/* histogram of merged codes per scale value, and of scale bytes */
void count_tensor(const uint8_t *codes, const uint8_t *scale, int64_t n_codes, int64_t *hist, int64_t *shist) {
    int64_t nb = n_codes / 32;
    for (int64_t b = 0; b < nb; b++) {
        uint8_t s = scale[b];
        shist[s]++;
        int64_t *h = hist + (int64_t)s * NS;
        const uint8_t *c = codes + b * 32;
        for (int j = 0; j < 32; j++) h[merged(c[j])]++;
    }
}

/* Encode one stream of n symbols (given by callbacks through arrays) backwards into tmp, copy to out. */
static int64_t put_stream_scale(const uint8_t *sym, int64_t n, const RansEncSymbol *es, uint8_t *tmp, int64_t cap,
                                uint8_t *out) {
    uint8_t *ptr = tmp + cap;
    RansState r;
    RansEncInit(&r);
    for (int64_t i = n - 1; i >= 0; i--) RansEncPutSymbol(&r, &ptr, &es[sym[i]]);
    RansEncFlush(&r, &ptr);
    int64_t len = (tmp + cap) - ptr;
    memcpy(out, ptr, (size_t)len);
    return len;
}

static int64_t put_stream_codes(const uint8_t *codes, const uint8_t *scale, int64_t i0, int64_t n,
                                const RansEncSymbol *ec /*[256][NS]*/, uint8_t *tmp, int64_t cap, uint8_t *out) {
    uint8_t *ptr = tmp + cap;
    RansState r;
    RansEncInit(&r);
    for (int64_t i = i0 + n - 1; i >= i0; i--)
        RansEncPutSymbol(&r, &ptr, &ec[(int64_t)scale[i >> 5] * NS + merged(codes[i])]);
    RansEncFlush(&r, &ptr);
    int64_t len = (tmp + cap) - ptr;
    memcpy(out, ptr, (size_t)len);
    return len;
}

/* Returns total bytes written to out (scale streams first, then code streams); stream_len gets each length;
   -1 if a symbol has zero frequency (table does not cover the data). */
int64_t encode_tensor(const uint8_t *codes, const uint8_t *scale, int64_t n_codes, int64_t chunk,
                      const uint32_t *cfreq, const uint32_t *ccum, const uint32_t *sfreq, const uint32_t *scum,
                      uint8_t *out, int64_t *stream_len) {
    int64_t nb = n_codes / 32;
    static __thread RansEncSymbol es[256], ec[256 * NS];
    for (int s = 0; s < 256; s++)
        if (sfreq[s]) RansEncSymbolInit(&es[s], scum[s], sfreq[s], SCALE_BITS);
    for (int64_t k = 0; k < 256 * NS; k++)
        if (cfreq[k]) RansEncSymbolInit(&ec[k], ccum[k], cfreq[k], SCALE_BITS);
    /* coverage check */
    for (int64_t b = 0; b < nb; b++) if (!sfreq[scale[b]]) return -1;
    for (int64_t i = 0; i < n_codes; i++) if (!cfreq[(int64_t)scale[i >> 5] * NS + merged(codes[i])]) return -1;
    int64_t cap = 2 * chunk + 64;
    uint8_t *tmp = (uint8_t *)malloc((size_t)cap);
    int64_t pos = 0, k = 0;
    for (int64_t b0 = 0; b0 < nb; b0 += chunk) {
        int64_t n = nb - b0 < chunk ? nb - b0 : chunk;
        stream_len[k] = put_stream_scale(scale + b0, n, es, tmp, cap, out + pos);
        pos += stream_len[k++];
    }
    for (int64_t i0 = 0; i0 < n_codes; i0 += chunk) {
        int64_t n = n_codes - i0 < chunk ? n_codes - i0 : chunk;
        stream_len[k] = put_stream_codes(codes, scale, i0, n, ec, tmp, cap, out + pos);
        pos += stream_len[k++];
    }
    free(tmp);
    return pos;
}

/* Decode all streams and compare with the original: mism[0] = scale bytes that differ,
   mism[1] = codes that differ after the +-0 merge. Decoded data are written to dec_scale / dec_sym. */
void decode_verify(const uint8_t *in, const int64_t *stream_len, int64_t n_codes, int64_t chunk,
                   const uint32_t *cfreq, const uint32_t *ccum, const uint32_t *sfreq, const uint32_t *scum,
                   const uint8_t *codes, const uint8_t *scale, uint8_t *dec_scale, uint8_t *dec_sym, int64_t *mism) {
    int64_t nb = n_codes / 32;
    static __thread uint8_t s_c2s[M];
    static __thread uint8_t c_c2s[256][M];
    static __thread RansDecSymbol ds[256], dc[256 * NS];
    for (int s = 0; s < 256; s++)
        if (sfreq[s]) {
            RansDecSymbolInit(&ds[s], scum[s], sfreq[s]);
            for (uint32_t j = 0; j < sfreq[s]; j++) s_c2s[scum[s] + j] = (uint8_t)s;
        }
    for (int s = 0; s < 256; s++) {
        if (!sfreq[s]) continue;
        for (int m = 0; m < NS; m++) {
            int64_t k = (int64_t)s * NS + m;
            if (!cfreq[k]) continue;
            RansDecSymbolInit(&dc[k], ccum[k], cfreq[k]);
            for (uint32_t j = 0; j < cfreq[k]; j++) c_c2s[s][ccum[k] + j] = (uint8_t)m;
        }
    }
    const uint8_t *p = in;
    int64_t k = 0;
    for (int64_t b0 = 0; b0 < nb; b0 += chunk) {
        int64_t n = nb - b0 < chunk ? nb - b0 : chunk;
        uint8_t *ptr = (uint8_t *)p;
        RansState r;
        RansDecInit(&r, &ptr);
        for (int64_t i = 0; i < n; i++) {
            uint8_t s = s_c2s[RansDecGet(&r, SCALE_BITS)];
            dec_scale[b0 + i] = s;
            RansDecAdvanceSymbol(&r, &ptr, &ds[s], SCALE_BITS);
        }
        p += stream_len[k++];
    }
    for (int64_t i0 = 0; i0 < n_codes; i0 += chunk) {
        int64_t n = n_codes - i0 < chunk ? n_codes - i0 : chunk;
        uint8_t *ptr = (uint8_t *)p;
        RansState r;
        RansDecInit(&r, &ptr);
        for (int64_t i = i0; i < i0 + n; i++) {
            uint8_t s = dec_scale[i >> 5];
            uint8_t m = c_c2s[s][RansDecGet(&r, SCALE_BITS)];
            dec_sym[i] = m;
            RansDecAdvanceSymbol(&r, &ptr, &dc[(int64_t)s * NS + m], SCALE_BITS);
        }
        p += stream_len[k++];
    }
    int64_t ms = 0, mc = 0;
    for (int64_t b = 0; b < nb; b++) ms += dec_scale[b] != scale[b];
    for (int64_t i = 0; i < n_codes; i++) mc += dec_sym[i] != merged(codes[i]);
    mism[0] = ms;
    mism[1] = mc;
}
