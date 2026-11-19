/* Spec 011 B4: context mixing (lpaq-style) for FP4 codes -- encoder, decoder and losslessness check.
 *
 *   cm <R> <W> <sym.bin> <scale.bin> <partner.bin|-> <partner_kind 0|1|2> <out.json>
 *
 * sym.bin: R*W bytes, merged symbols 0..14 (0 = zero, 1..7 = +0.5..+6, 8..14 = -0.5..-6), row-major.
 * scale.bin: R*(W/32) bytes. partner.bin: the earlier-decoded matrix (kind 1: same shape, (r,c);
 * kind 2: transposed, (c,r)); kind 0 = none.
 * A symbol is coded as binary decisions: zero?; sign; magnitude 1..7 as 3 bits (binary tree, 7 leaves).
 * Each decision is predicted by NM context models (12-bit probabilities in hash tables, adaptive rate),
 * mixed in the logistic domain by a mixer whose weight set is chosen by the decision node; then refined by an
 * APM (adaptive probability map) on (node, quantised mixer output). A 32-bit binary arithmetic coder writes the
 * bits; the decoder repeats the same computations and must return the input exactly.
 * Output: ideal code length (sum of -log2 p), actual coded size, encode and decode wall time, check.
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define NM 10
#define TB 22                 /* 2^22 slots per model */
#define NODES 9
#ifndef MIXLR
#define MIXLR 1
#endif
#ifndef MIXSH
#define MIXSH 11
#endif
#ifndef CLIM
#define CLIM 65535
#endif
#ifndef LRDEC
#define LRDEC 0
#endif
#ifndef APM2W
#define APM2W 1
#endif
#ifndef APMW
#define APMW 2
#endif

static int R, W, NB, PK;
static uint8_t *S, *E, *PT;

static int squash_t[4095];
static inline int squash_f(int d) {        /* 1/(1+e^-d), d in (-2047..2047) scaled by 256 -> 12 bit */
    return (int)(4096.0 / (1.0 + exp(-d / 256.0)));
}
static inline int squash(int d) {
    if (d > 2047) d = 2047;
    if (d < -2047) d = -2047;
    return squash_t[d + 2047];
}
static short stretch_t[4096];
static void init_stretch(void) {
    for (int x = -2047; x <= 2047; x++) squash_t[x + 2047] = squash_f(x);
    int pi = 0;
    for (int x = -2047; x <= 2047; x++) {
        int v = squash(x);
        for (int j = pi; j <= v && j < 4096; j++) stretch_t[j] = x;
        pi = v + 1;
    }
    for (int j = pi; j < 4096; j++) stretch_t[j] = 2047;
}

typedef struct { uint16_t p; uint16_t n; } Slot;
static Slot *T[NM];
static int32_t Wt[NODES * 4][NM + 1];    /* mixer weights (16.16), weight set = node x scale bucket */
static uint16_t APM[NODES][33];
static uint16_t APM2[NODES][256][33];     /* second map keyed by (node, scale byte) */
static int st_sc;

static inline uint32_t hsh(uint32_t a, uint32_t b) {
    uint32_t h = a * 0x9E3779B1u ^ (b + 0x7F4A7C15u + (a << 6) + (a >> 2));
    h ^= h >> 15; h *= 0x2C1B3C6Du; h ^= h >> 12;
    return h & ((1u << TB) - 1);
}

typedef struct { uint32_t x1, x2, x; FILE *f; uint8_t *buf; size_t pos, len; } AC;

static double ideal_bits;
static uint64_t n_dec;

/* per-symbol contexts, computed from already decoded data */
static uint32_t ctx[NM];
static void contexts(const uint8_t *dec, int r, int c, int bmax, int bcnt, int msum, int bodd, int bnz) {
    int pos = c & 31;
    int p1 = c > 0 ? dec[(size_t)r * W + c - 1] : 15;
    int p2 = c > 1 ? dec[(size_t)r * W + c - 2] : 15;
    int ab = r > 0 ? dec[(size_t)(r - 1) * W + c] : 15;
    int sc = E[(size_t)r * NB + (c >> 5)];
    int pt = 15;
    if (PK == 1) pt = PT[(size_t)r * W + c];
    else if (PK == 2) pt = PT[(size_t)c * R + r];
    int mag1 = p1 == 15 ? 8 : (p1 == 0 ? 0 : (p1 <= 7 ? p1 : p1 - 7));
    int mean2 = pos ? (2 * msum + pos / 2) / pos : 15;
    ctx[0] = 0;
    ctx[1] = 1 + sc;
    ctx[2] = 300 + p1;
    ctx[3] = 400 + p1 * 16 + p2;
    ctx[4] = 1000 + bmax * 4 + (pos >> 3);
    ctx[5] = 1100 + pt;
    ctx[6] = 1200 + sc * 9 + mag1;
    ctx[7] = 4000 + ab;
    ctx[8] = 4100 + bcnt * 16 + (mean2 > 15 ? 15 : mean2);
    /* A3: about 29% of blocks have no magnitude 0.5 / 1.5 -- odd levels seen so far, non-zero count, max */
    int nzb = bnz == 0 ? 0 : bnz < 2 ? 1 : bnz < 4 ? 2 : bnz < 8 ? 3 : bnz < 16 ? 4 : 5;
    ctx[9] = 6000 + ((bodd > 3 ? 3 : bodd) * 6 + nzb) * 9 + bmax;
}

static int mixw_set(int node) { return node * 4; }

/* predict one binary decision at a node; returns 12-bit p(1); fills state for update */
static int st_idx[NM], st_x[NM + 1], st_set, st_pm, st_apm_i, st_apm_w;
static int predict(int node, int sc) {
    int dot = 0;
    st_set = mixw_set(node) + (sc < 118 ? 0 : sc < 120 ? 1 : sc < 122 ? 2 : 3);
    for (int i = 0; i < NM; i++) {
        uint32_t h = hsh(ctx[i] * 16 + node, i);
        st_idx[i] = h;
        Slot *s = &T[i][h];
        if (s->n == 0 && s->p == 0) s->p = 32768;     /* 16-bit probability */
        st_x[i] = stretch_t[s->p >> 4];
        dot += (int)(((int64_t)st_x[i] * Wt[st_set][i]) >> 16);
    }
    st_x[NM] = 256;
    dot += (int)(((int64_t)256 * Wt[st_set][NM]) >> 16);
    int pm = squash(dot);
    if (pm < 1) pm = 1;
    if (pm > 4095) pm = 4095;
    st_pm = pm;
    int d = stretch_t[pm] + 2048;               /* 0..4095 */
    int lo = d >> 7, w = d & 127;
    st_apm_i = lo; st_apm_w = w;
    int pa = ((APM[node][lo] * (128 - w) + APM[node][lo + 1] * w) >> 7) >> 4;   /* 16 -> 12 bit */
    st_sc = sc;
    int pb = ((APM2[node][sc][lo] * (128 - w) + APM2[node][sc][lo + 1] * w) >> 7) >> 4;
#if APM2W
    pa = (pa + pb) >> 1;
#else
    (void)pb;
#endif
    int p = (pm * (4 - APMW) + pa * APMW) >> 2;
    if (p < 1) p = 1;
    if (p > 4095) p = 4095;
    return p;
}

static void update(int node, int bit, int p) {
    double q = bit ? p / 4096.0 : 1.0 - p / 4096.0;
    ideal_bits += -log2(q);
    int err = ((bit << 12) - st_pm);
    /* learning rate halves each time the number of decisions doubles beyond 2^20 (at most 4 times) */
    n_dec++;
    int extra = 0;
    for (uint64_t t = n_dec >> 20; t && extra < LRDEC; t >>= 1) extra++;
    for (int i = 0; i <= NM; i++) {
        int64_t w = Wt[st_set][i] + ((((int64_t)st_x[i] * err * MIXLR) >> MIXSH) >> extra);
        if (w > (8 << 16)) w = 8 << 16;
        if (w < -(8 << 16)) w = -(8 << 16);
        Wt[st_set][i] = (int32_t)w;
    }
    for (int i = 0; i < NM; i++) {
        Slot *s = &T[i][st_idx[i]];
        int target = bit ? 65535 : 0;
        int rate = s->n + 2;               /* 1/2, 1/3, ... down to 1/256: close to counting */
        int np_ = (int)s->p + (target - (int)s->p) / rate;
        if (np_ < 16) np_ = 16;
        if (np_ > 65519) np_ = 65519;
        s->p = (uint16_t)np_;
        if (s->n < CLIM) s->n++;
    }
    int target = bit ? 65535 : 0;
    APM[node][st_apm_i] += ((target - APM[node][st_apm_i]) * (128 - st_apm_w)) >> 13;
    APM[node][st_apm_i + 1] += ((target - APM[node][st_apm_i + 1]) * st_apm_w) >> 13;
    APM2[node][st_sc][st_apm_i] += ((target - APM2[node][st_sc][st_apm_i]) * (128 - st_apm_w)) >> 13;
    APM2[node][st_sc][st_apm_i + 1] += ((target - APM2[node][st_sc][st_apm_i + 1]) * st_apm_w) >> 13;
}

/* arithmetic coder (carryless, 32 bit) */
static void enc_bit(AC *a, int bit, int p) {
    uint32_t xm = a->x1 + (uint32_t)(((uint64_t)(a->x2 - a->x1) * p) >> 12);
    if (bit) a->x2 = xm; else a->x1 = xm + 1;
    while (((a->x1 ^ a->x2) & 0xff000000u) == 0) {
        a->buf[a->pos++] = a->x2 >> 24;
        a->x1 <<= 8; a->x2 = (a->x2 << 8) | 255;
    }
}
static int dec_bit(AC *a, int p) {
    uint32_t xm = a->x1 + (uint32_t)(((uint64_t)(a->x2 - a->x1) * p) >> 12);
    int bit = a->x <= xm;
    if (bit) a->x2 = xm; else a->x1 = xm + 1;
    while (((a->x1 ^ a->x2) & 0xff000000u) == 0) {
        a->x1 <<= 8; a->x2 = (a->x2 << 8) | 255;
        a->x = (a->x << 8) | (a->pos < a->len ? a->buf[a->pos++] : 0);
    }
    return bit;
}

static void reset_models(void) {
    for (int i = 0; i < NM; i++) memset(T[i], 0, sizeof(Slot) << TB);
    for (int s = 0; s < NODES * 4; s++) for (int i = 0; i <= NM; i++) Wt[s][i] = (1 << 16) / NM;
    n_dec = 0;
    for (int n = 0; n < NODES; n++) for (int j = 0; j < 33; j++) {
        APM[n][j] = (uint16_t)(squash((j - 16) * 128) * 16);
        for (int c = 0; c < 256; c++) APM2[n][c][j] = APM[n][j];
    }
}

/* code one symbol: returns decoded symbol (decode mode) */
static int code_symbol(AC *a, int enc, int sym, int sc) {
    int nz, sgn = 0, mag = 0;
    int p = predict(0, sc);
    if (enc) { nz = sym != 0; enc_bit(a, nz, p); } else nz = dec_bit(a, p);
    update(0, nz, p);
    if (!nz) return 0;
    p = predict(1, sc);
    if (enc) { sgn = sym >= 8; enc_bit(a, sgn, p); } else sgn = dec_bit(a, p);
    update(1, sgn, p);
    int m = enc ? (sym >= 8 ? sym - 7 : sym) - 1 : 0;      /* 0..6 */
    int node = 2, v = 0;
    for (int b = 2; b >= 0; b--) {
        int bit;
        p = predict(node, sc);
        if (enc) { bit = (m >> b) & 1; enc_bit(a, bit, p); } else bit = dec_bit(a, p);
        update(node, bit, p);
        v = v * 2 + bit;
        node = node * 2 + bit - 1;           /* 2 -> 3,4 -> 5..8 ... capped below */
        if (node >= NODES) node = NODES - 1;
    }
    mag = v + 1;
    return sgn ? mag + 7 : mag;
}

static double run(uint8_t *dec, int enc, AC *a) {
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    reset_models();
    for (int r = 0; r < R; r++) {
        for (int b = 0; b < NB; b++) {
            int bmax = 0, bcnt = 0, msum = 0, bodd = 0, bnz = 0;
            for (int j = 0; j < 32; j++) {
                int c = b * 32 + j;
                contexts(dec, r, c, bmax, bcnt, msum, bodd, bnz);
                int sc = E[(size_t)r * NB + b];
                int s = code_symbol(a, enc, enc ? S[(size_t)r * W + c] : 0, sc);
                dec[(size_t)r * W + c] = (uint8_t)s;
                int mg = s == 0 ? 0 : (s <= 7 ? s : s - 7);
                if (mg + 1 > bmax) bmax = mg + 1;
                if (mg >= 6) bcnt = bcnt < 3 ? bcnt + 1 : 3;
                msum += mg;
                if (mg == 1 || mg == 3) bodd++;
                if (mg > 0) bnz++;
            }
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    return (t1.tv_sec - t0.tv_sec) + 1e-9 * (t1.tv_nsec - t0.tv_nsec);
}

static uint8_t *readf(const char *p, size_t n) {
    uint8_t *b = malloc(n);
    FILE *f = fopen(p, "rb");
    if (!f || fread(b, 1, n, f) != n) { fprintf(stderr, "read %s\n", p); exit(1); }
    fclose(f);
    return b;
}

int main(int argc, char **argv) {
    if (argc < 8) { fprintf(stderr, "usage\n"); return 1; }
    R = atoi(argv[1]); W = atoi(argv[2]); NB = W / 32;
    S = readf(argv[3], (size_t)R * W);
    E = readf(argv[4], (size_t)R * NB);
    PK = atoi(argv[6]);
    if (PK) PT = readf(argv[5], (size_t)R * W);
    init_stretch();
    for (int i = 0; i < NM; i++) T[i] = malloc(sizeof(Slot) << TB);
    size_t cap = (size_t)R * W + 1024;
    AC a = {0, 0xffffffffu, 0, NULL, malloc(cap), 0, 0};
    uint8_t *dec = malloc((size_t)R * W);
    ideal_bits = 0;
    double te = run(dec, 1, &a);
    a.buf[a.pos++] = a.x1 >> 24; a.buf[a.pos++] = 255; a.buf[a.pos++] = 255; a.buf[a.pos++] = 255;
    size_t bytes = a.pos;
    double ideal = ideal_bits;
    AC d = {0, 0xffffffffu, 0, NULL, a.buf, 0, bytes};
    for (int i = 0; i < 4; i++) d.x = (d.x << 8) | d.buf[d.pos++];
    uint8_t *dec2 = malloc((size_t)R * W);
    double td = run(dec2, 0, &d);
    int ok = memcmp(dec2, S, (size_t)R * W) == 0;
    FILE *o = fopen(argv[7], "w");
    fprintf(o, "{\"n\": %zu, \"ideal_bits\": %.3f, \"coded_bytes\": %zu, \"encode_s\": %.4f, \"decode_s\": %.4f, \"lossless\": %s}\n",
            (size_t)R * W, ideal, bytes, te, td, ok ? "true" : "false");
    fclose(o);
    return ok ? 0 : 2;
}
