/* TX stage 2 helper: inverse cyclic BWT and move-to-front (sequential parts), called through ctypes.
   build: gcc -O2 -shared -fPIC -o data/010tx/bin/libbwt.so experiments/010-transforms/tx/bwt_helper.c */
#include <stdint.h>
#include <stdlib.h>

/* inverse of the cyclic BWT: L = last column (symbols < K), primary = row of the original rotation. */
void ibwt(const int32_t *L, int64_t n, int64_t primary, int32_t K, int32_t *out) {
    int64_t *cnt = calloc(K + 1, sizeof(int64_t));
    int64_t *T = malloc(n * sizeof(int64_t));
    int64_t *occ = calloc(K, sizeof(int64_t));
    for (int64_t i = 0; i < n; i++) cnt[L[i] + 1]++;
    for (int32_t k = 0; k < K; k++) cnt[k + 1] += cnt[k];
    for (int64_t i = 0; i < n; i++) { T[i] = cnt[L[i]] + occ[L[i]]; occ[L[i]]++; }
    int64_t j = primary;
    for (int64_t i = n - 1; i >= 0; i--) { out[i] = L[j]; j = T[j]; }
    free(cnt); free(T); free(occ);
}

/* move-to-front with the initial list 0..K-1; forward (enc = 1) or inverse (enc = 0). */
void mtf(const int32_t *in, int64_t n, int32_t K, int32_t enc, int32_t *out) {
    int32_t lst[256];
    for (int32_t k = 0; k < K; k++) lst[k] = k;
    for (int64_t i = 0; i < n; i++) {
        int32_t j, s;
        if (enc) { s = in[i]; for (j = 0; lst[j] != s; j++); out[i] = j; }
        else { j = in[i]; s = lst[j]; out[i] = s; }
        for (; j > 0; j--) lst[j] = lst[j - 1];
        lst[0] = s;
    }
}
