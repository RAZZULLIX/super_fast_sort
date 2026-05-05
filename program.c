#include <inttypes.h>
#include <math.h>
#include <time.h>
#include <limits.h>
#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <immintrin.h>
#include <assert.h>

#ifndef EXPORT
#define EXPORT __attribute__((visibility("default")))
#endif

/* --------------------------------------------------------------------- */
/*  Thread‑local scratch buffers                                        */
/* --------------------------------------------------------------------- */
static __thread uint32_t *radix_tmp        = NULL;
static __thread size_t    radix_tmp_size   = 0;
static __thread uint32_t *hist_buffer      = NULL;
static __thread size_t    hist_buffer_size = 0;
static __thread uint32_t *decode_tmp       = NULL;
static __thread size_t    decode_tmp_size  = 0;

/* --------------------------------------------------------------------- */
/*  Aligned allocation (fallback to malloc)                              */
/* --------------------------------------------------------------------- */
static inline void *aligned_alloc_compat(size_t alignment, size_t size) {
    void *ptr = NULL;
    if (posix_memalign(&ptr, alignment, size) != 0) ptr = malloc(size);
    return ptr;
}

/* --------------------------------------------------------------------- */
/*  Insertion sort (for very small arrays)                              */
/* --------------------------------------------------------------------- */
static inline void insertion_sort(uint32_t *restrict a, size_t n) {
    if (n <= 1) return;
    
    for (size_t i = 1; i < n; ++i) {
        uint32_t v = a[i];
        size_t j = i;
        
        // Optimization: check if it's already in place to avoid the while loop
        if (a[j-1] <= v) continue;
        
        // Use a pointer-based approach to reduce indexing overhead
        uint32_t *const p_start = a;
        uint32_t *const p_curr = &a[j];
        uint32_t *const p_prev = &a[j-1];
        
        // Shift elements to the right
        while (j > 0 && a[j-1] > v) {
            a[j] = a[j-1];
            --j;
        }
        a[j] = v;
    }
}

/* --------------------------------------------------------------------- */
/*  Compute min, max, used‑bits mask (one pass AVX2)                     */
/* --------------------------------------------------------------------- */
static inline void compute_stats_avx2(const uint32_t *restrict a, size_t n,
                                      uint32_t *out_min, uint32_t *out_max,
                                      uint32_t *out_used) {
    if (n == 0) { *out_min = *out_max = *out_used = 0; return; }
    __m256i v_min = _mm256_set1_epi32(-1);
    __m256i v_max = _mm256_setzero_si256();
    __m256i v_used = _mm256_setzero_si256();
    size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        __m256i v0 = _mm256_loadu_si256((const __m256i*)(a+i));
        __m256i v1 = _mm256_loadu_si256((const __m256i*)(a+i+8));
        __m256i v2 = _mm256_loadu_si256((const __m256i*)(a+i+16));
        __m256i v3 = _mm256_loadu_si256((const __m256i*)(a+i+24));
        v_used = _mm256_or_si256(v_used, _mm256_or_si256(_mm256_or_si256(v0,v1), _mm256_or_si256(v2,v3)));
        v_min = _mm256_min_epu32(v_min, _mm256_min_epu32(_mm256_min_epu32(v0,v1), _mm256_min_epu32(v2,v3)));
        v_max = _mm256_max_epu32(v_max, _mm256_max_epu32(_mm256_max_epu32(v0,v1), _mm256_max_epu32(v2,v3)));
    }
    for (; i + 8 <= n; i += 8) {
        __m256i v = _mm256_loadu_si256((const __m256i*)(a+i));
        v_used = _mm256_or_si256(v_used, v);
        v_min = _mm256_min_epu32(v_min, v);
        v_max = _mm256_max_epu32(v_max, v);
    }
    uint32_t tmp_min[8], tmp_max[8], tmp_used[8];
    _mm256_storeu_si256((__m256i*)tmp_min, v_min);
    _mm256_storeu_si256((__m256i*)tmp_max, v_max);
    _mm256_storeu_si256((__m256i*)tmp_used, v_used);
    uint32_t mn = tmp_min[0], mx = tmp_max[0], us = tmp_used[0];
    for (int k = 1; k < 8; ++k) {
        if (tmp_min[k] < mn) mn = tmp_min[k];
        if (tmp_max[k] > mx) mx = tmp_max[k];
        us |= tmp_used[k];
    }
    for (; i < n; ++i) {
        uint32_t v = a[i];
        if (v < mn) mn = v;
        if (v > mx) mx = v;
        us |= v;
    }
    *out_min = mn;
    *out_max = mx;
    *out_used = us;
}

/* --------------------------------------------------------------------- */
/*  Low‑cardinality sort for arrays with few distinct values (≤256)     */
/* --------------------------------------------------------------------- */
#define LC_HT_CAP          4096u
#define LC_SAMPLES         512u
#define LC_MAX_DISTINCT    256u
#define LC_MAX_PROBES      32u


static bool low_cardinality_sort(uint32_t *arr, size_t n) {
    if (n <= LC_SAMPLES) return false;
    const size_t cap = LC_HT_CAP;
    const size_t mask = cap - 1;
    const uint32_t max_probes = LC_MAX_PROBES;
    const size_t samples = LC_SAMPLES;
    struct { uint32_t val; uint32_t cnt; } table[cap];
    __builtin_memset(table, 0, sizeof(table));
    size_t step = n / samples;
    size_t distinct = 0;
    bool probe_ok = true;
    for (size_t i = 0; i < samples; ++i) {
        uint32_t v = arr[i * step];
        size_t h = (v * 0x9e3779b9u) & mask;
        uint32_t probes = 0;
        while (table[h].cnt && table[h].val != v && probes < max_probes) {
            h = (h + 1) & mask;
            ++probes;
        }
        if (!table[h].cnt) {
            table[h].val = v;
            table[h].cnt = 1;
            if (++distinct > cap / 2) { probe_ok = false; break; }
        } else {
            ++table[h].cnt;
        }
    }
    if (!probe_ok) return false;
    __builtin_memset(table, 0, sizeof(table));
    size_t full_distinct = 0;
    bool table_ok = true;
    for (size_t i = 0; i < n; ++i) {
        uint32_t v = arr[i];
        size_t h = (v * 0x9e3779b9u) & mask;
        uint32_t probes = 0;
        while (table[h].cnt && table[h].val != v && probes < max_probes) {
            h = (h + 1) & mask;
            ++probes;
        }
        if (!table[h].cnt) {
            table[h].val = v;
            table[h].cnt = 1;
            if (++full_distinct > LC_MAX_DISTINCT || full_distinct >= cap) {
                table_ok = false;
                break;
            }
        } else {
            ++table[h].cnt;
        }
    }
    if (!table_ok) return false;
    size_t m = 0;
    for (size_t k = 0; k < cap; ++k) {
        if (table[k].cnt) {
            if (k != m) {
                table[m].val = table[k].val;
                table[m].cnt = table[k].cnt;
            }
            ++m;
        }
    }
    for (size_t i = 1; i < m; ++i) {
        uint32_t v = table[i].val;
        uint32_t c = table[i].cnt;
        size_t j = i;
        while (j > 0 && table[j - 1].val > v) {
            table[j].val = table[j - 1].val;
            table[j].cnt = table[j - 1].cnt;
            --j;
        }
        table[j].val = v;
        table[j].cnt = c;
    }
    size_t pos = 0;
    for (size_t i = 0; i < m; ++i) {
        uint32_t v = table[i].val;
        uint32_t c = table[i].cnt;
        for (uint32_t k = 0; k < c; ++k) arr[pos++] = v;
    }
    return true;
}

/* --------------------------------------------------------------------- */
/*  Median of three (used by quicksort)                                 */
/* --------------------------------------------------------------------- */
static inline uint32_t med3(uint32_t a, uint32_t b, uint32_t c) {
    return (a < b) ? ((b < c) ? b : ((a < c) ? c : a)) : ((b > c) ? b : ((a > c) ? c : a));
}

/* --------------------------------------------------------------------- */
/*  3‑way quicksort (tail‑recursive, iterative)                        */
/* --------------------------------------------------------------------- */
static void quicksort_3way_pro(uint32_t *a, int32_t lo, int32_t hi) {
    while (hi - lo > 32) {
        uint32_t n = (uint32_t)(hi - lo + 1), s = n / 8;
        uint32_t pivot = med3(
            med3(a[lo],           a[lo+s],       a[lo+2*s]),
            med3(a[lo+n/2-s],     a[lo+n/2],     a[lo+n/2+s]),
            med3(a[hi-2*s],       a[hi-s],        a[hi])
        );
        int32_t lt = lo, gt = hi, i = lo;
        while (i <= gt) {
            if      (a[i] < pivot) { uint32_t t=a[lt]; a[lt]=a[i]; a[i]=t; ++lt; ++i; }
            else if (a[i] > pivot) { uint32_t t=a[gt]; a[gt]=a[i]; a[i]=t; --gt; }
            else ++i;
        }
        if (lt - lo < hi - gt) { quicksort_3way_pro(a, lo, lt-1); lo=gt+1; }
        else                   { quicksort_3way_pro(a, gt+1, hi); hi=lt-1; }
    }
    if (hi > lo) insertion_sort(&a[lo], (size_t)(hi-lo+1));
}

/* --------------------------------------------------------------------- */
/*  Detect heavily duplicated data (clumped)                            */
/* --------------------------------------------------------------------- */
#define DC_HT_CAP    1024u
#define DC_HT_MASK   (DC_HT_CAP-1)
#define DC_THRESHOLD 256u
#define DC_SAMPLES   512u

static bool detect_clumped_simd(const uint32_t *a, uint32_t n) {
    if (n < 512) return false;
    uint32_t threshold = a[n/2];
    __m256i t_vec = _mm256_set1_epi32((int)threshold);
    uint32_t matches = 0, step = n / 512;
    for (uint32_t i = 0; i < 512; i += 8) {
        uint32_t s[8];
        for (uint32_t j = 0; j < 8; ++j) s[j] = a[(i+j)*step];
        __m256i v    = _mm256_loadu_si256((const __m256i*)s);
        __m256i eq   = _mm256_cmpeq_epi32(v, t_vec);
        matches += (uint32_t)__builtin_popcount(
            _mm256_movemask_ps(_mm256_castsi256_ps(eq)));
    }
    return matches > 50;
}

/* --------------------------------------------------------------------- */
/*  Detect two distinct values (fast SIMD)                               */
/* --------------------------------------------------------------------- */
static bool detect_two_values_avx2(const uint32_t *arr, size_t n, uint32_t min_v, uint32_t max_v) {
    if (n == 0) return false;
    const __m256i min_vec = _mm256_set1_epi32((int)min_v);
    const __m256i max_vec = _mm256_set1_epi32((int)max_v);
    const __m256i all_ones = _mm256_set1_epi32(-1);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256i v = _mm256_loadu_si256((const __m256i*)(arr + i));
        __m256i eq_min = _mm256_cmpeq_epi32(v, min_vec);
        __m256i eq_max = _mm256_cmpeq_epi32(v, max_vec);
        __m256i eq = _mm256_or_si256(eq_min, eq_max);
        __m256i mismatch = _mm256_xor_si256(eq, all_ones);
        if (_mm256_movemask_epi8(mismatch)) return false;
    }
    for (; i < n; ++i) {
        if (arr[i] != min_v && arr[i] != max_v) return false;
    }
    return true;
}

/* --------------------------------------------------------------------- */
/*  Zero histogram with AVX2                                             */
/* --------------------------------------------------------------------- */
static inline void zero_histogram_avx2(uint32_t *hist, size_t size) {
    size_t i = 0;
    __m256i zero = _mm256_setzero_si256();
    for (; i + 8 <= size; i += 8) {
        _mm256_storeu_si256((__m256i*)(hist + i), zero);
    }
    for (; i < size; ++i) hist[i] = 0;
}

/* --------------------------------------------------------------------- */
/*  Generic radix pass (histogram prefix sum + scatter)                  */
/* --------------------------------------------------------------------- */
static inline void radix_pass(
    uint32_t *restrict src, uint32_t *restrict dst, uint32_t n,
    uint32_t *restrict hist, uint32_t shift, uint32_t mask)
{
    uint32_t sum = 0;
    for (uint32_t i = 0; i <= mask; ++i) { uint32_t c=hist[i]; hist[i]=sum; sum+=c; }
    for (uint32_t i = 0; i < n; ++i) {
        uint32_t v = src[i];
        dst[hist[(v>>shift)&mask]++] = v;
    }
}

/* --------------------------------------------------------------------- */
/*  8‑bit radix sort (four passes)                                       */
/* --------------------------------------------------------------------- */
static void radix_sort_8bit(uint32_t *restrict arr, uint32_t *restrict tmp, size_t n) {
    const unsigned passes = 4;
    uint32_t *src = arr;
    uint32_t *dst = tmp;
    for (unsigned p = 0; p < passes; ++p) {
        unsigned shift = p * 8;
        uint32_t hist[256];
        memset(hist, 0, sizeof(hist));
        for (size_t i = 0; i < n; ++i) {
            ++hist[(src[i] >> shift) & 0xFFu];
        }
        uint32_t pos = 0;
        for (int k = 0; k < 256; ++k) {
            uint32_t tmpcnt = hist[k];
            hist[k] = pos;
            pos += tmpcnt;
        }
        for (size_t i = 0; i < n; ++i) {
            uint32_t v = src[i];
            dst[hist[(v >> shift) & 0xFFu]++] = v;
        }
        uint32_t *t = src;
        src = dst;
        dst = t;
    }
    if (passes & 1) {
        memcpy(arr, tmp, n * sizeof(uint32_t));
    }
}

/* --------------------------------------------------------------------- */
/*  16‑bit radix sort (two passes)                                       */
/* --------------------------------------------------------------------- */
static inline void radix_sort_16bit(uint32_t *restrict a, uint32_t *restrict b, size_t n) {
    const size_t hist_size = 65536u * 2u;
    if (hist_buffer_size < hist_size) {
        free(hist_buffer);
        hist_buffer = aligned_alloc_compat(64, hist_size * sizeof(uint32_t));
        hist_buffer_size = hist_size;
    }
    uint32_t *h0 = hist_buffer, *h1 = hist_buffer + 65536u;
    memset(hist_buffer, 0, hist_size * sizeof(uint32_t));
    size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        uint32_t vals[32];
        _mm256_storeu_si256((__m256i*)vals,      _mm256_loadu_si256((const __m256i*)(a+i)));
        _mm256_storeu_si256((__m256i*)(vals+8),  _mm256_loadu_si256((const __m256i*)(a+i+8)));
        _mm256_storeu_si256((__m256i*)(vals+16), _mm256_loadu_si256((const __m256i*)(a+i+16)));
        _mm256_storeu_si256((__m256i*)(vals+24), _mm256_loadu_si256((const __m256i*)(a+i+24)));
        for (int k = 0; k < 32; ++k) { ++h0[vals[k] & 0xFFFFu]; ++h1[vals[k] >> 16]; }
    }
    for (; i < n; ++i) { ++h0[a[i] & 0xFFFFu]; ++h1[a[i] >> 16]; }
    uint32_t s = 0;
    for (int k = 0; k < 65536; ++k) { uint32_t c = h0[k]; h0[k] = s; s += c; }
    for (i = 0; i < n; ++i) { uint32_t x = a[i]; b[h0[x & 0xFFFFu]++] = x; }
    s = 0;
    for (int k = 0; k < 65536; ++k) { uint32_t c = h1[k]; h1[k] = s; s += c; }
    for (i = 0; i < n; ++i) { uint32_t x = b[i]; a[h1[x >> 16]++] = x; }
}

/* --------------------------------------------------------------------- */
/*  SIMD sortedness test (ascending / descending)                       */
/* --------------------------------------------------------------------- */
static inline int check_order_simd(const uint32_t *restrict a, uint32_t n) {
    if (n < 2) return 1;
    bool asc = true, desc = true;
    const __m256i xor_mask = _mm256_set1_epi32((int)0x80000000u);
    uint32_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256i v0 = _mm256_loadu_si256((const __m256i*)&a[i]);
        __m256i v1 = _mm256_loadu_si256((const __m256i*)&a[i+1]);
        __m256i u0 = _mm256_xor_si256(v0, xor_mask);
        __m256i u1 = _mm256_xor_si256(v1, xor_mask);
        if (asc && _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpgt_epi32(u0,u1)))) asc = false;
        if (desc && _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpgt_epi32(u1,u0)))) desc = false;
        if (!asc && !desc) return 0;
    }
    for (; i + 1 < n; ++i) {
        if (a[i] > a[i+1]) asc = false;
        if (a[i] < a[i+1]) desc = false;
    }
    return asc ? 1 : (desc ? -1 : 0);
}

/* --------------------------------------------------------------------- */
/*  Reverse array (used for reverse sorted data)                        */
/* --------------------------------------------------------------------- */
static inline void reverse_array(uint32_t *restrict a, size_t n) {
    size_t i = 0, j = n-1;
    while (i < j) {
        uint32_t t = a[i];
        a[i] = a[j];
        a[j] = t;
        ++i; --j;
    }
}

/* --------------------------------------------------------------------- */
/*  Main sort routine (hybrid)                                           */
/* --------------------------------------------------------------------- */
EXPORT void super_fast_sort_uint32(uint32_t *restrict arr, size_t n_in) {
    if (n_in <= 1) return;
    if (n_in < 64) { insertion_sort(arr, n_in); return; }

    /* Quick sortedness test on first 64 elements */
    int ord = check_order_simd(arr, (uint32_t)n_in);
    if (ord == 1) return;
    if (ord == -1) { reverse_array(arr, n_in); return; }

    /* Allocate scratch if needed */
    if (radix_tmp_size < n_in) {
        free(radix_tmp);
        radix_tmp = aligned_alloc_compat(64, n_in * sizeof(uint32_t));
        radix_tmp_size = n_in;
    }

    /* Compute min, max, used bits */
    uint32_t min_v, max_v, used_mask;
    compute_stats_avx2(arr, n_in, &min_v, &max_v, &used_mask);
    if (min_v == max_v) return; /* all equal */

    /* Small range counting sort (≤65535) */
    if (max_v - min_v <= 65535u) {
        size_t range = (size_t)(max_v - min_v + 1u);
        if (hist_buffer_size < range) {
            free(hist_buffer);
            hist_buffer = aligned_alloc_compat(64, range * sizeof(uint32_t));
            hist_buffer_size = range;
        }
        memset(hist_buffer, 0, range * sizeof(uint32_t));
        for (size_t i = 0; i < n_in; ++i) ++hist_buffer[arr[i] - min_v];
        size_t pos = 0;
        for (size_t k = 0; k < range; ++k) {
            uint32_t cnt = hist_buffer[k];
            hist_buffer[k] = pos;
            pos += cnt;
        }
        for (size_t i = 0; i < n_in; ++i) {
            uint32_t v = arr[i];
            radix_tmp[hist_buffer[v - min_v]++] = v;
        }
        memcpy(arr, radix_tmp, n_in * sizeof(uint32_t));
        return;
    }

    /* Detect two distinct values */
    if (detect_two_values_avx2(arr, n_in, min_v, max_v)) {
        size_t c1 = 0, c2 = 0;
        for (size_t i = 0; i < n_in; ++i) {
            if (arr[i] == min_v) ++c1;
            else ++c2;
        }
        size_t pos = 0;
        if (min_v < max_v) {
            while (c1--) arr[pos++] = min_v;
            while (c2--) arr[pos++] = max_v;
        } else {
            while (c2--) arr[pos++] = max_v;
            while (c1--) arr[pos++] = min_v;
        }
        return;
    }

    /* Detect heavily duplicated data */
    if (detect_clumped_simd(arr, (uint32_t)n_in)) {
        quicksort_3way_pro(arr, 0, (int32_t)n_in - 1);
        return;
    }

    /* Large arrays – 16‑bit radix */
    if (n_in >= 120000) {
        radix_sort_16bit(arr, radix_tmp, n_in);
        return;
    }

    /* Small arrays – 8‑bit radix */
    if (n_in <= 8192) {
        radix_sort_8bit(arr, radix_tmp, n_in);
        return;
    }

    /* Adaptive radix passes based on used bits */
    int used_bits = used_mask ? (32 - __builtin_clz((unsigned)used_mask)) : 1;
    int passes = (used_bits <= 11) ? 1 : (used_bits <= 22) ? 2 : 3;

    /* Allocate histograms */
    const size_t need_hist = 2048u + 2048u + 1024u;
    if (hist_buffer_size < need_hist) {
        free(hist_buffer);
        hist_buffer = aligned_alloc_compat(64, need_hist * sizeof(uint32_t));
        hist_buffer_size = need_hist;
    }
    uint32_t *h0 = hist_buffer;
    uint32_t *h1 = hist_buffer + 2048u;
    uint32_t *h2 = hist_buffer + 4096u;
    zero_histogram_avx2(h0, need_hist);

    for (size_t i = 0; i < n_in; ++i) {
        uint32_t x = arr[i];
        ++h0[x & 0x7FFu];
        if (passes > 1) ++h1[(x >> 11) & 0x7FFu];
        if (passes > 2) ++h2[x >> 22];
    }

    radix_pass(arr, radix_tmp, (uint32_t)n_in, h0, 0, 0x7FFu);
    if (passes == 1) {
        memcpy(arr, radix_tmp, n_in * sizeof(uint32_t));
        return;
    }
    radix_pass(radix_tmp, arr, (uint32_t)n_in, h1, 11, 0x7FFu);
    if (passes == 2) return;
    radix_pass(arr, radix_tmp, (uint32_t)n_in, h2, 22, 0x3FFu);
    memcpy(arr, radix_tmp, n_in * sizeof(uint32_t));
}

/* --------------------------------------------------------------------- */
/*  Cleanup scratch buffers                                               */
/* --------------------------------------------------------------------- */
EXPORT void cleanup_sort_buffers(void) {
    if (radix_tmp)   { free(radix_tmp);   radix_tmp   = NULL; radix_tmp_size   = 0; }
    if (hist_buffer) { free(hist_buffer); hist_buffer = NULL; hist_buffer_size = 0; }
    if (decode_tmp)  { free(decode_tmp);  decode_tmp  = NULL; decode_tmp_size  = 0; }
}
