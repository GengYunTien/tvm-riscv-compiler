// tests/test_conv2d.c
// Unit test for conv2d_rvv against an independent reference.
//
// Build (host, scalar fallback):
//   gcc -O2 -o test_conv2d tests/test_conv2d.c kernels/rvv/conv2d_rvv.c -lm
// Build (ON the board, RVV path):
//   gcc -O2 -march=rv64gcv -o test_conv2d test_conv2d.c conv2d_rvv.c -lm
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void conv2d_rvv(const float* input, const float* weight, float* output,
                int N, int C_in, int H, int W,
                int C_out, int kH, int kW,
                int stride_h, int stride_w, int pad_h, int pad_w);

// Independent reference: deliberately simple, do not "optimize" it.
// Accumulates in double so the result does not depend on summation order.
static void conv2d_ref(const float* in, const float* wt, float* out,
                       int N, int C_in, int H, int W,
                       int C_out, int kH, int kW,
                       int sh, int sw, int ph, int pw) {
    int H_out = (H + 2 * ph - kH) / sh + 1;
    int W_out = (W + 2 * pw - kW) / sw + 1;
    for (int n = 0; n < N; n++)
      for (int co = 0; co < C_out; co++)
        for (int oh = 0; oh < H_out; oh++)
          for (int ow = 0; ow < W_out; ow++) {
            double sum = 0.0;
            for (int ci = 0; ci < C_in; ci++)
              for (int kh = 0; kh < kH; kh++)
                for (int kw = 0; kw < kW; kw++) {
                    int ih = oh * sh - ph + kh;
                    int iw = ow * sw - pw + kw;
                    if (ih < 0 || ih >= H || iw < 0 || iw >= W) continue;
                    sum += (double)in[((n * C_in + ci) * H + ih) * W + iw] *
                           (double)wt[((co * C_in + ci) * kH + kh) * kW + kw];
                }
            out[((n * C_out + co) * H_out + oh) * W_out + ow] = (float)sum;
          }
}

typedef struct {
    int N, C_in, H, W, C_out, kH, kW, sh, sw, ph, pw;
} Case;

static float frand(void) { return (float)rand() / RAND_MAX * 2.0f - 1.0f; }

static int run_case(const Case* c, int idx) {
    int H_out = (c->H + 2 * c->ph - c->kH) / c->sh + 1;
    int W_out = (c->W + 2 * c->pw - c->kW) / c->sw + 1;
    size_t n_in  = (size_t)c->N * c->C_in * c->H * c->W;
    size_t n_wt  = (size_t)c->C_out * c->C_in * c->kH * c->kW;
    size_t n_out = (size_t)c->N * c->C_out * H_out * W_out;

    float* in  = malloc(n_in * sizeof(float));
    float* wt  = malloc(n_wt * sizeof(float));
    float* got = malloc(n_out * sizeof(float));
    float* ref = malloc(n_out * sizeof(float));
    for (size_t i = 0; i < n_in; i++) in[i] = frand();
    for (size_t i = 0; i < n_wt; i++) wt[i] = frand();
    // Poison the output so a kernel that skips elements is caught.
    for (size_t i = 0; i < n_out; i++) got[i] = NAN;

    conv2d_ref(in, wt, ref, c->N, c->C_in, c->H, c->W, c->C_out, c->kH, c->kW,
               c->sh, c->sw, c->ph, c->pw);
    conv2d_rvv(in, wt, got, c->N, c->C_in, c->H, c->W, c->C_out, c->kH, c->kW,
               c->sh, c->sw, c->ph, c->pw);

    double max_abs = 0.0;
    int bad = 0;
    for (size_t i = 0; i < n_out; i++) {
        double d = fabs((double)got[i] - (double)ref[i]);
        double tol = 1e-4 + 1e-4 * fabs((double)ref[i]);
        if (!(d <= tol)) bad++;          // also catches NaN
        if (d > max_abs || d != d) max_abs = d;
    }
    printf("[%s] case %2d: N=%d Cin=%d %dx%d Cout=%d k=%dx%d s=%d,%d p=%d,%d  W_out=%d  max_abs=%.2e\n",
           bad ? "FAIL" : "PASS", idx, c->N, c->C_in, c->H, c->W, c->C_out,
           c->kH, c->kW, c->sh, c->sw, c->ph, c->pw, W_out, max_abs);
    free(in); free(wt); free(got); free(ref);
    return bad != 0;
}

int main(void) {
    srand(0);
    const Case cases[] = {
        // N Cin  H   W  Cout kH kW sh sw ph pw
        {1,  3,  9,  9,   8, 3, 3, 1, 1, 1, 1},  //  0 basic 3x3, pad 1
        {1,  3,  9,  9,   8, 3, 3, 2, 2, 1, 1},  //  1 stride 2
        {1,  4, 16, 16,  16, 3, 3, 1, 1, 0, 0},  //  2 no padding
        {1,  4, 16, 16,   3, 3, 3, 1, 1, 1, 1},  //  3 Cout not multiple of 8
        {1,  4, 16, 16,  10, 3, 3, 2, 2, 1, 1},  //  4 Cout=10, stride 2
        {1,  8, 15, 15,  17, 1, 1, 1, 1, 0, 0},  //  5 1x1, Cout=17
        {1,  8, 13, 11,   5, 3, 3, 1, 1, 1, 1},  //  6 odd, non-square
        {1,  3, 33, 33,   8, 3, 3, 2, 2, 1, 1},  //  7 W_out not multiple of 8
        {2,  3,  9,  9,   8, 3, 3, 1, 1, 1, 1},  //  8 batch > 1
        {1, 16,  7,  7,  32, 3, 3, 1, 1, 1, 1},  //  9 small spatial
        // Long rows: several full 8-lane segments, so the in-bounds fast path runs.
        {1,  4, 40, 40,   8, 3, 3, 1, 1, 1, 1},  // 10 stride 1, W_out=40
        {1,  4, 40, 40,   8, 3, 3, 1, 1, 0, 0},  // 11 stride 1, no padding, W_out=38
        {1,  4, 64, 64,   8, 3, 3, 2, 2, 1, 1},  // 12 stride 2 (vlse32), W_out=32
        {1,  8, 32, 32,  16, 1, 1, 1, 1, 0, 0},  // 13 1x1 stride 1, every segment in bounds
        {1,  8, 32, 32,  16, 1, 1, 2, 2, 0, 0},  // 14 1x1 stride 2, every segment in bounds
        {1,  3, 37, 53,   5, 3, 3, 1, 1, 1, 1},  // 15 W_out=53: tail segment of 5 lanes
        {1,  4, 24, 24,   4, 3, 3, 1, 1, 1, 1},  // 16 Cout=4: exactly one block, no tail
        {1,  4, 24, 24,   9, 3, 3, 1, 1, 1, 1},  // 17 Cout=9: two blocks + 1 tail channel
    };
    int n = (int)(sizeof(cases) / sizeof(cases[0]));
    int failed = 0;
    for (int i = 0; i < n; i++) failed += run_case(&cases[i], i);
    printf("%d / %d cases passed\n", n - failed, n);
    return failed ? 1 : 0;
}