#include <stddef.h>
#include <string.h>
#ifdef __riscv_vector
#include <riscv_vector.h>
#endif

int rvv_kernel_abi_version(void) { return 1; }

#ifdef __riscv_vector

const char* rvv_kernel_name(void) { return "rvv-v3-ow-m1-co8"; }

typedef struct {
    int C_in, H, W, C_out, kH, kW;
    int stride_h, stride_w, pad_h, pad_w;
    int H_out, W_out;
} Params;

// Load the input values read by the vl lanes of one output segment for one kw.
// iw_first is the input column read by lane 0.
static inline __attribute__((always_inline)) vfloat32m1_t
load_x(const float* in_row, int iw_first, size_t vl, int W, int stride_w) {
    int iw_last = iw_first + ((int)vl - 1) * stride_w;
    if (iw_first >= 0 && iw_last < W) {
        // Fast path: the whole segment is inside the row.
        if (stride_w == 1) {
            return __riscv_vle32_v_f32m1(in_row + iw_first, vl);
        }
        return __riscv_vlse32_v_f32m1(
            in_row + iw_first, (ptrdiff_t)stride_w * (ptrdiff_t)sizeof(float), vl);
    }
    // Edge path: lanes that fall into the padding read 0.
    vuint32m1_t lane = __riscv_vid_v_u32m1(vl);
    vint32m1_t iw = __riscv_vadd_vx_i32m1(
        __riscv_vmul_vx_i32m1(__riscv_vreinterpret_v_u32m1_i32m1(lane), stride_w, vl),
        iw_first, vl);
    vbool32_t ok = __riscv_vmand_mm_b32(__riscv_vmsge_vx_i32m1_b32(iw, 0, vl),
                                        __riscv_vmslt_vx_i32m1_b32(iw, W, vl), vl);
    vuint32m1_t off = __riscv_vsll_vx_u32m1(__riscv_vreinterpret_v_i32m1_u32m1(iw), 2, vl);
    vfloat32m1_t g = __riscv_vluxei32_v_f32m1_m(ok, in_row, off, vl);
    return __riscv_vmerge_vvm_f32m1(__riscv_vfmv_v_f_f32m1(0.0f, vl), g, ok, vl);
}

// One output row (n, oh) for 8 consecutive output channels co .. co+7.
static void row_co8(const float* input, const float* weight, float* output,
                    int n, int co, int oh, const Params* p) {
    const size_t plane = (size_t)p->H_out * p->W_out;
    const size_t w_co = (size_t)p->C_in * p->kH * p->kW;  // weight stride between output channels
    float* o0 = output + ((size_t)n * p->C_out + co) * plane + (size_t)oh * p->W_out;
    float* o1 = o0 + plane;
    float* o2 = o1 + plane;
    float* o3 = o2 + plane;
    float* o4 = o3 + plane;
    float* o5 = o4 + plane;
    float* o6 = o5 + plane;
    float* o7 = o6 + plane;

    for (int ow = 0; ow < p->W_out;) {
        size_t vl = __riscv_vsetvl_e32m1((size_t)(p->W_out - ow));
        vfloat32m1_t acc0 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
        vfloat32m1_t acc1 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
        vfloat32m1_t acc2 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
        vfloat32m1_t acc3 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
        vfloat32m1_t acc4 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
        vfloat32m1_t acc5 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
        vfloat32m1_t acc6 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
        vfloat32m1_t acc7 = __riscv_vfmv_v_f_f32m1(0.0f, vl);

        for (int ci = 0; ci < p->C_in; ci++)
        for (int kh = 0; kh < p->kH; kh++) {
            int ih = oh * p->stride_h - p->pad_h + kh;
            if (ih < 0 || ih >= p->H) continue;

            const float* in_row = input + (((size_t)n * p->C_in + ci) * p->H + ih) * p->W;
            const float* w = weight + (((size_t)co * p->C_in + ci) * p->kH + kh) * p->kW;

            for (int kw = 0; kw < p->kW; kw++) {
                int iw_first = ow * p->stride_w - p->pad_w + kw;
                vfloat32m1_t x = load_x(in_row, iw_first, vl, p->W, p->stride_w);
                acc0 = __riscv_vfmacc_vf_f32m1(acc0, w[kw], x, vl);
                acc1 = __riscv_vfmacc_vf_f32m1(acc1, w[w_co + kw], x, vl);
                acc2 = __riscv_vfmacc_vf_f32m1(acc2, w[2 * w_co + kw], x, vl);
                acc3 = __riscv_vfmacc_vf_f32m1(acc3, w[3 * w_co + kw], x, vl);
                acc4 = __riscv_vfmacc_vf_f32m1(acc4, w[4 * w_co + kw], x, vl);
                acc5 = __riscv_vfmacc_vf_f32m1(acc5, w[5 * w_co + kw], x, vl);
                acc6 = __riscv_vfmacc_vf_f32m1(acc6, w[6 * w_co + kw], x, vl);
                acc7 = __riscv_vfmacc_vf_f32m1(acc7, w[7 * w_co + kw], x, vl);
            }
        }

        __riscv_vse32_v_f32m1(o0 + ow, acc0, vl);
        __riscv_vse32_v_f32m1(o1 + ow, acc1, vl);
        __riscv_vse32_v_f32m1(o2 + ow, acc2, vl);
        __riscv_vse32_v_f32m1(o3 + ow, acc3, vl);
        __riscv_vse32_v_f32m1(o4 + ow, acc4, vl);
        __riscv_vse32_v_f32m1(o5 + ow, acc5, vl);
        __riscv_vse32_v_f32m1(o6 + ow, acc6, vl);
        __riscv_vse32_v_f32m1(o7 + ow, acc7, vl);
        ow += (int)vl;
    }
}

// One output row (n, oh) for 4 consecutive output channels co .. co+3.
static void row_co4(const float* input, const float* weight, float* output,
                    int n, int co, int oh, const Params* p) {
    const size_t plane = (size_t)p->H_out * p->W_out;
    const size_t w_co = (size_t)p->C_in * p->kH * p->kW;  // weight stride between output channels
    float* o0 = output + ((size_t)n * p->C_out + co) * plane + (size_t)oh * p->W_out;
    float* o1 = o0 + plane;
    float* o2 = o1 + plane;
    float* o3 = o2 + plane;

    for (int ow = 0; ow < p->W_out;) {
        size_t vl = __riscv_vsetvl_e32m1((size_t)(p->W_out - ow));
        vfloat32m1_t acc0 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
        vfloat32m1_t acc1 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
        vfloat32m1_t acc2 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
        vfloat32m1_t acc3 = __riscv_vfmv_v_f_f32m1(0.0f, vl);

        for (int ci = 0; ci < p->C_in; ci++)
        for (int kh = 0; kh < p->kH; kh++) {
            int ih = oh * p->stride_h - p->pad_h + kh;
            if (ih < 0 || ih >= p->H) continue;

            const float* in_row = input + (((size_t)n * p->C_in + ci) * p->H + ih) * p->W;
            const float* w = weight + (((size_t)co * p->C_in + ci) * p->kH + kh) * p->kW;

            for (int kw = 0; kw < p->kW; kw++) {
                int iw_first = ow * p->stride_w - p->pad_w + kw;
                vfloat32m1_t x = load_x(in_row, iw_first, vl, p->W, p->stride_w);
                acc0 = __riscv_vfmacc_vf_f32m1(acc0, w[kw], x, vl);
                acc1 = __riscv_vfmacc_vf_f32m1(acc1, w[w_co + kw], x, vl);
                acc2 = __riscv_vfmacc_vf_f32m1(acc2, w[2 * w_co + kw], x, vl);
                acc3 = __riscv_vfmacc_vf_f32m1(acc3, w[3 * w_co + kw], x, vl);
            }
        }

        __riscv_vse32_v_f32m1(o0 + ow, acc0, vl);
        __riscv_vse32_v_f32m1(o1 + ow, acc1, vl);
        __riscv_vse32_v_f32m1(o2 + ow, acc2, vl);
        __riscv_vse32_v_f32m1(o3 + ow, acc3, vl);
        ow += (int)vl;
    }
}

// One output row (n, oh) for a single output channel (tail of C_out % 4).
static void row_co1(const float* input, const float* weight, float* output,
                    int n, int co, int oh, const Params* p) {
    float* o0 = output + (((size_t)n * p->C_out + co) * p->H_out + oh) * p->W_out;

    for (int ow = 0; ow < p->W_out;) {
        size_t vl = __riscv_vsetvl_e32m1((size_t)(p->W_out - ow));
        vfloat32m1_t acc = __riscv_vfmv_v_f_f32m1(0.0f, vl);

        for (int ci = 0; ci < p->C_in; ci++)
        for (int kh = 0; kh < p->kH; kh++) {
            int ih = oh * p->stride_h - p->pad_h + kh;
            if (ih < 0 || ih >= p->H) continue;

            const float* in_row = input + (((size_t)n * p->C_in + ci) * p->H + ih) * p->W;
            const float* w = weight + (((size_t)co * p->C_in + ci) * p->kH + kh) * p->kW;

            for (int kw = 0; kw < p->kW; kw++) {
                int iw_first = ow * p->stride_w - p->pad_w + kw;
                vfloat32m1_t x = load_x(in_row, iw_first, vl, p->W, p->stride_w);
                acc = __riscv_vfmacc_vf_f32m1(acc, w[kw], x, vl);
            }
        }

        __riscv_vse32_v_f32m1(o0 + ow, acc, vl);
        ow += (int)vl;
    }
}

void conv2d_rvv(const float* input, const float* weight, float* output,
                int N, int C_in, int H, int W,
                int C_out, int kH, int kW,
                int stride_h, int stride_w, int pad_h, int pad_w) {
    Params p;
    p.C_in = C_in; p.H = H; p.W = W; p.C_out = C_out; p.kH = kH; p.kW = kW;
    p.stride_h = stride_h; p.stride_w = stride_w; p.pad_h = pad_h; p.pad_w = pad_w;
    p.H_out = (H + 2 * pad_h - kH) / stride_h + 1;
    p.W_out = (W + 2 * pad_w - kW) / stride_w + 1;

    for (int n = 0; n < N; n++) {
        int co = 0;
        for (; co + 8 <= C_out; co += 8)
            for (int oh = 0; oh < p.H_out; oh++)
                row_co8(input, weight, output, n, co, oh, &p);
        for (; co + 4 <= C_out; co += 4)
            for (int oh = 0; oh < p.H_out; oh++)
                row_co4(input, weight, output, n, co, oh, &p);
        for (; co < C_out; co++)
            for (int oh = 0; oh < p.H_out; oh++)
                row_co1(input, weight, output, n, co, oh, &p);
    }
}
#else
// scalar fallback (host / non-vector builds)
const char* rvv_kernel_name(void) { return "scalar"; }

void conv2d_rvv(const float* input, const float* weight, float* output,
                int N, int C_in, int H, int W,
                int C_out, int kH, int kW,
                int stride_h, int stride_w, int pad_h, int pad_w) {
    int H_out = (H + 2 * pad_h - kH) / stride_h + 1;
    int W_out = (W + 2 * pad_w - kW) / stride_w + 1;
    for (int n = 0; n < N; n++)
    for (int co = 0; co < C_out; co++)
    for (int oh = 0; oh < H_out; oh++)
    for (int ow = 0; ow < W_out; ow++) {
        float sum = 0.0f;
        for (int ci = 0; ci < C_in; ci++)
        for (int kh = 0; kh < kH; kh++)
        for (int kw = 0; kw < kW; kw++) {
            int ih = oh * stride_h - pad_h + kh;
            int iw = ow * stride_w - pad_w + kw;
            if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                sum += input[(((size_t)n * C_in + ci) * H + ih) * W + iw] *
                       weight[(((size_t)co * C_in + ci) * kH + kh) * kW + kw];
        }
        output[(((size_t)n * C_out + co) * H_out + oh) * W_out + ow] = sum;
    }
}
#endif