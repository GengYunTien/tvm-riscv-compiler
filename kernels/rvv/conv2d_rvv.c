#include <stddef.h>
#include <string.h>
#ifdef __riscv_vector
#include <riscv_vector.h>
#endif

int rvv_kernel_abi_version(void) { return 1; }

#ifdef __riscv_vector
const char* rvv_kernel_name(void) { return "rvv-v1-ow-m1"; }

void conv2d_rvv(const float* input, const float* weight, float* output,
                int N, int C_in, int H, int W,
                int C_out, int kH, int kW,
                int stride_h, int stride_w, int pad_h, int pad_w) {
    int H_out = (H + 2 * pad_h - kH) / stride_h + 1;
    int W_out = (W + 2 * pad_w - kW) / stride_w + 1;

    for (int n = 0; n < N; n++)
    for (int co = 0; co < C_out; co++)
    for (int oh = 0; oh < H_out; oh++) {
        float* out_row = output + (((size_t)n * C_out + co) * H_out + oh) * W_out;
        for (int ow = 0; ow < W_out;) {
            size_t vl = __riscv_vsetvl_e32m1(W_out - ow);
            vfloat32m1_t acc = __riscv_vfmv_v_f_f32m1(0.0f, vl);

            // vector of iw base positions for this segment: (ow+i)*stride_w - pad_w
            vuint32m1_t idx = __riscv_vid_v_u32m1(vl);
            vint32m1_t iw0 = __riscv_vadd_vx_i32m1(
                __riscv_vreinterpret_v_u32m1_i32m1(idx), ow, vl);
            iw0 = __riscv_vmul_vx_i32m1(iw0, stride_w, vl);
            iw0 = __riscv_vsub_vx_i32m1(iw0, pad_w, vl);

            for (int ci = 0; ci < C_in; ci++)
            for (int kh = 0; kh < kH; kh++) {
                int ih = oh * stride_h - pad_h + kh;
                if (ih < 0 || ih >= H) continue;
                const float* in_row = input + (((size_t)n * C_in + ci) * H + ih) * W;
                const float* w_row = weight + (((size_t)co * C_in + ci) * kH + kh) * kW;
                for (int kw = 0; kw < kW; kw++) {
                    vint32m1_t iw = __riscv_vadd_vx_i32m1(iw0, kw, vl);
                    // valid lanes: 0 <= iw < W
                    vbool32_t ok = __riscv_vmand_mm_b32(
                        __riscv_vmsge_vx_i32m1_b32(iw, 0, vl),
                        __riscv_vmslt_vx_i32m1_b32(iw, W, vl), vl);
                    // byte offsets from in_row, only used under mask
                    vuint32m1_t off = __riscv_vsll_vx_u32m1(
                        __riscv_vreinterpret_v_i32m1_u32m1(iw), 2, vl);
                    vfloat32m1_t x = __riscv_vluxei32_v_f32m1_m(
                        ok, in_row, off, vl);
                    x = __riscv_vmerge_vvm_f32m1(
                        __riscv_vfmv_v_f_f32m1(0.0f, vl), x, ok, vl);
                    acc = __riscv_vfmacc_vf_f32m1(acc, w_row[kw], x, vl);
                }
            }
            __riscv_vse32_v_f32m1(out_row + ow, acc, vl);
            ow += (int)vl;
        }
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