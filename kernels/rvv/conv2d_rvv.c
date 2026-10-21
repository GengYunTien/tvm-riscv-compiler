#include <string.h>
#include <stdint.h>

void conv2d_rvv(
    const float* input,
    const float* weight,
    float*       output,
    int N,
    int C_in,  int H,     int W,
    int C_out, int kH,    int kW,
    int stride_h, int stride_w,
    int pad_h,    int pad_w
) {
    int H_out = (H + 2 * pad_h - kH) / stride_h + 1;
    int W_out = (W + 2 * pad_w - kW) / stride_w + 1;

    // initialize output to 0
    memset(output, 0, N * C_out * H_out * W_out * sizeof(float));

    // Naive scalar conv2d (NCHW)
    for (int n = 0; n < N; n++) {
        for (int co = 0; co < C_out; co++) {
            for (int oh = 0; oh < H_out; oh++) {
                for (int ow = 0; ow < W_out; ow++) {
                    float sum = 0.0f;
                    for (int ci = 0; ci < C_in; ci++) {
                        for (int kh = 0; kh < kH; kh++) {
                            for (int kw = 0; kw < kW; kw++) {
                                int ih = oh * stride_h - pad_h + kh;
                                int iw = ow * stride_w - pad_w + kw;
                                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                    float in_val = input[((n * C_in + ci) * H + ih) * W + iw];
                                    float w_val  = weight[((co * C_in + ci) * kH + kh) * kW + kw];
                                    sum += in_val * w_val;
                                }
                            }
                        }
                    }
                    output[((n * C_out + co) * H_out + oh) * W_out + ow] = sum;
                }
            }
        }
    }
}