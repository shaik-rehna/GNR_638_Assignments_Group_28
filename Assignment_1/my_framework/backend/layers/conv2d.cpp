#include "conv2d.h"
#include <cstdlib>
#include <cmath>

// --------------------------------------------------
// Constructor
// --------------------------------------------------
Conv2D::Conv2D(int in_channels,
               int out_channels,
               int kernel_size,
               int stride,
               int padding)
    : in_channels(in_channels),
      out_channels(out_channels),
      kernel_size(kernel_size),
      stride(stride),
      padding(padding)
{
    std::vector<float> w_data(
        out_channels * in_channels * kernel_size * kernel_size
    );

    // -------- He Initialization --------
    float fan_in = in_channels * kernel_size * kernel_size;
    float scale = std::sqrt(2.0f / fan_in);

    for (auto &v : w_data) {
        float r = (float)std::rand() / RAND_MAX;  // [0,1]
        r = r * 2.0f - 1.0f;                      // [-1,1]
        v = r * scale;
    }
    // -----------------------------------

    weight = std::make_shared<Tensor>(
        w_data,
        std::vector<int>{
            out_channels,
            in_channels,
            kernel_size,
            kernel_size
        },
        true
    );

    std::vector<float> b_data(out_channels, 0.0f);

    bias = std::make_shared<Tensor>(
        b_data,
        std::vector<int>{out_channels},
        true
    );
}

// --------------------------------------------------
// Forward
// --------------------------------------------------
std::shared_ptr<Tensor> Conv2D::forward(std::shared_ptr<Tensor> input) {

    int batch = input->shape[0];
    int in_ch = input->shape[1];
    int in_h = input->shape[2];
    int in_w = input->shape[3];

    int out_h = (in_h - kernel_size + 2 * padding) / stride + 1;
    int out_w = (in_w - kernel_size + 2 * padding) / stride + 1;

    std::vector<float> output_data(
        batch * out_channels * out_h * out_w,
        0.0f
    );

    for (int b = 0; b < batch; ++b) {
        for (int oc = 0; oc < out_channels; ++oc) {
            for (int oh = 0; oh < out_h; ++oh) {
                for (int ow = 0; ow < out_w; ++ow) {

                    float sum = bias->data[oc];

                    for (int ic = 0; ic < in_ch; ++ic) {
                        for (int kh = 0; kh < kernel_size; ++kh) {
                            for (int kw = 0; kw < kernel_size; ++kw) {

                                int ih = oh * stride + kh - padding;
                                int iw = ow * stride + kw - padding;

                                if (ih >= 0 && ih < in_h &&
                                    iw >= 0 && iw < in_w) {

                                    int input_index =
                                        b * (in_ch * in_h * in_w) +
                                        ic * (in_h * in_w) +
                                        ih * in_w + iw;

                                    int weight_index =
                                        oc * (in_ch * kernel_size * kernel_size) +
                                        ic * (kernel_size * kernel_size) +
                                        kh * kernel_size + kw;

                                    sum += input->data[input_index] *
                                           weight->data[weight_index];
                                }
                            }
                        }
                    }

                    int output_index =
                        b * (out_channels * out_h * out_w) +
                        oc * (out_h * out_w) +
                        oh * out_w + ow;

                    output_data[output_index] = sum;
                }
            }
        }
    }

    auto out = std::make_shared<Tensor>(
        output_data,
        std::vector<int>{batch, out_channels, out_h, out_w},
        input->requires_grad || weight->requires_grad
    );

    out->parents = {input, weight, bias};

    out->backward_fn = [=]() {

    // ----- dBias -----
    if (bias->requires_grad) {
        for (int oc = 0; oc < out_channels; ++oc) {

            float grad_sum = 0.0f;

            for (int b = 0; b < batch; ++b)
                for (int oh = 0; oh < out_h; ++oh)
                    for (int ow = 0; ow < out_w; ++ow) {

                        int out_index =
                            b * (out_channels * out_h * out_w) +
                            oc * (out_h * out_w) +
                            oh * out_w + ow;

                        grad_sum += out->grad[out_index];
                    }

            bias->grad[oc] += grad_sum;
        }
    }

    // ----- dWeight -----
    if (weight->requires_grad) {

        for (int oc = 0; oc < out_channels; ++oc) {
            for (int ic = 0; ic < in_ch; ++ic) {
                for (int kh = 0; kh < kernel_size; ++kh) {
                    for (int kw = 0; kw < kernel_size; ++kw) {

                        float grad_sum = 0.0f;

                        for (int b = 0; b < batch; ++b)
                            for (int oh = 0; oh < out_h; ++oh)
                                for (int ow = 0; ow < out_w; ++ow) {

                                    int ih = oh * stride + kh - padding;
                                    int iw = ow * stride + kw - padding;

                                    if (ih >= 0 && ih < in_h &&
                                        iw >= 0 && iw < in_w) {

                                        int input_index =
                                            b * (in_ch * in_h * in_w) +
                                            ic * (in_h * in_w) +
                                            ih * in_w + iw;

                                        int out_index =
                                            b * (out_channels * out_h * out_w) +
                                            oc * (out_h * out_w) +
                                            oh * out_w + ow;

                                        grad_sum +=
                                            input->data[input_index] *
                                            out->grad[out_index];
                                    }
                                }

                        int weight_index =
                            oc * (in_ch * kernel_size * kernel_size) +
                            ic * (kernel_size * kernel_size) +
                            kh * kernel_size + kw;

                        weight->grad[weight_index] += grad_sum;
                    }
                }
            }
        }
    }

    // ----- dInput -----
    if (input->requires_grad) {

        for (int b = 0; b < batch; ++b) {
            for (int ic = 0; ic < in_ch; ++ic) {
                for (int ih = 0; ih < in_h; ++ih) {
                    for (int iw = 0; iw < in_w; ++iw) {

                        float grad_sum = 0.0f;

                        for (int oc = 0; oc < out_channels; ++oc)
                            for (int kh = 0; kh < kernel_size; ++kh)
                                for (int kw = 0; kw < kernel_size; ++kw) {

                                    int oh = ih + padding - kh;
                                    int ow = iw + padding - kw;

                                    if (oh % stride == 0 &&
                                        ow % stride == 0) {

                                        oh /= stride;
                                        ow /= stride;

                                        if (oh >= 0 && oh < out_h &&
                                            ow >= 0 && ow < out_w) {

                                            int out_index =
                                                b * (out_channels * out_h * out_w) +
                                                oc * (out_h * out_w) +
                                                oh * out_w + ow;

                                            int weight_index =
                                                oc * (in_ch * kernel_size * kernel_size) +
                                                ic * (kernel_size * kernel_size) +
                                                kh * kernel_size + kw;

                                            grad_sum +=
                                                weight->data[weight_index] *
                                                out->grad[out_index];
                                            }
                                    }
                                }

                        int input_index =
                            b * (in_ch * in_h * in_w) +
                            ic * (in_h * in_w) +
                            ih * in_w + iw;

                        input->grad[input_index] += grad_sum;
                    }
                }
            }
        }
    }

    // CRITICAL: FREE COMPUTATION GRAPH
    out->parents.clear();
    out->backward_fn = nullptr;
};

    return out;
}

// --------------------------------------------------
// Parameters 
// --------------------------------------------------
std::vector<std::shared_ptr<Tensor>> Conv2D::parameters() {
    return {weight, bias};
}