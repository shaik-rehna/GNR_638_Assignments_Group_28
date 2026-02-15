#include "maxpool2d.h"
#include <limits>

MaxPool2D::MaxPool2D(int kernel_size, int stride)
    : kernel_size(kernel_size), stride(stride) {}

std::shared_ptr<Tensor> MaxPool2D::forward(std::shared_ptr<Tensor> input) {

    int batch = input->shape[0];
    int channels = input->shape[1];
    int in_h = input->shape[2];
    int in_w = input->shape[3];

    int out_h = (in_h - kernel_size) / stride + 1;
    int out_w = (in_w - kernel_size) / stride + 1;

    std::vector<float> output_data(
        batch * channels * out_h * out_w,
        0.0f
    );

    // store max indices for backward
    std::vector<int> max_indices(
        batch * channels * out_h * out_w,
        0
    );

    for (int b = 0; b < batch; ++b) {
        for (int c = 0; c < channels; ++c) {
            for (int oh = 0; oh < out_h; ++oh) {
                for (int ow = 0; ow < out_w; ++ow) {

                    float max_val = -std::numeric_limits<float>::infinity();
                    int max_index = 0;

                    for (int kh = 0; kh < kernel_size; ++kh) {
                        for (int kw = 0; kw < kernel_size; ++kw) {

                            int ih = oh * stride + kh;
                            int iw = ow * stride + kw;

                            int input_index =
                                b * (channels * in_h * in_w) +
                                c * (in_h * in_w) +
                                ih * in_w + iw;

                            float val = input->data[input_index];

                            if (val > max_val) {
                                max_val = val;
                                max_index = input_index;
                            }
                        }
                    }

                    int out_index =
                        b * (channels * out_h * out_w) +
                        c * (out_h * out_w) +
                        oh * out_w + ow;

                    output_data[out_index] = max_val;
                    max_indices[out_index] = max_index;
                }
            }
        }
    }

    auto out = std::make_shared<Tensor>(
        output_data,
        std::vector<int>{batch, channels, out_h, out_w},
        input->requires_grad
    );

    out->parents = {input};

    out->backward_fn = [=]() {

        if (!input->requires_grad)
            return;

        for (int i = 0; i < max_indices.size(); ++i) {

            int input_index = max_indices[i];

            input->grad[input_index] += out->grad[i];
        }
    };

    return out;
}