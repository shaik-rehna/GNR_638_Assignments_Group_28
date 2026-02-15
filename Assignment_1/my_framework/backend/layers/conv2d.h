#pragma once
#include <memory>
#include <vector>
#include "../core/tensor.h"

class Conv2D {
public:
    std::shared_ptr<Tensor> weight;
    std::shared_ptr<Tensor> bias;

    int in_channels;
    int out_channels;
    int kernel_size;
    int stride;
    int padding;

    Conv2D(int in_channels,
           int out_channels,
           int kernel_size,
           int stride = 1,
           int padding = 0);

    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input);

    std::vector<std::shared_ptr<Tensor>> parameters();
};