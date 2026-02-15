#pragma once
#include <memory>
#include "../core/tensor.h"

class MaxPool2D {
public:
    int kernel_size;
    int stride;

    MaxPool2D(int kernel_size, int stride);

    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input);
};