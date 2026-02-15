#pragma once
#include <memory>
#include <vector>
#include "../core/tensor.h"

class Linear {
public:
    std::shared_ptr<Tensor> weight;
    std::shared_ptr<Tensor> bias;

    int in_features;
    int out_features;

    Linear(int in_features, int out_features);

    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input);

    std::vector<std::shared_ptr<Tensor>> parameters();
};