#pragma once
#include <vector>
#include <memory>
#include "../core/tensor.h"

class SGD {
public:
    std::vector<std::shared_ptr<Tensor>> params;
    float lr;

    SGD(const std::vector<std::shared_ptr<Tensor>>& parameters,
        float learning_rate);

    void step();
    void zero_grad();
};