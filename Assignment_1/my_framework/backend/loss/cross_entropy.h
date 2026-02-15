#pragma once
#include <memory>
#include <vector>
#include "../core/tensor.h"

std::shared_ptr<Tensor> cross_entropy(
    std::shared_ptr<Tensor> logits,
    const std::vector<int>& targets
);