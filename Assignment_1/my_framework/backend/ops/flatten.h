#pragma once
#include <memory>
#include "../core/tensor.h"

std::shared_ptr<Tensor> flatten(std::shared_ptr<Tensor> input);
