#pragma once
#include <memory>
#include "../core/tensor.h"

std::shared_ptr<Tensor> relu(std::shared_ptr<Tensor> a);