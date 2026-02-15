#include "sgd.h"

SGD::SGD(const std::vector<std::shared_ptr<Tensor>>& parameters,
         float learning_rate)
    : params(parameters), lr(learning_rate) {}

void SGD::step() {

    for (auto& p : params) {

        if (!p->requires_grad)
            continue;

        for (int i = 0; i < p->numel(); ++i) {
            p->data[i] -= lr * p->grad[i];
        }
    }
}

void SGD::zero_grad() {

    for (auto& p : params) {
        p->zero_grad();
    }
}