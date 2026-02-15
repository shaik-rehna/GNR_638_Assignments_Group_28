#include "activation.h"

std::shared_ptr<Tensor> relu(std::shared_ptr<Tensor> a) {

    std::vector<float> result(a->numel());

    for (int i = 0; i < a->numel(); ++i)
        result[i] = (a->data[i] > 0.0f) ? a->data[i] : 0.0f;

    auto out = std::make_shared<Tensor>(
        result,
        a->shape,
        a->requires_grad
    );

    out->parents = {a};

    out->backward_fn = [a, out]() {

        if (a->requires_grad) {
            for (int i = 0; i < a->numel(); ++i) {
                if (a->data[i] > 0.0f)
                    a->grad[i] += out->grad[i];
            }
        }

        //FREE GRAPH
        out->parents.clear();
        out->backward_fn = nullptr;
    };

    return out;
}