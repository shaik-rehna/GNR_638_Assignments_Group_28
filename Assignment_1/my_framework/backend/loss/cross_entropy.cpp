#include "cross_entropy.h"
#include <cmath>
#include <algorithm>

std::shared_ptr<Tensor> cross_entropy(
    std::shared_ptr<Tensor> logits,
    const std::vector<int>& targets
) {

    int batch_size = logits->shape[0];
    int num_classes = logits->shape[1];

    std::vector<float> softmax(logits->numel());
    float total_loss = 0.0f;

    for (int i = 0; i < batch_size; ++i) {

        float max_logit = -1e9f;
        for (int j = 0; j < num_classes; ++j) {
            float val = logits->data[i * num_classes + j];
            if (val > max_logit)
                max_logit = val;
        }

        float sum_exp = 0.0f;
        for (int j = 0; j < num_classes; ++j) {
            float exp_val = std::exp(
                logits->data[i * num_classes + j] - max_logit
            );
            softmax[i * num_classes + j] = exp_val;
            sum_exp += exp_val;
        }

        for (int j = 0; j < num_classes; ++j)
            softmax[i * num_classes + j] /= sum_exp;

        int target = targets[i];
        total_loss += -std::log(
            softmax[i * num_classes + target] + 1e-9f
        );
    }

    total_loss /= batch_size;

    auto out = std::make_shared<Tensor>(
        std::vector<float>{total_loss},
        std::vector<int>{1},
        logits->requires_grad
    );

    out->parents = {logits};

    std::vector<int> targets_copy = targets;

    out->backward_fn = [logits, softmax, targets_copy,
                        batch_size, num_classes, out]() {

        if (logits->requires_grad) {

            for (int i = 0; i < batch_size; ++i) {
                for (int j = 0; j < num_classes; ++j) {

                    float grad_val = softmax[i * num_classes + j];

                    if (j == targets_copy[i])
                        grad_val -= 1.0f;

                    grad_val /= batch_size;

                    logits->grad[i * num_classes + j] +=
                        grad_val * out->grad[0];
                }
            }
        }

        // FREE GRAPH
        out->parents.clear();
        out->backward_fn = nullptr;
    };

    return out;
}