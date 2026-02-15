# --------------------------------------------------
# Parameter Count
# --------------------------------------------------

def count_parameters(model):
    total = 0
    for p in model.parameters():
        total += p.numel()
    return total


# --------------------------------------------------
# MAC Computation
# --------------------------------------------------

def compute_macs(model, input_size=32):

    macs = 0

    # ----- Conv1 -----
    w1 = model.conv1.parameters()[0]
    w1_shape = w1.shape

    out_channels1 = w1_shape[0]
    in_channels1 = w1_shape[1]
    kernel1 = w1_shape[2]

    out1 = input_size - kernel1 + 1

    macs += (
        out1 * out1 *
        in_channels1 *
        kernel1 * kernel1 *
        out_channels1
    )

    # ----- Conv2 -----
    w2 = model.conv2.parameters()[0]
    w2_shape = w2.shape

    out_channels2 = w2_shape[0]
    in_channels2 = w2_shape[1]
    kernel2 = w2_shape[2]

    out2 = out1 - kernel2 + 1

    macs += (
        out2 * out2 *
        in_channels2 *
        kernel2 * kernel2 *
        out_channels2
    )

    # ----- After Pool -----
    pooled = out2 // 2
    flatten_dim = out_channels2 * pooled * pooled

    # ----- FC1 -----
    w_fc1 = model.fc1.parameters()[0]
    macs += w_fc1.shape[0] * w_fc1.shape[1]

    # ----- FC2 -----
    w_fc2 = model.fc2.parameters()[0]
    macs += w_fc2.shape[0] * w_fc2.shape[1]

    return macs


# --------------------------------------------------
# FLOPs
# --------------------------------------------------

def compute_flops(macs):
    return 2 * macs


# --------------------------------------------------
# Accuracy
# --------------------------------------------------

def compute_accuracy(outputs, targets, num_classes):

    correct = 0
    B = len(targets)

    for b in range(B):

        max_score = -1e9
        pred_class = -1

        for c in range(num_classes):
            idx = b * num_classes + c
            val = outputs.data[idx]
            if val > max_score:
                max_score = val
                pred_class = c

        if pred_class == targets[b]:
            correct += 1

    return correct