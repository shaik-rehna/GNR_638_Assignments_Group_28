import torch
import random
import numpy as np
from ptflops import get_model_complexity_info

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Count Total Parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

# Count Trainable Parameters
def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Compute FLOPs and MACs
def compute_model_complexity(model, input_size=(3,224,224)):

    macs, params = get_model_complexity_info(
        model,
        input_size,
        as_strings=False,
        print_per_layer_stat=False
    )

    flops = macs * 2

    return macs, flops, params

# Pretty Print Function
def print_model_stats(model, model_name):

    macs, flops, params = compute_model_complexity(model)

    print(f"Model Efficiency Metrics: {model_name}")
    print("------------------------")
    print(f"Parameters: {params/1e6:.2f} M")
    print(f"MACs: {macs/1e9:.2f} G")
    print(f"FLOPs: {flops/1e9:.2f} G")