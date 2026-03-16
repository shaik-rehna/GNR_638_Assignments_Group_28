import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

# Feature Extraction From Specific Layers
# Forward hooks to capture intermediate activations
def extract_layer_features(model, loader, device, layer):

    features = []
    labels = []

    def hook_fn(module, input, output):

        # Global Average Pooling
        pooled = torch.mean(output, dim=[2,3])

        features.append(pooled.detach().cpu())

    hook = layer.register_forward_hook(hook_fn)

    model.eval()

    with torch.no_grad():

        for images, y in loader:

            images = images.to(device)

            _ = model(images)

            labels.append(y.cpu())

    hook.remove()

    features = torch.cat(features)
    labels = torch.cat(labels)

    return features.numpy(), labels.numpy()
    

# def extract_layer_features(model, loader, device, layer):

#     features = []
#     labels = []

#     def hook_fn(module, input, output):

#         out = output.detach()

#         # If convolutional output (B,C,H,W)
#         if out.dim() == 4:
#             out = torch.mean(out, dim=[2,3])

#         # If already flattened (B,C)
#         elif out.dim() == 2:
#             pass

#         else:
#             out = out.flatten(1)

#         features.append(out.cpu())

#     hook = layer.register_forward_hook(hook_fn)

#     model.eval()

#     with torch.no_grad():

#         for images, y in loader:

#             images = images.to(device)

#             _ = model(images)

#             labels.append(y.cpu())

#     hook.remove()

#     if len(features) == 0:
#         raise RuntimeError("No features captured. Hook layer may be incorrect.")

#     features = torch.cat(features)
#     labels = torch.cat(labels)

#     return features.numpy(), labels.numpy()

# Train Linear Probe on Features
# Instead of deep learning, logistic regression is used
# This is faster and common in representation probing
def train_linear_probe(train_features,
                       train_labels,
                       val_features,
                       val_labels):

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=5000,
            solver="lbfgs",
            n_jobs=-1
        )
    )

    clf.fit(train_features, train_labels)

    val_acc = clf.score(val_features, val_labels)

    return val_acc

# Feature Norm Statistics across layers
def compute_feature_norms(features):

    norms = np.linalg.norm(features, axis=1)

    mean_norm = np.mean(norms)
    std_norm = np.std(norms)

    return mean_norm, std_norm


def plot_pca(features, labels, title):

    pca = PCA(n_components=2)

    reduced = pca.fit_transform(features)

    plt.figure(figsize=(6,6))

    scatter = plt.scatter(
        reduced[:,0],
        reduced[:,1],
        c=labels,
        cmap="tab20",
        s=10
    )

    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")

    plt.colorbar()

    plt.show()

def sample_fixed_subset(labels, samples_per_class=30,seed=42):

    np.random.seed(seed)

    selected_idx = []

    classes = np.unique(labels)

    for c in classes:

        idx = np.where(labels == c)[0]

        # If class has fewer samples than required
        replace_flag = len(idx) < samples_per_class

        chosen = np.random.choice(idx, samples_per_class, replace=replace_flag)

        selected_idx.extend(chosen)

    return np.array(selected_idx)


def plot_acc_vs_depth(depth_acc,BASE_DIR,model_name):
    depths = ["Early", "Middle", "Final"]

    plt.plot(depths, depth_acc, marker='o')

    plt.xlabel("Network Depth")
    plt.ylabel("Validation Accuracy")
    plt.title("Layer-wise Representation Quality")

    plot_dir = os.path.join(BASE_DIR, "plots")
    model_dir = os.path.join(plot_dir, model_name)
    plt.savefig(os.path.join(model_dir, f"{model_name}_4.5_acc_vs_depth.png"))
    plt.show()

def plot_feature_norms(norm_stats,BASE_DIR, model_name):
    

    df = pd.DataFrame(
        norm_stats,
        columns=["mean_norm","std_norm"],
        index=["Early","Middle","Final"]
    )

    print(df)

    results_dir = os.path.join(BASE_DIR, "results")
    model_dir = os.path.join(results_dir, model_name)
    df.to_csv(os.path.join(model_dir, "4.5_feature_norms.csv"), index=False)

# Layer names differ by architecture
# ResNet50
# early_layer = model.layer1
# middle_layer = model.layer3
# final_layer = model.layer4

# DenseNet121
# early_layer = model.features.denseblock1
# middle_layer = model.features.denseblock3
# final_layer = model.features.denseblock4

#Efficientnet-B0
# early_layer = model.blocks[1]
# middle_layer = model.blocks[4]
# final_layer = model.blocks[-1]

# from feature_probe import *

# train_feats, train_labels = extract_layer_features(
#     model,
#     train_loader,
#     device,
#     model.layer3
# )

# val_feats, val_labels = extract_layer_features(
#     model,
#     val_loader,
#     device,
#     model.layer3
# )

# acc = train_linear_probe(
#     train_feats,
#     train_labels,
#     val_feats,
#     val_labels
# )

# print("Probe accuracy:", acc)

# depths = ["Early", "Middle", "Final"]
# accs = [acc1, acc2, acc3]

# plt.plot(depths, accs, marker='o')
# plt.xlabel("Layer Depth")
# plt.ylabel("Validation Accuracy")
# plt.title("Layer-wise Feature Quality")
# plt.show()