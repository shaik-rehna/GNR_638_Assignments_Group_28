import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import matplotlib.pyplot as plt

def plot_train_val_acc_curves_and_save(history, model_name, BASE_DIR):

  plt.figure(figsize=(12,6))

  plt.plot(history["train_acc"], label="Train Accuracy")
  plt.plot(history["val_acc"], label="Validation Accuracy")

  plt.xlabel("Epoch")
  plt.ylabel("Accuracy")
  plt.title("Linear Probe Transfer")
  plt.legend()

  plt.xticks(range(len(history["train_acc"])))

  plot_dir = os.path.join(BASE_DIR, "plots")
  model_dir = os.path.join(plot_dir, model_name)
  plt.savefig(os.path.join(model_dir, f"{model_name}_4.1_train_val_accuracy_curves.png"))

  plt.show()

# Collect Predictions(for confusion matrix and accuracy)
def get_predictions(model, loader, device):

    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)

            outputs = model(images)

            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    return np.array(all_preds), np.array(all_labels)



# Confusion Matrix
def compute_confusion_matrix(preds, labels):

    cm = confusion_matrix(labels, preds)

    return cm

# Plot confusion matrix
def plot_confusion_matrix(cm, class_names):

    plt.figure(figsize=(10,8))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()

    plt.xticks(range(len(class_names)), class_names, rotation=90)
    plt.yticks(range(len(class_names)), class_names)

    plt.xlabel("Predicted")
    plt.ylabel("True")

    plt.tight_layout()
    plt.show()


# confusion matrix
def plot_confusion_matrix_and_save(model, val_loader, device, BASE_DIR, model_name, dataset):

    print("Check if image is saved correctly in the google drive.If not save manually")
    preds, labels = get_predictions(model, val_loader, device)
    cm = compute_confusion_matrix(preds, labels)

    plot_dir = os.path.join(BASE_DIR, "plots")
    model_dir = os.path.join(plot_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)

    fig = plot_confusion_matrix(cm, dataset.classes)  # capture return value
    if fig is None:
        fig = plt.gcf()  # fallback
    fig.canvas.draw()  # force render
    fig.savefig(os.path.join(model_dir, f"{model_name}_4.1_confusion_matrix.png"))
    plt.show()

# Feature extraction
def extract_features(model, loader, device):

    model.eval()

    features = []
    labels = []

    with torch.no_grad():

        for images, y in loader:

            images = images.to(device)

            feats = model.forward_features(images)

            feats = torch.flatten(feats, 1)

            features.append(feats.cpu())
            labels.append(y)

    features = torch.cat(features).numpy()
    labels = torch.cat(labels).numpy()

    return features, labels

# PCA Visualization
def plot_pca(features, labels, BASE_DIR, model_name):

    pca = PCA(n_components=2)

    reduced = pca.fit_transform(features)

    plt.figure(figsize=(8,6))

    scatter = plt.scatter(
        reduced[:,0],
        reduced[:,1],
        c=labels,
        cmap="tab20",
        s=10
    )

    plt.title("PCA Feature Visualization")

    plot_dir = os.path.join(BASE_DIR, "plots")
    model_dir = os.path.join(plot_dir, model_name)
    plt.savefig(os.path.join(model_dir, f"{model_name}_4.1_pca_plot.png"))

    plt.show()

# t-SNE Visualization
def plot_tsne(features, labels,BASE_DIR,model_name):

    tsne = TSNE(n_components=2, perplexity=30)

    reduced = tsne.fit_transform(features)

    plt.figure(figsize=(8,6))

    plt.scatter(
        reduced[:,0],
        reduced[:,1],
        c=labels,
        cmap="tab20",
        s=10
    )

    plt.title("t-SNE Feature Visualization")

    plot_dir = os.path.join(BASE_DIR, "plots")
    model_dir = os.path.join(plot_dir, model_name)
    plt.savefig(os.path.join(model_dir, f"{model_name}_4.1_tsne_plot.png"))

    plt.show()

# UMAP Visualization
def plot_umap(features, labels, BASE_DIR, model_name):

    reducer = umap.UMAP(n_components=2)

    reduced = reducer.fit_transform(features)

    plt.figure(figsize=(8,6))

    plt.scatter(
        reduced[:,0],
        reduced[:,1],
        c=labels,
        cmap="tab20",
        s=10
    )

    plt.title("UMAP Feature Visualization")

    plot_dir = os.path.join(BASE_DIR, "plots")
    model_dir = os.path.join(plot_dir, model_name)
    plt.savefig(os.path.join(model_dir, f"{model_name}_4.1_umap_plot.png"))

    plt.show()


def plot_train_val_acc_percentage_param(histories,percent_unfrozen,BASE_DIR,model_name):
    

    strategies = list(histories.keys())

    val_acc = [histories[s]["val_acc"][-1] for s in strategies]
    train_acc = [histories[s]["train_acc"][-1] for s in strategies]
    params = [percent_unfrozen[s] for s in strategies]

    plt.figure(figsize=(6,5))

    plt.plot(params, train_acc, marker="o", label="Train Accuracy")
    plt.plot(params, val_acc, marker="o", label="Validation Accuracy")

    plt.xlabel("% Unfrozen Parameters")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Percentage of Unfrozen Parameters")

    plt.legend()
    plt.grid(True)

    plot_dir = os.path.join(BASE_DIR, "plots")
    model_dir = os.path.join(plot_dir, model_name)
    plt.savefig(os.path.join(model_dir, f"{model_name}_4.2_accuracy_vs_params.png"))

    plt.show()

def plot_train_loss_vs_epoch(histories,BASE_DIR,model_name):
    strategies = list(histories.keys())
    plt.figure(figsize=(7,5))

    for s in strategies:
        plt.plot(histories[s]["train_loss"], label=s)

    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Convergence Stability (Training Loss vs Epoch)")

    plt.legend()
    plt.grid(True)

    plot_dir = os.path.join(BASE_DIR, "plots")
    model_dir = os.path.join(plot_dir, model_name)
    plt.savefig(os.path.join(model_dir, f"{model_name}_4.2_convergence_loss.png"))

    plt.show()

def grad_norm_stats(histories,BASE_DIR, model_name):
    strategies = list(histories.keys())
    plt.figure(figsize=(7,5))

    for s in strategies:
      plt.plot(histories[s]["grad_norm"], label=s)

    plt.xlabel("Epoch")
    plt.ylabel("Gradient Norm")
    plt.title("Gradient Norm Statistics Across Training")

    plt.legend()
    plt.grid(True)

    plot_dir = os.path.join(BASE_DIR, "plots")
    model_dir = os.path.join(plot_dir, model_name)
    plt.savefig(os.path.join(model_dir, f"{model_name}_4.2_grad_norm_stats.png"))
    plt.show()

def summary_table(histories, percent_unfrozen,BASE_DIR,model_name):
    
    strategies = list(histories.keys())
    rows = []

    for s in strategies:
        rows.append({
            "strategy": s,
            "%params": percent_unfrozen[s],
            "final_val_acc": histories[s]["val_acc"][-1],
            "final_train_acc": histories[s]["train_acc"][-1]
        })

    df = pd.DataFrame(rows)

    results_dir = os.path.join(BASE_DIR, "results")
    model_dir = os.path.join(results_dir, model_name)
    df.to_csv(os.path.join(model_dir, "4.2_finetuning_results.csv"), index=False)

    return df

def plot_val_acc_across_epochs(fewshot_results,BASE_DIR,model_name):

    plt.figure(figsize=(7,5))

    for frac, history in fewshot_results.items():

        val_acc = history["val_acc"]
        epochs = range(1, len(val_acc)+1)

        plt.plot(epochs, val_acc, marker="o", label=f"{int(frac*100)}% data")

    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.title("Few-Shot Learning: Validation Accuracy Across Epochs")

    plt.legend()
    plt.grid(True)

    plot_dir = os.path.join(BASE_DIR, "plots")
    model_dir = os.path.join(plot_dir, model_name)
    plt.savefig(os.path.join(model_dir, f"{model_name}_4.3_few_shot_val_acc_across_epochs.png"))
    plt.show()

def train_val_acc_gap_acroos_epochs(fewshot_results,BASE_DIR,model_name):

    # Plot gap
    plt.figure(figsize=(7,5))

    for frac, history in fewshot_results.items():

        train_acc = history["train_acc"]
        val_acc = history["val_acc"]

        gap = [t - v for t, v in zip(train_acc, val_acc)]
        epochs = range(1, len(gap)+1)

        plt.plot(epochs, gap, marker="o", label=f"{int(frac*100)}% data")

    plt.xlabel("Epoch")
    plt.ylabel("Train–Validation Accuracy Gap")
    plt.title("Overfitting Gap Across Epochs (Few-Shot)")
    plt.legend()
    plt.grid(True)

    plot_dir = os.path.join(BASE_DIR, "plots")
    model_dir = os.path.join(plot_dir, model_name)
    plt.savefig(os.path.join(model_dir, f"{model_name}_4.3_train_val_acc_gap_acroos_epochs.png"))
    plt.show()

def plot_acc_vs_data_frac(fewshot_results,data_sizes,accs,BASE_DIR,model_name):
  

  acc_100 = fewshot_results[1.0]["val_acc"][-1]
  acc_20  = fewshot_results[0.2]["val_acc"][-1]
  acc_5   = fewshot_results[0.05]["val_acc"][-1]
  accs = [acc_100, acc_20, acc_5]

  plt.plot(data_sizes, accs, marker='o')
  plt.xlabel("Training Data (%)")
  plt.ylabel("Validation Accuracy")
  plt.title("Few-Shot Learning Performance")
  plt.grid(True)

  plot_dir = os.path.join(BASE_DIR, "plots")
  model_dir = os.path.join(plot_dir, model_name)
  plt.savefig(os.path.join(model_dir, f"{model_name}_4.3_fewshot_val_acc_vs_data_frac.png"))
  plt.show()


