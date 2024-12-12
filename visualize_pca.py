from apply_pca import pca
from matplotlib import pyplot as plt
import seaborn as sns
import os
import numpy as np
import yaml


def plot_feature_distribution(features, feature_names, save_dir="figures/PCA/distributions"):
    os.makedirs(save_dir, exist_ok=True) 

    for i, feature_name in enumerate(feature_names):
        plt.figure(figsize=(8, 4), constrained_layout=True)

        hist_series = sns.histplot(features[:, i], bins=30, kde=False, color='blue', alpha=0.7, label="Histogram")
        density_ax = plt.twinx()
        kde_series = sns.kdeplot(features[:, i], color='red', label="Density", fill=True, alpha=0.4, ax=density_ax)
        series_handles = hist_series.get_legend_handles_labels()[0] + kde_series.get_legend_handles_labels()[0]

        plt.xlabel(feature_name)
        plt.ylabel("Frequency / Density")
        plt.title(f"Distribution of {feature_name}")
        plt.legend(handles=series_handles)
        plt.grid(True)
        
        save_path = os.path.join(save_dir, f"{feature_name}_distribution.png")
        plt.savefig(save_path)
        print(f"Saved distribution plot: {save_path}")
        plt.close()


def plot_feature_vs_target(features, target, feature_names, save_dir="figures/PCA/relationships"):
    os.makedirs(save_dir, exist_ok=True)  

    for i, feature_name in enumerate(feature_names):
        plt.figure(figsize=(8, 4), constrained_layout=True)
        plt.scatter(features[:, i], target, s=10, alpha=0.7, color="green")
        plt.xlabel(feature_name)
        plt.ylabel("Target")
        plt.title(f"{feature_name} vs Target")
        plt.grid(True)
        
        save_path = os.path.join(save_dir, f"{feature_name}_vs_target.png")
        plt.savefig(save_path)
        print(f"Saved feature vs target plot: {save_path}")
        plt.close()


def plot_correlation_matrix(features, feature_names, save_path="figures/PCA/correlation_matrix.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True) 

    correlation_matrix = np.corrcoef(features.T)

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        correlation_matrix,
        xticklabels=feature_names, 
        yticklabels=feature_names, 
        annot=True, 
        fmt=".2f", 
        cmap="coolwarm", 
        cbar=True
    )
    plt.title("PCA Features Correlation Matrix", fontsize=18)
    plt.tight_layout()
    
    plt.savefig(save_path)
    print(f"Saved correlation matrix: {save_path}")
    plt.close()


if __name__ == "__main__":
    with open("experiment_params.yaml", "r") as params:
        pca_params = yaml.safe_load(params)["PCA"]

    components_to_use = range(*pca_params["components_to_use"])
    train_data = np.load("train_data/collected_images.npz")

    features_pca = []
    for n_components in components_to_use:
        pca_save_path = f"train_data/pca_{n_components}components.npz"
        if os.path.exists(pca_save_path):
            train_pca = np.load(pca_save_path)["images"]
        else:
            train_pca = pca(train_data["images"], n_components)
            np.savez(pca_save_path, images=train_pca, labels=train_data["labels"])
    
    if isinstance(train_pca, np.ndarray) and train_pca.ndim == 2:
        features_pca.append(train_pca)
    else:
        print(f"Invalid PCA result for {n_components} components: {type(train_pca)}, {train_pca.shape}")


    target = train_data["labels"]
    feature_names = [f"PCA Component {i+1}" for i in range(n_components)]

    plot_feature_distribution(features_pca[0], feature_names)

    plot_feature_vs_target(features_pca[0], target, feature_names)

    plot_correlation_matrix(features_pca[0], feature_names)
