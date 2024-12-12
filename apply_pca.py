from matplotlib import pyplot as plt
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import yaml


def pca(features, n_components=10, std_scale=False):
    sci_pca = PCA(n_components=n_components)
    if std_scale:
        features = StandardScaler().fit_transform(features)
    result_pca = sci_pca.fit_transform(features)
    return result_pca


def calculate_PCAs(train_data, components_to_use):
    train_var = sum(np.var(train_data["images"], axis=0))

    var_kept_pca = []
    for n_components in components_to_use:
        pca_save_path = f"train_data/pca_{n_components}components.npz"
        if os.path.exists(pca_save_path):
            train_pca = np.load(pca_save_path)["images"]
        else:
            train_pca = pca(train_data["images"], n_components)
            np.savez(pca_save_path, images=train_pca, labels=train_data["labels"])

        pca_var = np.var(train_pca, axis=0)
        var_kept_per_comp = pca_var / train_var
        var_kept_pca.append(var_kept_per_comp)        
        print(f"{n_components} components: {sum(var_kept_per_comp)} of variance maintained")
    return var_kept_pca


def plot_var_by_components(ax, var_kept):
    component_labels = [f"#{i + 1}" for i in range(len(var_kept))]
    ax.bar(component_labels, var_kept)
    if len(component_labels) > 10:
        ax.tick_params(axis="x", labelbottom=False)
    ax.set_xlabel("Individual PCA Components", fontsize=12)
    ax.set_ylabel("Amount of Variance Captured", fontsize=12)
    ax.set_title("Variance Captured Per Component", fontsize=15)


def plot_pca_var_kept(var_kept_pca):
    fig_total, ax_total = plt.subplots(figsize=(max(len(var_kept_pca) / 8, 6), 5), constrained_layout=True)    
    comps_to_plot = [len(var_kept) for var_kept in var_kept_pca]
    ax_total.plot(comps_to_plot, [sum(var_kept) for var_kept in var_kept_pca])
    ax_total.set_xlabel("Number of PCA components", fontsize=15)
    ax_total.xaxis.set_ticks(np.linspace(min(comps_to_plot), max(comps_to_plot),
                                         len(comps_to_plot) // 5 if len(comps_to_plot) > 10 else len(comps_to_plot), dtype=int))
    ax_total.tick_params(labelsize=12)
    ax_total.set_ylabel("Amount of Total Variance Maintained", fontsize=15)
    ax_total.set_title("Variance Maintained Under PCA", fontsize=18)
    figs_by_component = []
    for var_kept in var_kept_pca:
        fig, ax = plt.subplots(constrained_layout=True)
        plot_var_by_components(ax, var_kept)
        figs_by_component.append(fig)
    return fig_total, figs_by_component


if __name__ == "__main__":
    with open("experiment_params.yaml", "r") as params:
        pca_params = yaml.safe_load(params)["PCA"]
    components_to_use = range(*pca_params["components_to_use"])
    train_data = np.load("train_data/collected_images.npz")

    var_kept_pca = calculate_PCAs(train_data, components_to_use)
    fig_total, figs_by_component = plot_pca_var_kept(var_kept_pca)
    fig_total.savefig("figures/PCA/PCA_Total_Variance_Maintained.png")
    for fig, n_component in zip(figs_by_component, components_to_use):
        fig.savefig(f"figures/PCA/Variance_Captured/PCA_Variance_Captured_{n_component}Components.png")
