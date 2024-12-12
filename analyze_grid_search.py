import ast
from functools import partial
from matplotlib import pyplot as plt
import numpy as np
import os
from os.path import join
import pandas as pd
from train_classifier import get_model
from train_cnn import HPARAM_NAME_MAP, VALIDATION_METRICS
import yaml


FEAT_TRANSFORM_NAME = "feature_transformation"
CNN_HPARAM_REVERSE_MAP = {code_name: display_name for display_name, code_name in HPARAM_NAME_MAP.items()} | {FEAT_TRANSFORM_NAME: FEAT_TRANSFORM_NAME}


def filter_and_save_gs_dataframe(tables_path, gs_data, model_type, is_cnn):
    columns_to_drop = [name for name in gs_data.columns if "time" in name or "rank" in name or "split" in name]
    gs_data.drop(columns_to_drop, axis=1, inplace=True)
    
    def decode_params(params_str, is_cnn):
        params_dict = ast.literal_eval(params_str)
        decoded_params = [f'{CNN_HPARAM_REVERSE_MAP[param] if is_cnn else param}={value}' for param, value in params_dict.items()]
        return ", ".join(decoded_params)
    
    gs_data["params"] = gs_data["params"].apply(partial(decode_params, is_cnn=is_cnn))
    gs_data[f"param_{FEAT_TRANSFORM_NAME}"] = gs_data[f"param_{FEAT_TRANSFORM_NAME}"].apply(lambda feat_transform: feat_transform.strip("_"))
    columns = list(gs_data)
    for col_to_move in ["params", f"param_{FEAT_TRANSFORM_NAME}"]:
        columns.insert(0, columns.pop(columns.index(col_to_move)))
    gs_data = gs_data.loc[:, columns]
    gs_data.rename(columns={f"param_{FEAT_TRANSFORM_NAME}": "feature transformations", "params": "hyperparameters"}, inplace=True)
    param_cols = [name for name in gs_data.columns if "param_" in name and name != "params"]
    gs_data.rename(columns={param_col: CNN_HPARAM_REVERSE_MAP[param_col.removeprefix("param_")] if is_cnn else param_col.removeprefix("param_") for param_col in param_cols}, inplace=True)

    train_columns, test_columns = [name for name in gs_data.columns if "train" in name], [name for name in gs_data.columns if "test" in name]
    gs_test, gs_train = gs_data.drop(train_columns, axis=1), gs_data.drop(test_columns, axis=1)
    gs_test.rename(columns={test_col: test_col.replace("test_", "") for test_col in test_columns}, inplace=True)
    gs_train.rename(columns={train_col: train_col.replace("train_", "") for train_col in train_columns}, inplace=True)
    gs_test.to_csv(join(tables_path, f"{model_type}_merged_grid_search_test.csv"), index=False)
    gs_train.to_csv(join(tables_path, f"{model_type}_merged_grid_search_train.csv"), index=False)


def visualize_single_param_change(param_ind, params_fixed, metrics_to_plot, train_test, gs_data, is_cnn):
    param_ind_display = CNN_HPARAM_REVERSE_MAP[param_ind].capitalize() if is_cnn else param_ind.capitalize()
    gs_params = [name.removeprefix("param_") for name in gs_data.columns if "param" in name and name != "params"]
    if len(uncovered_params := set(gs_params) - set([param_ind] + list(params_fixed.keys()))):
        raise ValueError(f"Not all parameters in provided dataframe is covered, missing {uncovered_params}")
    fixed_param_query = " & ".join([f"param_{fixed_param} == {f"""'{fixed_value}'""" if isinstance(fixed_value, str) else fixed_value}"
                                    if fixed_value is not None else f"{fixed_value}.isna()"
                                    for fixed_param, fixed_value in params_fixed.items()])
    isol_df = gs_data.query(fixed_param_query)
    ind_values = isol_df[f"param_{param_ind}"].replace(r"^\s*$", "None", regex=True).to_numpy()
    fig, ax = plt.subplots(figsize=(max(8, len(ind_values)), max(6, len(metrics_to_plot) / 2)), constrained_layout=True)
    color = iter(plt.cm.rainbow(np.linspace(0, 1, len(metrics_to_plot))))
    for metric_name in metrics_to_plot:
        c = next(color)
        metric_mean, metric_std = isol_df[f"mean_{train_test}_{metric_name}"].to_numpy(), isol_df[f"std_{train_test}_{metric_name}"].to_numpy()
        ax.plot(ind_values, metric_mean, c=c, label=metric_name.capitalize())
        metric_upper, metric_lower = metric_mean + metric_std, metric_mean - metric_std
        ax.fill_between(ind_values if np.issubdtype(ind_values.dtype, np.number) else range(len(ind_values)), metric_upper, metric_lower, facecolor=c, alpha=0.3)
    ax.legend()
    if param_ind != FEAT_TRANSFORM_NAME:
        ax.set_xlabel(param_ind_display, fontsize=15)
    ax.set_ylabel("Scores of Various Metrics", fontsize=15)
    ax.tick_params(labelsize=12)
    ax.set_title(f"Fixed hyperparameters: {', '.join([f'{CNN_HPARAM_REVERSE_MAP[param] if is_cnn else param}={value}' for param, value in params_fixed.items()])}", fontsize=11)
    fig.suptitle(f"Performance vs. Different {'Feature Transformations' if param_ind == FEAT_TRANSFORM_NAME else f'values of {param_ind_display}'}", fontsize=18)
    return fig


def plot_parameter_changes_and_merge_gs_data(model_arches, model_metrics, best_params, results_path, tables_path, figures_path, is_cnn):
    for model_arch, model_names in model_arches.items():
        gs_merged_df, arch_params = [], {}
        for model_name in model_names:
            gs_data_path = join(results_path, model_name, "grid_search_results.csv")
            grid_search_data = pd.read_csv(gs_data_path, comment="#")
            arch_params[model_name] = best_params[model_name]

            param_change_figs_path = join(figures_path, model_name)
            os.makedirs(param_change_figs_path, exist_ok=True)
            for param_name in arch_params[model_name]:
                params_fixed = {fixed_name: fixed_value for fixed_name, fixed_value in arch_params[model_name].items() if fixed_name != param_name}
                param_change_fig = visualize_single_param_change(param_name, params_fixed, model_metrics[model_arch], "test", grid_search_data, is_cnn)
                param_display = CNN_HPARAM_REVERSE_MAP[param_name].capitalize() if is_cnn else param_name.capitalize()
                param_change_fig.savefig(join(param_change_figs_path, f"{param_display}_change_wrt_best_params.png"))
                plt.close()
            
            gs_to_merge = grid_search_data.copy()
            gs_to_merge[f"param_{FEAT_TRANSFORM_NAME}"] = model_name[model_name.index("_") + 1:] if "_" in model_name else model_name
            gs_merged_df.append(gs_to_merge)
        gs_merged_df = pd.concat(gs_merged_df)
        for model_name, params in arch_params.items():
            transform_change_fig = visualize_single_param_change(FEAT_TRANSFORM_NAME, params, model_metrics[model_arch], "test", gs_merged_df, is_cnn)
            transform_change_figs_path = join(figures_path, model_name)
            transform_change_fig.savefig(join(transform_change_figs_path, f"Feature_transformations_under_best_params.png"))
            plt.close()
        filter_and_save_gs_dataframe(tables_path, gs_merged_df, model_arch, is_cnn)


def read_cnn_info(params, results_path):
    cnn_arches, cnn_metrics, best_params = {}, {}, {}
    for arch_entry in params["CNN"]["architecture"]:
        arch = list(arch_entry.keys())[0]
        variant = arch_entry[arch]
        if arch == "resnet" or arch == "vgg":
            cnn_name = f"{arch}{variant['num_layers']}"
            if arch in cnn_arches:
                cnn_arches[arch].append(cnn_name)
            else:
                cnn_arches[arch] = [cnn_name]
            cnn_metrics[arch] = VALIDATION_METRICS + ["accuracy"]
            best_param_path = join(results_path, cnn_name, "grid_search_results.csv")
            with open(best_param_path, "r") as best_param_file:
                param_line = best_param_file.readline()
            best_params[cnn_name] = ast.literal_eval(param_line[param_line.index("{") : param_line.index("}") + 1])
        else:
            raise NotImplementedError(f"Model architecture {arch} is not implemented yet.")
    return cnn_arches, cnn_metrics, best_params


def read_classifier_info(params, classifier_types, results_path):
    cls_arches, cls_metrics, best_params = {}, {}, {}
    for cls_type in classifier_types:
        cls_arches[cls_type] = []
        for transform_group in params[cls_type]["transform_groups"]:
            cls_name = f"{cls_type}_{'_'.join(transform_group)}"
            cls_arches[cls_type].append(cls_name)
            best_param_path = join(results_path, cls_name, "grid_search_metrics.txt")
            with open(best_param_path, "r") as best_param_file:
                param_line = best_param_file.readline()
            best_params[cls_name] = ast.literal_eval(param_line[param_line.index("{"):])
        cls_metrics[cls_type] = get_model(cls_type, {})[2]
    return cls_arches, cls_metrics, best_params


if __name__ == "__main__":
    results_path, tables_path, figures_path = "results", "tables", "figures"
    os.makedirs(tables_path, exist_ok=True)
    classifier_types = ["Logistic", "SVM_RBF"]
    test_data_path = join("test_data", "collected_images.npz")
    with open("experiment_params.yaml", "r") as exp_params:
        params = yaml.safe_load(exp_params)
    
    plot_parameter_changes_and_merge_gs_data(*read_cnn_info(params, results_path), results_path, tables_path, figures_path, True)
    plot_parameter_changes_and_merge_gs_data(*read_classifier_info(params, classifier_types, results_path), results_path, tables_path, figures_path, False)
