import csv
from functools import partial
import joblib
import json
from matplotlib import pyplot as plt
import numpy as np
import os
from os.path import exists, join
from sklearn.metrics import get_scorer
from sklearn.preprocessing import StandardScaler
from skorch.callbacks import Checkpoint
from skorch.classifier import NeuralNetClassifier
from skorch.history import History
from timm.models import create_model
from train_classifier import get_model, prepare_data_with_transformations
from train_cnn import load_dataset, ImageDataset, VALIDATION_METRICS
import torch
import yaml


CIFAR_NUM_CLS = 10


def predict_and_store_if_needed(save_path, model, X):
    predictions_path = join(save_path, "best_model_predictions.npy")
    if not exists(predictions_path):
        y_pred = model.predict(X)
        np.save(predictions_path, y_pred)
        print(f"Saved predictions to {save_path}")
    else:
        print(f"{predictions_path} already exists, skipping")


def predict_probabilities_and_store_if_needed(save_path, model, X):
    predict_proba_path = join(save_path, "best_model_predict_probas.npy")
    if not exists(predict_proba_path):
        y_probas = model.predict_proba(X)
        np.save(predict_proba_path, y_probas)
        print(f"Saved predicted probabilities to {save_path}")
    else:
        print(f"{predict_proba_path} already exists, skipping")


def score_and_store_if_needed(save_path, model, X, y, metrics):
    scores_path = join(save_path, "best_model_test_scores.json")
    if not exists(scores_path):
        scores = {metric: get_scorer(metric)(model, X, y) for metric in metrics}
        with open(scores_path, "w") as score_file:
            json.dump(scores, score_file)
        print(f"Saved testing scores to {save_path}")
    else:
        print(f"{scores_path} already exists, skipping")


def test_classifiers(classifier_params, results_path, test_out_path, test_data_path, classifier_metrics):
    for classifier_type, classifier_params in classifier_params.items():
        for transform_group in classifier_params["transform_groups"]:
            print(f"Testing {classifier_type} with transformations: {transform_group}")
            print(f"Loading test data from {test_data_path}...")
            X_test, y_test = prepare_data_with_transformations(test_data_path, transform_group)
            X_test = StandardScaler().fit_transform(X_test)
            model_name = f"{classifier_type}_{'_'.join(transform_group)}"
            classifier_model = joblib.load(join(results_path, model_name, "best_model.pkl"))
            model_out_path = join(test_out_path, model_name)
            os.makedirs(model_out_path, exist_ok=True)
            predict_and_store_if_needed(model_out_path, classifier_model, X_test)
            if classifier_type == "Logistic":
                predict_probabilities_and_store_if_needed(model_out_path, classifier_model, X_test)
            score_and_store_if_needed(model_out_path, classifier_model, X_test, y_test, classifier_metrics[classifier_type])


def test_cnn(cnn_params, results_path, test_out_path, test_data_path, cnn_metrics):
    for arch_entry in cnn_params["architecture"]:
        arch = list(arch_entry.keys())[0]
        variant = arch_entry[arch]
        if arch == "resnet" or arch == "vgg":
            cnn_name = f"{arch}{variant['num_layers']}"
            print(f"Testing CNN: {cnn_name}")
            ckpt = Checkpoint(dirname=join(results_path, cnn_name))
            cnn_creator = partial(create_model, cnn_name, num_classes=CIFAR_NUM_CLS)
            cnn_model = NeuralNetClassifier(cnn_creator, criterion=torch.nn.CrossEntropyLoss, dataset=ImageDataset, device="cuda", classes=np.arange(10))
            cnn_model.initialize()
            cnn_model.load_params(checkpoint=ckpt)
            print(f"Loading test data from {test_data_path}...")
            X_test, y_test = load_dataset(test_data_path)
            X_test = torch.tensor(X_test.reshape((X_test.shape[0], 3, 32, 32))).float()
            model_out_path = join(test_out_path, cnn_name)
            os.makedirs(model_out_path, exist_ok=True)
            predict_and_store_if_needed(model_out_path, cnn_model, X_test)
            predict_probabilities_and_store_if_needed(model_out_path, cnn_model, X_test)
            score_and_store_if_needed(model_out_path, cnn_model, X_test, y_test, cnn_metrics)
        else:
            raise NotImplementedError(f"Model architecture {arch} is not implemented yet.")


def save_table(model_names, scores, tables_path, score_type):
    with open(join(tables_path, f"best_models_{score_type}_scores.csv"), "w") as table_file:
        writer = csv.writer(table_file, lineterminator="\n")
        writer.writerow([""] + model_names)
        for score_type, model_perfs in scores.items():
            score_for_models = ["" for _ in range(len(model_names))]
            for model_name, perf_score in model_perfs.items():
                score_for_models[model_names.index(model_name)] = perf_score
            writer.writerow([score_type] + score_for_models)


def plot_model_performance_comparisons(test_score, val_score, fig_save_path):
    for score_name, model_perfs in test_score.items():
        fig, ax = plt.subplots(figsize=(len(model_perfs), 6), constrained_layout=True)
        if val_score is not None:
            x_inds = np.arange(len(model_perfs))
            bar_width = 0.3
            ax.bar(x_inds, model_perfs.values(), bar_width, label="test")
            ax.bar(x_inds + bar_width, val_score[score_name].values(), bar_width, label="validation")
            ax.set_xticks(x_inds + bar_width / 2, model_perfs.keys())
            ax.legend(fontsize=18)
        else:
            ax.bar(model_perfs.keys(), model_perfs.values())
        ax.set_ylabel(score_name.capitalize(), fontsize=15)
        ax.tick_params(axis="x", labelsize=12, labelrotation=45)
        ax.tick_params(axis="y", labelsize=12)
        ax.set_title(f"{score_name.capitalize()} for Models Trained With Best Parameters", fontsize=16 if len(model_perfs) < 8 else 18)
        fig.savefig(join(fig_save_path, f"best_models_{score_name}.png"))


def get_model_scores(classifier_params, cnn_params, cnn_metrics, test_out_path, results_path):
    classifier_names, cnn_names = [], []
    
    for classifier_type, classifier_params in classifier_params.items():
        for transform_group in classifier_params["transform_groups"]:
            classifier_names.append(f"{classifier_type}_{'_'.join(transform_group)}")
    for arch_entry in cnn_params["architecture"]:
        arch = list(arch_entry.keys())[0]
        variant = arch_entry[arch]
        if arch == "resnet" or arch == "vgg":
            cnn_names.append(f"{arch}{variant['num_layers']}")
    
    scores_test, scores_val = {}, {}
    for model_name in classifier_names + cnn_names:
        with open(join(test_out_path, model_name, "best_model_test_scores.json"), "r") as score_file:
            model_score_test = json.load(score_file)
        for score_type, value in model_score_test.items():
            if score_type in scores_test:
                scores_test[score_type][model_name] = value
            else:
                scores_test[score_type] = {model_name: value}
    
    for classifier_name in classifier_names:
        with open(join(results_path, classifier_name, "best_model_results.txt"), "r") as score_file:
            model_score_val = json.load(score_file)
        for score_type, value in model_score_val.items():
            if score_type in scores_val:
                scores_val[score_type][classifier_name] = value
            else:
                scores_val[score_type] = {classifier_name: value}
    
    for cnn_name in cnn_names:
        history = History.from_file(join(results_path, cnn_name, "history.json"))
        for score_type, value in history[-1].items():
            score_name = None
            if score_type == "valid_acc": score_name = "accuracy"
            elif score_type.removeprefix("valid_") in cnn_metrics: score_name = score_type.removeprefix("valid_")
            if score_name is not None:
                if score_name in scores_val:
                    scores_val[score_name][cnn_name] = value
                else:
                    scores_val[score_name][cnn_name] = {cnn_name: value}
    return scores_test, scores_val, classifier_names + cnn_names


if __name__ == "__main__":
    results_path, test_out_path, tables_path, figures_path = "results", "test_outputs", "tables", join("figures", "best_models_test")
    if not exists(figures_path): os.mkdir(figures_path)
    classifier_types = ["Logistic", "SVM_RBF"]
    test_data_path = join("test_data", "collected_images.npz")
    with open("experiment_params.yaml", "r") as exp_params:
        params = yaml.safe_load(exp_params)
    
    classifier_params = {cls_type: params[cls_type] for cls_type in classifier_types}
    classifier_metrics = {cls_type: get_model(cls_type, {})[2] for cls_type in classifier_types}
    # test_classifiers(classifier_params, results_path, test_out_path, test_data_path, classifier_metrics)
    # test_cnn(params["CNN"], results_path, test_out_path, test_data_path, VALIDATION_METRICS + ["accuracy"])
    scores_test, scores_val, model_names = get_model_scores(classifier_params, params["CNN"], VALIDATION_METRICS, test_out_path, results_path)
    save_table(model_names, scores_test, tables_path, "test")
    save_table(model_names, scores_val, tables_path, "val")
    plot_model_performance_comparisons(scores_test, scores_val, figures_path)
