import csv
from functools import partial
import joblib
import json
from matplotlib import pyplot as plt
import numpy as np
from os.path import exists, join
from sklearn.metrics import get_scorer
from skorch.callbacks import Checkpoint
from skorch.classifier import NeuralNetClassifier
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
            model_name = f"{classifier_type}_{'_'.join(transform_group)}"
            classifier_model = joblib.load(join(results_path, model_name, "best_model.pkl"))
            predict_and_store_if_needed(join(test_out_path, model_name), classifier_model, X_test)
            if classifier_type == "Logistic":
                predict_probabilities_and_store_if_needed(join(test_out_path, model_name), classifier_model, X_test)
            score_and_store_if_needed(join(test_out_path, model_name), classifier_model, X_test, y_test, classifier_metrics[classifier_type])


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
            predict_and_store_if_needed(join(test_out_path, cnn_name), cnn_model, X_test)
            predict_probabilities_and_store_if_needed(join(test_out_path, cnn_name), cnn_model, X_test)
            score_and_store_if_needed(join(test_out_path, cnn_name), cnn_model, X_test, y_test, cnn_metrics)
        else:
            raise NotImplementedError(f"Model architecture {arch} is not implemented yet.")


def plot_model_performance_comparisons(classifier_params, cnn_params, test_out_path, fig_save_path):
    model_names = []
    
    for classifier_type, classifier_params in classifier_params.items():
        for transform_group in classifier_params["transform_groups"]:
            model_names.append(f"{classifier_type}_{'_'.join(transform_group)}")
    for arch_entry in cnn_params["architecture"]:
        arch = list(arch_entry.keys())[0]
        variant = arch_entry[arch]
        if arch == "resnet" or arch == "vgg":
            model_names.append(f"{arch}{variant['num_layers']}")

    scores = {}
    for model_name in model_names:
        with open(join(test_out_path, model_name, "best_model_test_scores.json"), "r") as score_file:
            model_score = json.load(score_file)
        for score_type, value in model_score.items():
            if score_type in scores:
                scores[score_type][model_name] = value
            else:
                scores[score_type] = {model_name: value}
    
    with open(join(test_out_path, "best_models_test_scores.csv"), "w") as table_file:
        writer = csv.writer(table_file, lineterminator="\n")
        writer.writerow([""] + model_names)
        for score_type, model_perfs in scores.items():
            score_for_models = ["" for _ in range(len(model_names))]
            for model_name, perf_score in model_perfs.items():
                score_for_models[model_names.index(model_name)] = perf_score
            writer.writerow([score_type] + score_for_models)
    
    for score_name, model_perfs in scores.items():
        fig, ax = plt.subplots(figsize=(len(model_perfs), 6), constrained_layout=True)
        ax.bar(model_perfs.keys(), model_perfs.values())
        ax.set_ylabel(score_name.capitalize())
        ax.tick_params(axis="x", labelrotation=45)
        ax.set_title(f"{score_name.capitalize()} for Models Trained With Best Parameters")
        fig.savefig(join(fig_save_path, f"best_models_{score_name}.png"))


if __name__ == "__main__":
    results_path, test_out_path, figures_path = "results", "test_outputs", "figures"
    classifier_types = ["Logistic", "SVM_RBF"]
    test_data_path = join("test_data", "collected_images.npz")
    with open("experiment_params.yaml", "r") as exp_params:
        params = yaml.safe_load(exp_params)
    
    classifier_params = {cls_type: params[cls_type] for cls_type in classifier_types}
    classifier_metrics = {cls_type: get_model(cls_type, {})[2] for cls_type in classifier_types}
    # test_classifiers(classifier_params, results_path, test_out_path, test_data_path, classifier_metrics)
    # test_cnn(params["CNN"], results_path, test_out_path, test_data_path, VALIDATION_METRICS + ["accuracy"])
    plot_model_performance_comparisons(classifier_params, params["CNN"], test_out_path, figures_path)
