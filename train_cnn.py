from functools import partial
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from skorch.callbacks import Checkpoint, EpochScoring, TrainEndCheckpoint, EarlyStopping
from skorch.classifier import NeuralNetClassifier
from timm.models import create_model
import torch
import yaml


CIFAR_NUM_CLS = 10
TORCH_OPTMZR_CLSMAP = {"SGD": torch.optim.SGD}
HPARAM_NAME_MAP = {"learning_rates": "lr",
                   "dropout": "module__drop_rate",
                   "momentum": "optimizer__momentum",
                   "l2_reg": "optimizer__weight_decay"}


# Currently unused
def get_resnet_creator(training, num_layers, pretrained, num_classes):
    def resnet_creator(training, **kwargs):
        resnet = create_model(f"resnet{num_layers}", pretrained=pretrained, num_classes=num_classes, **kwargs)
        resnet.fc = torch.nn.Sequential(resnet.fc, torch.nn.LogSoftmax(dim=1) if training else torch.nn.Softmax(dim=1))
        return resnet
    return partial(resnet_creator, training)


def run_grid_search_on_model(model_creator, hyperperameters, device, data_path):
    """
    Grid search with 3-Fold stratified cross validation and no refit with best hyperperameters. Except for optimizer, hyperperameters
    should be lists of values to try.
    """
    trainer = NeuralNetClassifier(model_creator, criterion=torch.nn.CrossEntropyLoss, optimizer=TORCH_OPTMZR_CLSMAP[hyperperameters["optimizer"]],
                                  max_epochs=50, train_split=None, callbacks=[EarlyStopping("train_loss")], verbose=0, device=device)

    gsearch_params = {}
    for parameter, values in hyperperameters.items():
        if parameter == "optimizer": continue
        if parameter not in HPARAM_NAME_MAP:
            raise KeyError(f"Invalid/Unknown hyperperameter name for training: {parameter}")
        gsearch_params[HPARAM_NAME_MAP[parameter]] = values

    dataset = np.load(data_path)
    X, y = dataset["images"], dataset["labels"].flatten()
    X = torch.tensor(X.reshape((X.shape[0], 3, 32, 32))).float()
    gs = GridSearchCV(trainer, gsearch_params, scoring="accuracy", n_jobs=3, refit=False, cv=3, verbose=4, return_train_score=True)
    gs.fit(X, y)
    return gs


def train_and_save_model(model_creator, hyperperameters, device, data_path, save_path):
    """
    Except for optimizer, hyperperameter names should conform to skorch conventions and match parameter names of their destination
    (eg. module, optimizer)
    """
    score_types = [f"{base_type}_{avg}" for base_type in ["f1", "jaccard", "precision", "recall"] for avg in ["micro", "macro"]]
    score_types.extend([f"roc_auc_{multicls}" for multicls in ["ovr", "ovo"]])
    performance_callbacks = [EpochScoring(scorer_name, lower_is_better=False, name=f"valid_{scorer_name}") for scorer_name in score_types]
    performance_callbacks.append(EpochScoring("accuracy", on_train=True, name="train_accuracy"))
    checkpoint_callbacks = [Checkpoint(dirname=save_path), TrainEndCheckpoint(dirname=save_path)]

    hyp_params_no_optim = hyperperameters.copy()
    optimizer = hyp_params_no_optim.pop("optimizer")
    trainer = NeuralNetClassifier(model_creator, criterion=torch.nn.CrossEntropyLoss, optimizer=TORCH_OPTMZR_CLSMAP[optimizer], max_epochs=150, batch_size=64,
                                  callbacks=performance_callbacks + checkpoint_callbacks + [EarlyStopping()], device=device, **hyp_params_no_optim)
    dataset = np.load(data_path)
    X, y = dataset["images"], dataset["labels"].flatten()
    X = torch.tensor(X.reshape((X.shape[0], 3, 32, 32))).float()
    trainer.fit(X, y)


if __name__ == "__main__":
    train_data_path = os.path.join("train_data", "collected_images.npz")
    results_path = "results"
    with open("experiment_params.yaml", "r") as exp_params:
        cnn_params = yaml.safe_load(exp_params)["CNN"]
    
    models_to_fit = []
    for arch, variant in cnn_params["architecture"].items():
        if arch == "resnet":
            resnet_name = f"resnet{variant['num_layers']}"
            resnet_creator = partial(create_model, resnet_name, variant["pretrained"], num_classes=CIFAR_NUM_CLS)
            models_to_fit.append((resnet_name, resnet_creator))
        else:
            raise NotImplementedError(f"Model architecture {arch} is not implemented yet.")

    for model_name, model_creator in models_to_fit:
        print(f"Begin grid search and training for {model_name}")
        gs_done = run_grid_search_on_model(model_creator, cnn_params["hyperparameters"], "cuda", train_data_path)
        model_save_path = os.path.join(results_path, model_name)
        if not os.path.exists(model_save_path): os.makedirs(model_save_path)
        gs_dframe, dframe_path = pd.DataFrame.from_dict(gs_done.cv_results_), os.path.join(model_save_path, "grid_search_results.csv")
        gs_dframe.to_csv(dframe_path)    # read with pd.read_csv("path to csv", index_col=0, comment="#")
        with open(dframe_path, "r+") as dframe_file:
            content = dframe_file.read()
            dframe_file.seek(0, 0)
            dframe_file.write(f"# Best parameters: {gs_done.best_params_}; index: {gs_done.best_index_}\n" + content)
        train_hyp_params = {"optimizer": cnn_params["hyperparameters"]["optimizer"]} | gs_done.best_params_
        train_and_save_model(model_creator, train_hyp_params, "cuda", train_data_path, model_save_path)
