from functools import partial
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import GridSearchCV
from skorch.callbacks import Checkpoint, EpochScoring, TrainEndCheckpoint, EarlyStopping
from skorch.classifier import NeuralNetClassifier
from skorch.dataset import Dataset
from timm.models import create_model
import torch
import torchvision
import yaml


CIFAR_NUM_CLS = 10
TORCH_OPTMZR_CLSMAP = {"SGD": torch.optim.SGD}
HPARAM_NAME_MAP = {"learning_rates": "lr",
                   "dropout": "module__drop_rate",
                   "momentum": "optimizer__momentum",
                   "l2_reg": "optimizer__weight_decay"}

VALIDATION_METRICS = [f"{base_type}_{avg}" for base_type in ["f1", "jaccard", "precision", "recall"] for avg in ["micro", "macro"]] \
    + [f"roc_auc_{multicls}" for multicls in ["ovr", "ovo"]]


def load_dataset(data_path):
    dataset = np.load(data_path)
    X, y = dataset["images"].astype(float), dataset["labels"].flatten()
    X /= 255
    return X, y


class ImageDataset(Dataset):

    imagenet_transform = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def transform(self, X, y):
        X, y = super().transform(X, y)
        return self.imagenet_transform(X), y


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
                                  max_epochs=50, dataset=ImageDataset, train_split=None, callbacks=[EarlyStopping("train_loss")], verbose=0, device=device)

    gsearch_params = {}
    for parameter, values in hyperperameters.items():
        if parameter == "optimizer": continue
        if parameter not in HPARAM_NAME_MAP:
            raise KeyError(f"Invalid/Unknown hyperperameter name for training: {parameter}")
        gsearch_params[HPARAM_NAME_MAP[parameter]] = values

    X, y = load_dataset(data_path)
    X = torch.tensor(X.reshape((X.shape[0], 3, 32, 32))).float()
    gs = GridSearchCV(trainer, gsearch_params, scoring=VALIDATION_METRICS + ["accuracy"], n_jobs=3, refit=False, cv=4, verbose=4, return_train_score=True)
    gs.fit(X, y)
    gs.best_index_ = gs.cv_results_["rank_test_accuracy"].argmin()    # Hardcoded best params retrieval to avoid refitting under multimetric
    gs.best_params_ = gs.cv_results_["params"][gs.best_index_]
    return gs


def train_and_save_model(model_creator, hyperperameters, device, data_path, save_path):
    """
    Except for optimizer, hyperperameter names should conform to skorch conventions and match parameter names of their destination
    (eg. module, optimizer)
    """
    performance_callbacks = [EpochScoring(scorer_name, lower_is_better=False, name=f"valid_{scorer_name}") for scorer_name in VALIDATION_METRICS]
    performance_callbacks.append(EpochScoring("accuracy", on_train=True, name="train_accuracy"))
    checkpoint_callbacks = [Checkpoint(dirname=save_path), TrainEndCheckpoint(dirname=save_path)]

    hyp_params_no_optim = hyperperameters.copy()
    optimizer = hyp_params_no_optim.pop("optimizer")
    trainer = NeuralNetClassifier(model_creator, criterion=torch.nn.CrossEntropyLoss, optimizer=TORCH_OPTMZR_CLSMAP[optimizer], max_epochs=150,
                                  batch_size=64, dataset=ImageDataset, callbacks=performance_callbacks + checkpoint_callbacks + [EarlyStopping()],
                                  device=device, **hyp_params_no_optim)
    X, y = load_dataset(data_path)
    X = torch.tensor(X.reshape((X.shape[0], 3, 32, 32))).float()
    trainer.fit(X, y)


if __name__ == "__main__":
    train_data_path = os.path.join("train_data", "collected_images.npz")
    results_path = "results"
    with open("experiment_params.yaml", "r") as exp_params:
        cnn_params = yaml.safe_load(exp_params)["CNN"]
    
    models_to_fit = []
    for arch_entry in cnn_params["architecture"]:
        arch = list(arch_entry.keys())[0]
        variant = arch_entry[arch]
        if arch == "resnet" or arch == "vgg":
            cnn_name = f"{arch}{variant['num_layers']}"
            cnn_creator = partial(create_model, cnn_name, variant["pretrained"], num_classes=CIFAR_NUM_CLS)
            models_to_fit.append((cnn_name, cnn_creator))
        else:
            raise NotImplementedError(f"Model architecture {arch} is not implemented yet.")

    for model_name, model_creator in models_to_fit:
        print(f"Begin grid search and training for {model_name}")
        gs_done = run_grid_search_on_model(model_creator, cnn_params["hyperparameters"], "cuda", train_data_path)
        model_save_path = os.path.join(results_path, model_name)
        if not os.path.exists(model_save_path): os.makedirs(model_save_path)
        gs_dframe, dframe_path = pd.DataFrame.from_dict(gs_done.cv_results_), os.path.join(model_save_path, "grid_search_results.csv")
        gs_dframe.to_csv(dframe_path, index=False)
        with open(dframe_path, "r+") as dframe_file:
            content = dframe_file.read()
            dframe_file.seek(0, 0)
            dframe_file.write(f"# Best parameters: {gs_done.best_params_}; index: {gs_done.best_index_}\n" + content)
        train_hyp_params = {"optimizer": cnn_params["hyperparameters"]["optimizer"]} | gs_done.best_params_
        train_and_save_model(model_creator, train_hyp_params, "cuda", train_data_path, model_save_path)
