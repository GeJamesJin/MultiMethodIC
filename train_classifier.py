import joblib
import json
import numpy as np
import os
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, get_scorer
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.svm import SVC
import yaml


METRICS_MICRO_MACRO = [f"{base_type}_{avg}" for base_type in ["f1", "jaccard", "precision", "recall"] for avg in ["micro", "macro"]]
METRICS_ROC_AUC = [f"roc_auc_{multicls}" for multicls in ["ovr", "ovo"]]
CIFAR_NUM_CLS = 10


def prepare_data_with_transformations(data_path, transformations, degree=2, n_components=100, gamma=0.1):
    """
    Loads and prepares the dataset with PCA and Polynomial Features.
    """
    dataset = np.load(data_path)
    X, y = dataset["images"], dataset["labels"].flatten()
    X = X.reshape((X.shape[0], -1))  # Flatten the images
    X_transformed = X

    for transform in transformations:
        if transform == "PCA":
            # Apply PCA to reduce dimensionality
            print(f"Applying PCA to reduce dimensions to {n_components} components...")
            pca = PCA(n_components=n_components)
            X_transformed = pca.fit_transform(X_transformed)
        elif transform == "Polynomial":
            # Apply PolynomialFeatures
            print(f"Applying Polynomial Features with degree {degree}...")
            poly = PolynomialFeatures(degree=degree, include_bias=False)
            X_transformed = poly.fit_transform(X_transformed)
        elif transform == "RBF":
            # Apply RBF Kernel Transformation
            print(f"Applying RBF Kernel Transformation with gamma={gamma}...")
            rbf_sampler = RBFSampler(gamma=gamma, random_state=42)
            X_transformed = rbf_sampler.fit_transform(X_transformed)
        else:
            raise ValueError(f"Data preparation failed, unknown feature transformation {transform}")

    print(f"Original feature shape: {X.shape} -- {transformations} --> {X_transformed.shape}")
    return X_transformed, y


def run_grid_search(model, X_train, y_train, param_grids, metrics, results_path):
    """
    Performs grid search for Logistic Regression and saves detailed metrics.
    """
    _, inds = next(StratifiedKFold(2).split(np.zeros(X_train.shape[0]), y_train))
    X_train, y_train = X_train[inds], y_train[inds]
    print(f"Performing grid search on half of train data ({X_train.shape[0]} samples)")
    cv = StratifiedKFold(shuffle=True, random_state=42)
    # Find the best parameters and refit model on entire dataset using best parameters
    grid_search = GridSearchCV(model, param_grids, scoring=metrics, refit=False, cv=cv, n_jobs=-1, verbose=4, return_train_score=True)
    grid_search.fit(X_train, y_train)
    grid_search.best_index_ = grid_search.cv_results_["rank_test_accuracy"].argmin()    # Hardcoded best params retrieval to avoid refitting under multimetric
    grid_search.best_params_ = grid_search.cv_results_["params"][grid_search.best_index_]

    # Save grid search results
    results_df = pd.DataFrame(grid_search.cv_results_)
    results_file = os.path.join(results_path, "grid_search_results.csv")
    results_df.to_csv(results_file, index=False)
    print(f"Grid search results saved to: {results_file}")

    # Extract additional metrics
    mean_train_accuracy = grid_search.cv_results_['mean_train_accuracy'][grid_search.best_index_]
    mean_test_accuracy = grid_search.cv_results_['mean_test_accuracy'][grid_search.best_index_]
    mean_fit_time = grid_search.cv_results_['mean_fit_time'][grid_search.best_index_]

    # Save best model metrics
    metrics_file = os.path.join(results_path, "grid_search_metrics.txt")
    with open(metrics_file, "w") as f:
        f.write(f"Best Parameters: {grid_search.best_params_}\n")
        f.write(f"Best Train accuracy: {mean_train_accuracy:.4f}\n")
        f.write(f"Best Test accuracy: {mean_test_accuracy:.4f}\n")
        f.write(f"Mean Fit Time: {mean_fit_time:.4f} seconds\n")
    print(f"Best model metrics saved to: {metrics_file}")

    return grid_search


def train_and_save_logistic(best_model, X_train, y_train, X_test, y_test, metrics, save_path):
    """
    Evaluate the best logistic regression model and saves it.
    """
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    scores = {score_name: get_scorer(score_name)(best_model, X_test, y_test) for score_name in metrics}
    report = classification_report(y_test, y_pred, target_names=[f"Class {i}" for i in range(CIFAR_NUM_CLS)])

    # Save the model
    model_file = os.path.join(save_path, "best_model.pkl")
    joblib.dump(best_model, model_file)
    print(f"Model saved to: {model_file}")

    # Save evaluation results
    report_file = os.path.join(save_path, "classification_report.txt")
    with open(report_file, "w") as f:
        f.write(f"Accuracy: {scores['accuracy']:.4f}\n")
        f.write(report)
    eval_file = os.path.join(save_path, "best_model_results.txt")
    with open(eval_file, "w") as f:
        json.dump(scores, f)
    print(f"Evaluation results saved to {save_path}")


def get_model(model_type, model_parameters):
    model_kwargs = {model_param: value for model_param, value in model_parameters.items()
                        if model_param not in ["transform_groups", "hyperparameters"]}
    if model_type == "Logistic":
        ModelCls = LogisticRegression
        scoring_metrics = METRICS_MICRO_MACRO + METRICS_ROC_AUC+ ["accuracy"]
    elif model_type == "SVM_RBF":
        ModelCls = SVC
        scoring_metrics = METRICS_MICRO_MACRO + ["accuracy"]
    else:
        raise ValueError(f"{model_type} is not supported, please use 'Logistic' or 'SVM_RBF'")
    return ModelCls, model_kwargs, scoring_metrics


if __name__ == "__main__":
    # Path to the dataset
    train_data_path = os.path.join("train_data", "collected_images.npz")
    model_type = "SVM_RBF"    # One of "Logistic" or "SVM_RBF"
    results_path = "results"
    with open("experiment_params.yaml", "r") as params_file:
        model_parameters = yaml.safe_load(params_file)[model_type]
    print(f"Experimenting with model: {model_type}")

    ModelCls, model_kwargs, scoring_metrics = get_model(model_type, model_parameters)
    for transform_group in model_parameters["transform_groups"]:
        save_path = os.path.join(results_path, f"{model_type}_{'_'.join(transform_group)}")
        os.makedirs(save_path, exist_ok=True)

        # Prepare data and applying any transformations
        print(f"Preparing data with transformations: {', '.join(transform_group)}...")
        X, y = prepare_data_with_transformations(train_data_path, transform_group)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Standardize the data
        print("Standardizing the data...")
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Run Grid Search for selected model
        # print(f"Running Grid Search for {model_type}...")
        # grid_search = run_grid_search(ModelCls(**model_kwargs), X_train, y_train, model_parameters["hyperparameters"], scoring_metrics, save_path)

        # Train and Save the Best Model
        print("Training and saving the best model...")
        train_and_save_logistic(ModelCls(**model_kwargs, **{'C': 1, 'gamma': 'auto'}), X_train, y_train, X_test, y_test, scoring_metrics, save_path)
