import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
)
from yellowbrick.model_selection import LearningCurve, ValidationCurve
from yellowbrick.classifier import DiscriminationThreshold
from cycler import cycler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import BaseEstimator
from sklearn.model_selection import StratifiedKFold
from helper_functions.ml_functions import get_calibration_curve_values
from typing import Tuple, List, Callable


def plot_coefficients(
    model_coef_list: List[Pipeline],
    leg_labels_list: List[str],
    x_labels: List[str] = None,
    title: str = None,
    y_scale: str = None,
) -> plt.figure:
    """Plots up to 7 different models coeficients on the same plot"""

    markers = ["o", "v", "s", "p", "^", "<", ">"]

    if len(model_coef_list) > 7:
        return f"Too many models to plot"

    fig, ax = plt.subplots(figsize=(10, 8))

    for i in range(len(model_coef_list)):

        ax.plot(
            model_coef_list[i],
            marker=markers[i],
            c=np.random.rand(
                3,
            ),
            linestyle="None",
        )

    ax.legend(
        leg_labels_list, bbox_to_anchor=(1.35, 0.5), edgecolor="white", fontsize=12
    )

    ax.set_xticks(range(len(model_coef_list[i])))
    if x_labels != None:
        ax.set_xticklabels(x_labels, rotation=45, ha="right")
    else:
        ax.set_xticklabels(range(len(model_coef_list[0])))
    ax.ticklabel_format(axis="y", useOffset=False, style="plain")
    ax.set_ylabel("Coefficient magnitude")
    ax.set_title(title, fontsize=13, y=1.03)
    if y_scale != None:
        ax.set_yscale(y_scale)
        mn = min(list(map(min, model_coef_list)))
        mx = max(list(map(max, model_coef_list)))
        ax.set_ylim(mn + mn * 0.3, mx + mx * 0.3)

    plt.show()


def plot_cm(
    Y: np.ndarray,
    Y_predicted: np.ndarray,
    display_labels: List[str] = None,
    cmap: str = None,
    colorbar: bool = None,
    title: str = None,
    normalize: str = "true",
) -> plt.figure:
    """Plots a confusion matrix of a classification model"""

    fig, ax = plt.subplots(figsize=(10, 6))
    plt.rc("font", size=12)

    if colorbar == None:
        colorbar = False

    cm = confusion_matrix(Y, Y_predicted, normalize=normalize)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    disp.plot(
        colorbar=colorbar, cmap=cmap, ax=ax,
    )
    plt.title(
        title, fontsize=13, y=1.03,
    )
    plt.grid(False)

    plt.show()
    

def plot_multi_cm(
    model_lst: List[Pipeline],
    X_test: np.ndarray,
    y_test: np.ndarray,
    display_labels: List[str] = None,
    cmap: str = None,
    normalize: str = "true",
) -> plt.figure:
    "Plots multiple models classification results"

    fig, axes = plt.subplots(
        1, len(model_lst), figsize=(6 * len(model_lst), 6), sharey=True
    )

    plt.rc("font", size=12)

    for i, ax in enumerate(axes.flatten()):

        try:
            y_predicted = model_lst[i][0].predict(X_test)
        except:
            y_predicted = model_lst[i].predict(X_test)
        
        cm = confusion_matrix(y_test, y_predicted, normalize=normalize)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=display_labels
        )
        disp.plot(colorbar=False, cmap=cmap, ax=ax)

        try:
            ax.set_title(
                model_lst[i][1], fontsize=13, y=1.03,
            )
        except:
            ax.set_title(
                model_lst[i].__class__.__name__, fontsize=13, y=1.03,
            )

        ax.grid(False)

    cax, kw = mpl.colorbar.make_axes(
        [ax for ax in axes.flatten()], shrink=0.75, pad=0.03
    )
    fig.colorbar(disp.im_, cax=cax, **kw)

    plt.show()


def plot_cm_without_model(
    Y: np.ndarray,
    Y_predicted: np.ndarray,
    display_labels: List[str] = None,
    cmap: str = None,
    colorbar: bool = None,
    title: str = None,
    normalize: str = "true",
) -> plt.figure:
    """Plots a confusion matrix without a model"""

    fig, ax = plt.subplots(figsize=(10, 6))
    plt.rc("font", size=12)

    if colorbar == None:
        colorbar = False

    cm = confusion_matrix(Y, Y_predicted, normalize=normalize)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    disp.plot(
        colorbar=colorbar, cmap=cmap, ax=ax,
    )
    plt.title(
        title, fontsize=13, y=1.03,
    )
    plt.grid(False)

    plt.show()


def plot_roc_auc(
    clf_list: List[Callable],
    X_test: np.ndarray,
    y_test: np.ndarray,
    title_lst: List[str],
    thresh: bool = None,
) -> plt.figure:
    "Plots models roc auc curve and thresholds"

    fig, ax = plt.subplots(figsize=(8, 6))

    for model, title in zip(clf_list, title_lst):

        y_predicted = model.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_predicted)
        roc_score = roc_auc_score(y_test, y_predicted)
        ax.plot(
            fpr, tpr, lw=2, label=f"{title}, Roc Auc = {roc_score:.3f}",
        )
        if thresh:
            J = tpr - fpr
            best_idx = np.argmax(J)
            best_thresh = thresholds[best_idx]
            ax.plot(
                fpr[best_idx],
                tpr[best_idx],
                marker="o",
                color=ax.get_lines()[-1].get_c(),
                markersize=12,
                fillstyle="none",
                mew=3,
                label=f"{title}, Best threshold = {best_thresh:.3f}",
            )
    ax.plot(
        [0, 0, 1, 1],
        [0, 1, 1, 1],
        color="r",
        lw=2,
        linestyle="-",
        label="Perfect model",
    )
    ax.plot([0, 1], [0, 1], color="k", lw=1, linestyle="--", label="Random model")
    ax.set_xlabel("False Positive Rate", fontsize=13, labelpad=10)
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=13, labelpad=10)

    ax.legend(
        fontsize=12, loc="lower right", ncol=1, facecolor="white", edgecolor="white",
    )
    ax.set_xlim([-0.01, 1.00])
    ax.set_ylim([-0.01, 1.01])

    ax.set_title(
        "Different classifiers Area under the curve", fontsize=13, y=1.02,
    )

    plt.show()


def plot_pr_auc(
    clf_list: List[Callable],
    X_test: np.ndarray,
    y_test: np.ndarray,
    title_lst: List[str],
    thresh: bool = None,
) -> plt.figure:
    "Plots models PR auc curve"

    fig, ax = plt.subplots(figsize=(8, 6))

    for model, title in zip(clf_list, title_lst):

        y_predicted = model.predict_proba(X_test)[:, 1]
        precision, recall, thresholds = precision_recall_curve(y_test, y_predicted)
        average_precision = average_precision_score(y_test, y_predicted)
        ax.plot(
            recall, precision, lw=2, label=f"{title}, PR AUC = {average_precision:.3f}",
        )
        if thresh:
            fscore = (2 * precision * recall) / (precision + recall)
            best_idx = np.argmax(fscore)
            best_thresh = thresholds[best_idx]
            ax.plot(
                recall[best_idx],
                precision[best_idx],
                marker="o",
                color=ax.get_lines()[-1].get_c(),
                markersize=12,
                fillstyle="none",
                mew=3,
                label=f"{title}, Best threshold = {best_thresh:.3f}",
            )

    ax.plot(
        [0, 1, 1, 1],
        [1, 1, 1, 0],
        color="r",
        lw=2,
        linestyle="-",
        label="Perfect model",
    )
    random_model = len(y_test[y_test == 1]) / len(y_test)
    ax.plot(
        [0, 1],
        [random_model, random_model],
        color="k",
        lw=1,
        linestyle="--",
        label="Random model",
    )
    ax.set_xlabel("Recall", fontsize=13, labelpad=10)
    ax.set_ylabel("Precision", fontsize=13, labelpad=10)

    ax.legend(
        fontsize=12, loc="lower left", ncol=1, facecolor="white", edgecolor="white",
    )
    ax.set_xlim([0.00, 1.01])
    ax.set_ylim([-0.01, 1.01])

    ax.set_title(
        "Different classifiers Precision-Recall Area under the curve",
        fontsize=13,
        y=1.02,
    )

    plt.show()
    

def plot_learning_curve(
    model: BaseEstimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    kf: StratifiedKFold,
    metric: str,
) -> plt.figure:
    "Plots model's learning curve"
    
    fig, ax = plt.subplots(figsize=(8, 6))

    plt.rcParams["axes.prop_cycle"] = cycler("color", ["purple", "darkblue"])

    sizes = np.linspace(0.3, 1.0, 10)
    visualizer = LearningCurve(
        model, cv=kf, scoring=metric, train_sizes=sizes, n_jobs=-1,
    )

    visualizer.fit(X_train, y_train)
    visualizer.show()

    plt.show()


def plot_validation_curve(
    model: BaseEstimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    kf: StratifiedKFold,
    param_to_check: str,
    param_range: List[int],
    metric: str,
) -> plt.figure:
    "Plots model's learning curve"
    
    fig, ax = plt.subplots(figsize=(8, 6))

    plt.rcParams["axes.prop_cycle"] = cycler("color", ["purple", "darkblue"])

    visualizer = ValidationCurve(
        model,
        param_name=param_to_check,
        param_range=param_range,
        cv=kf,
        scoring=metric,
        n_jobs=-1,
    )

    visualizer.fit(X_train, y_train)
    visualizer.show()

    plt.show()

    
def plot_discrimination_threshold(
    model: BaseEstimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    kf: StratifiedKFold,
    n_trials: int,
    argmax: str,
) -> plt.figure:
    """Plots discrimination treshhold according to argmax metric. Available metrics are:
    "precision", "recall", "queue_rate" and "fscore"""

    sns.set_theme(style="darkgrid")

    fig, ax = plt.subplots(figsize=(8, 6))

    visualizer = DiscriminationThreshold(
        model, cv=kf, argmax=argmax, n_trials=n_trials, n_jobs=-1
    )

    visualizer.fit(
        X_train, y_train,
    )
    visualizer.show()

    return fig


def plot_calibration_curve(
    model: BaseEstimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    kf: StratifiedKFold,
) -> plt.figure:
    "Plots model's probability calibration curves"

    fig, ax = plt.subplots(figsize=(10, 8))

    for method in ["uncalibrated", "sigmoid", "isotonic"]:
        if method == "uncalibrated":
            fraction_p, mean_val, score_dict = get_calibration_curve_values(
                model, X_train, y_train, X_test, y_test
            )
            ax.plot(
                mean_val,
                fraction_p,
                lw=3,
                label=f"Method = {method},\nBrier Loss = {score_dict['Brier loss']},\nLog Loss = {score_dict['Log loss']}",
            )
        else:
            calib_model = CalibratedClassifierCV(model, cv=kf, method=method, n_jobs=-1)
            fraction_p, mean_val, score_dict = get_calibration_curve_values(
                calib_model, X_train, y_train, X_test, y_test
            )
            ax.plot(
                mean_val,
                fraction_p,
                lw=3,
                label=f"Method = {method},\nBrier Loss = {score_dict['Brier loss']},\nLog Loss = {score_dict['Log loss']}",
            )

    ax.plot(
        [0, 1], [0, 1], color="k", lw=1, linestyle="--", label="Perfectly calibrated",
    )
    ax.set_xlabel("Mean predicted probability", fontsize=13, labelpad=10)
    ax.set_ylabel("Fraction of positives", fontsize=13, labelpad=10)

    ax.legend(
        fontsize=12, loc="upper left", ncol=1, facecolor="white", edgecolor="white",
    )
    ax.set_xlim([-0.01, 1.00])
    ax.set_ylim([-0.01, 1.01])

    ax.set_title(
        f"{model.__class__.__name__} Calibration plot ", fontsize=13, y=1.02,
    )

    plt.show()
    


def plot_residuals(
    values_list: List[np.ndarray],
    errors_lst: List[np.ndarray],
    x_labels_list: List[str],
    title_list: List[str],
) -> None:
    """From input lists, plots residuals plots against train and test data"""

    fig, axes = plt.subplots(1, len(values_list), figsize=(16, 7), sharey=True)

    for i, ax in enumerate(axes.flatten()):

        sns.scatterplot(x=values_list[i], y=errors_lst[i], alpha=0.5, ax=ax)
        lin = ax.axhline(0, color="r")
        ax.set_xlabel(x_labels_list[i], fontsize=12, labelpad=10)
        ax.set_ylabel("Residuals", fontsize=12, labelpad=10)
        ax.set_title(title_list[i], fontsize=13, y=1.03)

    ax.legend(
        [lin],
        ["Perfect guess by a model"],
        bbox_to_anchor=(0.13, -0.13),
        edgecolor="white",
        fontsize=12,
    )


def plot_reg(
    prediction: np.ndarray,
    y_test_data: np.ndarray,
    title: str = None,
    ylabel: str = None,
) -> plt.figure:
    """
    Plots a lineplot for true y_test values and a scatter plot for predictions
    """

    fig, ax = plt.subplots(figsize=(20, 8))

    x_points = range(len(y_test_data))
    sns.lineplot(
        x=x_points, y=y_test_data, color="b", lw=2, label="Test data values", ax=ax
    )
    sns.scatterplot(
        x=x_points, y=prediction, alpha=0.7, color="red", label="Predicted data", ax=ax
    )
    ax.set_xlabel("Data points", fontsize=13, labelpad=10)
    ax.set_xticklabels([])
    ax.set_xticks([])
    ax.set_ylabel(ylabel, fontsize=13, labelpad=10)
    ax.set_title(title, fontsize=15, y=1.03)
    ax.legend(
        loc="center",
        ncol=2,
        bbox_to_anchor=(0.5, -0.13),
        edgecolor="white",
        fontsize=13,
    )

    plt.show()
    
    
def line_plot(
    df: pd.DataFrame,
    x: str,
    y: str,
    x_ticks: List[int],
    x_ticklabels: List[str],
    x_label: str,
    y_label: str,
    title: str,
) -> plt.figure:
    "Plots a single line plot"

    fig, ax = plt.subplots(figsize=(10, 6))

    ax = sns.lineplot(x=x, y=y, data=df)

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticklabels, fontsize=12)
    ax.set_xlabel(x_label, fontsize=12, labelpad=10)
    ax.set_ylabel(y_label, fontsize=12, labelpad=10)

    ax.set_title(
        title, fontsize=13, y=1.03,
    )

    plt.show()
    

def two_line_plot(
    df: pd.DataFrame,
    x: List[str],
    y: List[str],
    y_value_line: List[float],
    text_coord: Tuple[float],
    x_ticks: List[range],
    x_ticklabels: List[str],
    x_label: List[str],
    y_label: List[str],
    title: List[str],
) -> plt.figure:
    "Plots two lines plot next to each other"

    plt.figure(figsize=(18, 6))

    ax1 = plt.subplot(121)
    ax1 = sns.lineplot(x=x[0], y=y[0], data=df)

    ax1.axhline(y=y_value_line[0], color="r", linestyle="dashed")
    ax1.text(text_coord[0][0], text_coord[0][1], "Year's average", size=12)

    ax1.set_xticks(x_ticks[0])
    ax1.set_xticklabels(x_ticklabels[0], fontsize=12)
    ax1.set_xlabel(x_label[0], fontsize=12, labelpad=10)
    ax1.set_ylabel(y_label[0], fontsize=12, labelpad=10)

    ax1.set_title(
        title[0], fontsize=13, y=1.03,
    )
    ax1.margins(0.0, 0.05)

    ax2 = plt.subplot(122)
    ax2 = sns.lineplot(x=x[1], y=y[1], data=df)

    ax2.axhline(y=y_value_line[1], color="r", linestyle="dashed")
    ax2.text(text_coord[1][0], text_coord[1][1], "Year's average", size=12)

    ax2.set_xticks(x_ticks[1])
    ax2.set_xticklabels(x_ticklabels[1], fontsize=12)
    ax2.set_xlabel(x_label[1], fontsize=12, labelpad=10)
    ax2.set_ylabel(y_label[1], fontsize=12, labelpad=10)

    ax2.set_title(
        title[1], fontsize=13, y=1.03,
    )
    ax2.margins(0.0, 0.05)

    plt.show()
    
    
def two_line_plot_2(
    df: List[pd.DataFrame],
    x: List[str],
    y: List[str],
    y_value_line: List[float],
    text_coord: Tuple[float],
    x_ticks: List[range],
    x_ticklabels: List[str],
    x_label: List[str],
    y_label: List[str],
    title: List[str],
) -> plt.figure:
    "Plots two lines plot next to each other"

    plt.figure(figsize=(18, 6))

    ax1 = plt.subplot(121)
    ax1 = sns.lineplot(x=x[0], y=y[0], data=df[0])

    ax1.axhline(y=y_value_line[0], color="r", linestyle="dashed")
    ax1.text(text_coord[0][0], text_coord[0][1], "Year's average", size=12)

    ax1.set_xticks(x_ticks[0])
    ax1.set_xticklabels(x_ticklabels[0], fontsize=12)
    ax1.set_xlabel(x_label[0], fontsize=12, labelpad=10)
    ax1.set_ylabel(y_label[0], fontsize=12, labelpad=10)

    ax1.set_title(
        title[0], fontsize=13, y=1.03,
    )
    ax1.margins(0.0, 0.05)

    ax2 = plt.subplot(122)
    ax2 = sns.lineplot(x=x[1], y=y[1], data=df[1])

    ax2.axhline(y=y_value_line[1], color="r", linestyle="dashed")
    ax2.text(text_coord[1][0], text_coord[1][1], "Year's average", size=12)

    ax2.set_xticks(x_ticks[1])
    ax2.set_xticklabels(x_ticklabels[1], fontsize=12)
    ax2.set_xlabel(x_label[1], fontsize=12, labelpad=10)
    ax2.set_ylabel(y_label[1], fontsize=12, labelpad=10)

    ax2.set_title(
        title[1], fontsize=13, y=1.03,
    )
    ax2.margins(0.0, 0.05)

    plt.show()


def plot_single_violin(
    df: pd.DataFrame,
    x: str,
    y: str,
    x_label: str,
    y_label: str,
    title: str,
    leg_title="",
    split=False,
    hue=None,
) -> plt.figure:
    "Plots a single violin plot"

    g = sns.catplot(
        x=x,
        y=y,
        data=df,
        cut=1.2,
        kind="violin",
        hue=hue,
        split=split,
        height=8,
        aspect=2,
    )

    g._legend.set_title(leg_title)
    plt.gca().tick_params(axis="both", which="major", labelsize=12)
    plt.gca().set_xlabel(x_label, fontsize=13, labelpad=10)
    plt.gca().set_ylabel(y_label, fontsize=13, labelpad=10)
    plt.title(title, fontsize=14, y=1.03)
    plt.ticklabel_format(style="plain", axis="y")

    plt.show()

def plot_violin(
    df: pd.DataFrame,
    x: str,
    y: str,
    col: str,
    x_labels: List[str],
    y_label: str,
    subplot_titles: List[str],
    title: str,
    leg_title=None,
    split=False,
    hue=None,
) -> plt.figure:
    "Plots violin plots in two subplots"

    g = sns.catplot(
        x=x,
        y=y,
        col=col,
        data=df,
        col_wrap=2,
        cut=1.2,
        kind="violin",
        hue=hue,
        split=split,
        height=6,
        aspect=1,
    )

    for i, ax in enumerate(g.axes.flat):
        ax.set_title(subplot_titles[i], fontsize=11)
        ax.tick_params(axis="both", which="major", labelsize=11)
        ax.set_xticklabels(x_labels, rotation=30, ha="right")
        ax.set_xlabel("")
        ax.set_ylabel(y_label, fontsize=11, labelpad=10)

    if hue:
        g._legend.set_title(leg_title)
    g.fig.suptitle(
        title, fontsize=13, y=1.03,
    )
    plt.ticklabel_format(style="plain", axis="y")

    plt.show()
    

def cat_plot(
    df: pd.DataFrame,
    x: str,
    y: str,
    y_label: str,
    leg_title: str,
    title: str,
    hue: str = None,
) -> plt.figure:
    "Plots categorical plot"

    g = sns.catplot(
        x=x,
        y=y,
        data=df,
        hue=hue,
        kind="bar",
        ci=None,
        height=6,
        aspect=2,
    )

    g.axes.flat[0].set_xlabel("")
    g.axes.flat[0].set_ylabel(y_label, fontsize=12, labelpad=10)
    g.axes.flat[0].set_xticklabels(g.axes.flat[0].get_xticklabels(), fontsize=12)
    g.axes.flat[0].set_yticklabels(g.axes.flat[0].get_yticklabels(), fontsize=12)
    g._legend.set_title(leg_title)

    plt.title(
        title, fontsize=13, y=1.02,
    )

    plt.show()


def small_catplot(
    df: pd.DataFrame, x: str, y: str, hue: str, leg_title: str, title: str
) -> plt.figure:
    "Plots small catplot with percentage"

    g = sns.catplot(
        x=x, y=y, hue=hue, data=df, kind="bar", ci=None, height=6, aspect=1.5,
    )

    g._legend.set_title(leg_title)

    ax = g.axes.flat[0]
    ax.set_xlabel("")
    for p in ax.patches:
        ax.annotate(
            f"{p.get_height():.2f}%",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="center",
            xytext=(0, 5),
            size=11,
            textcoords="offset points",
        )

    ax.get_yaxis().set_visible(False)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=12)
    g.fig.suptitle(title, fontsize=13, y=1.03)
    plt.setp(g._legend.get_title(), fontsize=12)

    plt.show() 


def catplot_with_pct(
    df: pd.DataFrame, x: str, y: str, hue: str, leg_title: str, title: str
) -> plt.figure:
    "Plots categorical data with percentage"

    g = sns.catplot(
        x=x,
        y=y,
        hue=hue,
        data=df,
        kind="bar",
        ci=None,
        height=7,
        aspect=2,
    )

    g._legend.set_title(leg_title)

    ax = g.axes.flat[0]
    ax.set_xlabel("")
    for p in ax.patches:
        ax.annotate(
            f"{p.get_height():.2f}%",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="center",
            xytext=(0, 5),
            size=10,
            textcoords="offset points",
        )

    ax.get_yaxis().set_visible(False)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=12)
    g.fig.suptitle(title, fontsize=13, y=1.03)
    plt.setp(g._legend.get_title(), fontsize=12)

    plt.show()


def catplot_with_pct_2(
    df: pd.DataFrame,
    x: str,
    y: str,
    hue: str,
    col: str,
    leg_title,
    subplot_titles: List[str],
    title: str,
) -> plt.figure:
    "Plots catplot with percentage in two subplots"

    g = sns.catplot(
        x=x,
        y=y,
        hue=hue,
        col=col,
        data=df,
        kind="bar",
        sharex=False,
        col_wrap=2,
        ci=None,
        height=7,
        aspect=1,
    )

    for i, ax in enumerate(g.axes.flat):
        ax.set_xlabel("")
        ax.set_title(f"{subplot_titles[i]}s", fontsize=13)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=12)
        if i == 0:
            ax.set_ylabel("")
            ax.tick_params(axis="y", label1On=False)
        for p in ax.patches:
            ax.annotate(
                f"{p.get_height():.2f}%",
                (p.get_x() + p.get_width() / 2.0, p.get_height()),
                ha="center",
                va="center",
                xytext=(0, 5),
                size=11,
                textcoords="offset points",
            )
    g._legend.set_title(leg_title)
    g.fig.suptitle(
        title, fontsize=14, y=1.03,
    )

    plt.show()
    
    
def catplot_with_pct_3(
    df: pd.DataFrame,
    x: str,
    y: str,
    hue: str,
    col: str,
    x_label: str,
    leg_title: str,
    subplot_titles: List[str],
    title: str,
) -> plt.figure:
    "Plots catplot with percentage in two rows"

    g = sns.catplot(
        x=x,
        y=y,
        hue=hue,
        col=col,
        data=df,
        kind="bar",
        sharex=False,
        col_wrap=1,
        ci=None,
        height=6,
        aspect=1.8,
    )

    g._legend.set_title(leg_title)
    g._legend.get_title().set_fontsize(13)
    g._legend.set_bbox_to_anchor((1.15, 0.5))
    for t in g._legend.texts:
        t.set_fontsize(13)
    plt.subplots_adjust(left=-0.1, hspace=0.2)

    for i, ax in enumerate(g.axes.flat):
        ax.set_title(f"{subplot_titles[i]}s", fontsize=13)
        ax.tick_params(axis="x", which="major", labelsize=13)
        ax.set_xlabel(x_label, fontsize=13, labelpad=10)
        ax.set_ylabel("")
        ax.tick_params(axis="y", label1On=False)
        for p in ax.patches:
            ax.annotate(
                f"{p.get_height():.2f}%",
                (p.get_x() + p.get_width() / 2.0, p.get_height()),
                ha="center",
                va="center",
                xytext=(0, 5),
                size=12,
                textcoords="offset points",
            )

    g.fig.suptitle(
        title, fontsize=15, x=0.4, y=1.02,
    )

    plt.show()


def catplot_with_pct_4(
    df: pd.DataFrame,
    x: str,
    y: str,
    hue: str,
    col: str,
    leg_title: str,
    subplot_titles: List[str],
    title: str,
) -> plt.figure:
    "Plots catplot with percentage in two rows"

    g = sns.catplot(
        x=x,
        y=y,
        hue=hue,
        col=col,
        data=df,
        kind="bar",
        sharex=False,
        col_wrap=1,
        ci=None,
        height=6,
        aspect=2,
    )

    g._legend.set_title(leg_title)
    g._legend.get_title().set_fontsize(13)
    g._legend.set_bbox_to_anchor((1.20, 0.55))
    for t in g._legend.texts:
        t.set_fontsize(13)
    plt.subplots_adjust(left=-0.1, hspace=0.3)

    for i, ax in enumerate(g.axes.flat):
        ax.set_title(f"{subplot_titles[i]}s", fontsize=14, y=1.02)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=13)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="y", label1On=False)
        for p in ax.patches:
            ax.annotate(
                f"{p.get_height():.0f}%",
                (p.get_x() + p.get_width() / 2.0, p.get_height()),
                ha="center",
                va="center",
                xytext=(0, 5),
                size=12,
                textcoords="offset points",
            )

    g.fig.suptitle(
        title, fontsize=16, x=0.4, y=1.04,
    )

    plt.show()


def bar_plot(
    data: pd.DataFrame,
    x: str,
    y: str,
    hue: str,
    palette: str,
    title: str,
    leg_title: str = None,
) -> plt.figure:
    "Plots bar plots"

    fig, ax = plt.subplots(figsize=(12, 8))

    sns.barplot(
        x=x,
        y=y,
        hue=hue,
        data=data,
        ci=None,
        palette=palette,
        dodge=False,
        ax=ax,
    )
    if not leg_title:
        leg_title = hue.split("_")[0].capitalize()
    ax.legend(
        title=leg_title,
        fontsize=12,
        loc="center",
        ncol=1,
        bbox_to_anchor=(1.2, 0.5),
        facecolor="white",
        edgecolor="white",
    )

    ax.set_xlabel("")
    ax.set_ylabel(" ".join(y.split("_")).capitalize(), fontsize=12, labelpad=10)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=12)

    plt.title(
        title,
        fontsize=13,
        y=1.02,
    )

    plt.show()


def grouped_bar_plots(
    df: pd.DataFrame, x: str, y: str, hue: str, title: str = None
) -> plt.figure:
    "Plots grouped bar plots"

    g = sns.catplot(
        x=x,
        y=y,
        hue=hue,
        data=df,
        kind="bar",
        ci=None,
        height=8,
        aspect=2.5,
        palette="Set2",
    )

    g._legend.set_title("")

    for t in g._legend.texts:
        t.set_fontsize(14)

    ax = g.axes.flat[0]
    ax.set_xlabel("")
    for p in ax.patches:
        ax.annotate(
            f"{p.get_height():.0f}%",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="center",
            xytext=(4, 5),
            size=14,
            textcoords="offset points",
        )

    g.axes.flat[0].set_xticklabels(
        g.axes.flat[0].get_xticklabels(), rotation=30, ha="right", fontsize=14
    )
    g.axes.flat[0].get_yaxis().set_visible(False)
    g.fig.suptitle(
        title,
        fontsize=18,
        y=1.03,
    )

    plt.show()


def stacked_bars(
    df: pd.DataFrame,
    y1: str,
    y2: str,
    teams_lst: List[str],
    legend_labels_lst: List[str],
    color1: str,
    color2: str,
    title: str,
) -> plt.figure:
    "Plots stacked bar plots"

    fig, ax = plt.subplots(figsize=(12, 8))

    bar1 = sns.barplot(
        x=[i for i in range(df.shape[0])],
        y=y1,
        data=df,
        color=color1,
    )

    bar2 = sns.barplot(
        x=[i for i in range(df.shape[0])],
        y=y2,
        data=df,
        color=color2,
    )

    ax.set_xticklabels(teams_lst, rotation=45, fontsize=12, ha="right")
    ax.set_yticks(range(max(df[y1]) + 1))
    ax.set_ylabel("Goals", fontsize=12, labelpad=30, rotation=0)

    top_bar = mpatches.Patch(color=color1, label=legend_labels_lst[0])
    bottom_bar = mpatches.Patch(color=color2, label=legend_labels_lst[1])

    ax.legend(
        handles=[top_bar, bottom_bar],
        fontsize=12,
        loc="center",
        ncol=1,
        bbox_to_anchor=(1.2, 0.5),
        facecolor="white",
        edgecolor="white",
    )

    ax.set_title(
        title,
        fontsize=13,
        y=1.03,
    )

    plt.show()


def plot_pairgrid(
    df: pd.DataFrame,
    columns: List[str],
    target: str,
    leg_title: str,
    title: str,
    height: float = 4,
    alpha: float = 0.8,
    color_palette: List[str] = ["r", "b"]
) -> plt.figure:
    """Plots pairwise correlation of numerical columns.
    Target identification is added as hue variable"""

    g = sns.PairGrid(
        df,
        vars=df[columns],
        hue=target,
        palette=sns.color_palette(color_palette),
        height=height,
    )
    g.map_diag(sns.histplot)
    g.map_offdiag(sns.scatterplot, alpha=alpha)
    g.add_legend()
    g.legend.set_title(leg_title)
    g.legend.set_bbox_to_anchor((1, 0.5))
    g.fig.suptitle(
        title, fontsize=13, y=1.02,
    )

    plt.show()


def heat_map(
    df: pd.DataFrame,
    title: str = None,
    cbar_title: str = "Correlation Score",
    fig_size: Tuple[int, int] = None,
    vmin: int = -1,
    cmap: str = None,
    fmt=".3f",
) -> plt.figure:
    "Plots a heatmap"

    mask = np.zeros_like(df)
    mask[np.triu_indices_from(mask)] = True
    if fig_size == None:
        fig_size = (14, 10)

    plt.figure(figsize=fig_size)

    ax = sns.heatmap(
        df,
        mask=mask,
        vmin=vmin,
        vmax=1,
        annot=True,
        cmap=cmap,
        fmt=fmt,
        annot_kws={"size": 38 / np.sqrt(len(df))},
    )
    cbar = ax.collections[0].colorbar
    cbar.set_label(cbar_title, labelpad=15)
    ax.figure.axes[-1].yaxis.label.set_size(14)
    ax.tick_params(axis="y", which="major", labelsize=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=12)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha="right", fontsize=12)
    ax.set_title(title, fontsize=15, y=1.03)

    plt.show()


def plot_perm_diffs(
    perm_diffs: np.ndarray, actual_mean_diff: float, title: str
) -> plt.figure:
    """Plots permuted mean differences as well as actual mean difference between
    two distributions"""

    fig, ax = plt.subplots(figsize=(10, 6))

    ax = sns.histplot(data=perm_diffs, kde=False)

    lin = ax.axvline(actual_mean_diff, color="r", lw=2)

    ax.legend(
        [lin],
        ["Actual mean difference"],
        bbox_to_anchor=(1.35, 0.5),
        facecolor="white",
        edgecolor="white",
        fontsize=12,
    )

    plt.xlabel("Difference in means", fontsize=12, labelpad=10)
    plt.ylabel("Count", fontsize=12, labelpad=10)
    plt.title(
        title, fontsize=13, y=1.03,
    )

    plt.show()


def plot_mean_diff_conf_int(
    distribution: np.ndarray,
    lcb: float,
    actual_mean_diff: float,
    ucb: float,
    alpha: float = 0.95,
    color: str = "k",
) -> plt.figure:
    """Plots a sampled distribution from sampling_mean_diff_ci function together
    with lower and upper bounds of sampled distribution condfidence intervals
    and actual mean of real distribution"""

    text_lst = [
        "Lower bound = ",
        "Upper bound = ",
        "Actual mean difference = ",
        "Sampled mean difference = ",
    ]
    dist_mean = distribution.mean()

    var_tup = (lcb, ucb, actual_mean_diff, dist_mean)

    fig, ax = plt.subplots(figsize=(10, 8))

    ax = sns.kdeplot(distribution)

    ax.axvline(lcb, ls="--", color="b")
    ax.axvline(ucb, ls="--", color="b")
    ax.axvline(actual_mean_diff, color="IndianRed")
    ax.axvline(dist_mean, color="y")

    y = ax.get_ylim()[1] * 0.5

    for i in range(len(text_lst)):
        x_text = 4
        if i == 0:
            x_text = -12
        elif i == 2:
            x_text = -25
        elif i == 3:
            x_text = 18
        ax.annotate(
            f"{text_lst[i]}{var_tup[i]:.3f}",
            (var_tup[i], y),
            rotation="90",
            va="center",
            color=color,
            xytext=(x_text, 0),
            size=12,
            textcoords="offset points",
        )

    kde = ax.get_lines()[0].get_data()
    ax.fill_between(
        kde[0],
        kde[1],
        where=(kde[0] > lcb) & (kde[0] < ucb),
        interpolate=True,
        color="Lavender",
    )

    ax.set_title(
        f"Sampled distribution of difference in means with {alpha*100} % CI",
        fontsize=13,
        y=1.01,
    )

    plt.show()