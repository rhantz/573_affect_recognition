"""
Module to handle class_imbalance after vector creation

Available methods:

Under-Sampling:
    `x_new, y_new = random_under_sample(x, y, kwargs)`  randomly under-samples

Over-Sampling:
    `x_new, y_new = random_over_sample(x, y, kwargs)`   randomly over-samples
    `x_new, y_new = smote(x, y, kwargs)`                randomly adds synthetic data

Combination:
    `x_new, y_new = random_over_under_sample(x, y)`     randomly over AND under-samples each class to the average
    `x_new, y_new = smote_tomek(x, y, kwargs)`          randomly adds synthetic data and cleans Tomek links

Each function takes the initial x and y vector, label lists, performs a class imbalance strategy
and returns the new lists. Functions that take kwargs, can optionally specify additional arguments,
particularly useful could be dictionaries specifying the desired resulting class distribution:

    e.g. Passing `sampling_strategy = {"anger": 400, "fear": 200}` into random_over_sample() would
    ONLY oversample anger to have 400 instances and fear to 200 instances.
"""

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.combine import SMOTETomek
import pandas as pd
from statistics import mean
from collections import Counter

from preprocess import update_data
from create_vectors import build_vectors_emobow


def random_under_sample(x_vectors: list, y_labels: list, **kwargs) -> tuple:
    """
    Given a list of x vectors and their corresponding y labels, randomly undersamples the majority class(es)
    Args:
        x_vectors: list of vectors
        y_labels: list of labels
        **kwargs: additional arguments to pass into RandomUnderSampler()
            e.g. sampling_strategy = { classes_to_under_sample : sample_size }

    Returns:
        tuple of randomly undersampled vectors and their labels
    """
    rus = RandomUnderSampler(random_state=1, **kwargs)
    x_rus, y_rus = rus.fit_resample(x_vectors, y_labels)
    return x_rus, y_rus


def random_over_sample(x_vectors: list, y_labels: list, **kwargs) -> tuple:
    """
    Given a list of x vectors and their corresponding y labels, randomly oversamples the minority class(es)
    Args:
        x_vectors: list of vectors
        y_labels: list of labels
        **kwargs: additional arguments to pass into RandomOverSampler()
            e.g. sampling_strategy = { classes_to_over_sample : sample_size }

    Returns:
        tuple of randomly oversampled vectors and their labels
    """
    ros = RandomOverSampler(random_state=1, **kwargs)
    x_ros, y_ros = ros.fit_resample(x_vectors, y_labels)
    return x_ros, y_ros


def random_over_under_sample(x_vectors: list, y_labels: list) -> tuple:
    """
    Given a list of x vectors and their corresponding y labels, combines random over and undersampling to return
    an averaged amount of vectors for each class. (i.e. undersample majority classes and oversample minority classes)
    Args:
        x_vectors: list of vectors
        y_labels: list of labels

    Returns:
        tuple of randomly under + oversampled vectors and their labels
    """
    class_distribution = Counter(y_labels)
    average_class_size = int(round(mean(class_distribution.values()), 0))

    over_sample_classes = {class_label: average_class_size for class_label, class_size in class_distribution.items()
                           if class_size < average_class_size}
    under_sample_classes = {class_label: average_class_size for class_label, class_size in class_distribution.items()
                            if class_size > average_class_size}

    x_rus, y_rus = random_under_sample(x_vectors, y_labels, sampling_strategy=under_sample_classes)
    x_rous, y_rous = random_over_sample(x_rus, y_rus, sampling_strategy=over_sample_classes)

    return x_rous, y_rous


def smote(x_vectors: list, y_labels: list, **kwargs) -> tuple:
    """
    Uses Synthetic Minority Over-sampling Technique to add synthetic vectors for minority classes

    Args:
        x_vectors: list of vectors
        y_labels: list of labels
        **kwargs: additional arguments to pass into SMOTE()
            e.g. sampling_strategy = { classes_to_over_sample : sample_size }

    Returns:
        tuple of vectors and labels and their synthetic additions
    """

    sm = SMOTE(random_state=1, **kwargs)
    x_sm, y_sm = sm.fit_resample(x_vectors, y_labels)
    return x_sm, y_sm


def smote_tomek(x_vectors: list, y_labels: list, **kwargs) -> tuple:
    """
    Uses Synthetic Minority Over-sampling Technique to add synthetic vectors for minority classes + cleaning
    with Tomek link removal (undersampling)

    "Tomek links are pairs of very close instances (nearest neighbors) but of opposite classes.
    Removing the instances of the majority class of each pair increases the space between the two classes,
    facilitating the classification process."
    - (https://www.analyticsvidhya.com/blog/2020/07/10-techniques-to-deal-with-class-imbalance-in-machine-learning/)

    Args:
        x_vectors: list of vectors
        y_labels: list of labels
        **kwargs: additional arguments to pass into SMOTETomek()
            e.g. sampling_strategy = { classes_to_over_sample : sample_size }

    Returns:
        tuple of vectors and labels and their synthetic additions minus Tomek link removals
    """

    smt = SMOTETomek(random_state=1, **kwargs)
    x_smt, y_smt = smt.fit_resample(x_vectors, y_labels)
    return x_smt, y_smt

