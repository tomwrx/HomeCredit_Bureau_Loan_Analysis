import numpy as np
import pandas as pd
import scipy.stats as stats
from typing import List, Tuple


def two_p_diff_ci(
    series_a_r_count: int, series_b_r_count: int, total_count: int
) -> Tuple[float, ...]:
    """Takes series_a row count, series_b row count and total count of rows
    in a data set and returns difference in two proportions together with
    lower and upper bounds of 95 % confidence interval."""

    first_p = series_a_r_count / total_count
    second_p = series_b_r_count / total_count
    first_p_se = np.sqrt(first_p * (1 - first_p) / total_count)
    second_p_se = np.sqrt(second_p * (1 - second_p) / total_count)
    se_diff = np.sqrt(first_p_se ** 2 + second_p_se ** 2)
    diff_in_p = first_p - second_p
    lcb = diff_in_p - 1.96 * se_diff
    ucb = diff_in_p + 1.96 * se_diff

    return diff_in_p, lcb, ucb


def diff_in_two_means_unpooled_ci(
    series_a: pd.Series, series_b: pd.Series
) -> Tuple[float, ...]:
    """Takes series_a and series_b as an input and returns two series
    mean difference, lower and upper bounds of 95 % confidence interval"""

    s1_sem = series_a.std() / np.sqrt(series_a.count())
    s2_sem = series_b.std() / np.sqrt(series_b.count())
    sem_diff = np.sqrt(s1_sem ** 2 + s2_sem ** 2)
    mean_diff = series_a.mean() - series_b.mean()
    lcb = mean_diff - 2 * sem_diff
    ucb = mean_diff + 2 * sem_diff

    return mean_diff, lcb, ucb


def sampling_mean_diff_ci(
    series_a: pd.Series,
    series_b: pd.Series,
    number_of_samples: int,
    alpha: float = 0.95,
) -> Tuple[np.ndarray, float]:
    """Takes two series, number of samples and alpha parameter and returns
    actual mean difference, array od sampled mean differences, lower and upper bounds
    of confidence interval controled by alpha parameter"""

    actual_mean_diff = series_a.mean() - series_b.mean()
    mean_diff = np.zeros((number_of_samples,))
    for i in range(number_of_samples):
        mean1 = np.random.choice(series_a, series_a.size).mean()
        mean2 = np.random.choice(series_b, series_b.size).mean()
        mean_diff[i] = mean1 - mean2
    quant = (1 - alpha) / 2
    lcb = np.quantile(mean_diff, quant)
    ucb = np.quantile(mean_diff, 1 - quant)

    return actual_mean_diff, mean_diff, lcb, ucb


def sampling_mean(
    series_a: pd.Series, series_b: pd.Series, sample_size: int, number_of_samples: int
) -> Tuple[np.ndarray]:
    """Samples number_of_samples from two series with with given sample_size
    and returns numpy arrays with means"""

    s1 = np.empty(sample_size)
    s2 = np.empty(sample_size)
    for i in range(number_of_samples):
        s1[i] = series_a.sample(sample_size, replace=True).mean()
        s2[i] = series_b.sample(sample_size, replace=True).mean()

    return s1, s2


def perm_test(series_a: pd.Series, series_b: pd.Series) -> float:
    """Makes a permutation test from provided input of two series
    and returns mean difference of permutation result"""

    s1 = series_a.reset_index(drop=True)
    s2 = series_b.reset_index(drop=True)
    data = pd.concat([s1, s2], ignore_index=True)
    total_n = s1.shape[0] + s2.shape[0]
    index2 = set(np.random.choice(range(total_n), s2.shape[0], replace=False))
    index1 = set(range(total_n)) - index2

    return data.iloc[list(index1)].mean() - data.iloc[list(index2)].mean()


def two_proportions_ztest(success: List[int], sample_sizes: List[int]):
    "Returns z statistic and p value of two different group proportions z test"

    diff_in_proportions = (success[0] / sample_sizes[0]) - (
        success[1] / sample_sizes[1]
    )
    prop_comb = (
        (success[0] / sample_sizes[0]) * sample_sizes[0]
        + (success[1] / sample_sizes[1]) * sample_sizes[1]
    ) / (sample_sizes[0] + sample_sizes[1])
    va = prop_comb * (1 - prop_comb)
    se = np.sqrt(va * (1 / sample_sizes[0] + 1 / sample_sizes[1]))
    test_stat = diff_in_proportions / se
    pvalue = 2 * stats.norm.cdf(-np.abs(test_stat))

    return test_stat, pvalue