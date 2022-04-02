import os
import zipfile
import urllib.request
import numpy as np
import pandas as pd
import gc
from typing import Tuple, Dict, List, Callable


def fetch_data(url: str, path: str, filename: str) -> None:
    "Creates a directory for a data set, downloads and extracts files there"

    if not os.path.isdir(path):
        os.makedirs(path)
    zip_path = os.path.join(path, filename)
    urllib.request.urlretrieve(url, zip_path)
    file = zipfile.ZipFile(zip_path)
    file.extractall(path=zip_path)
    file.close()

def calc_ratio(group: pd.DataFrame) -> pd.DataFrame:
    "Takes a grouped pandas DataFrame and returns DataFrame"

    group["ratio"] = (group["size"] / group["size"].sum() * 100).round(2)

    return group


def find_outliers(data_frame: pd.DataFrame, factor: float) -> Dict[str, str]:
    """Finds outliers in DataFrame and returns a dictionary with column names
    where outliers were found"""

    outliers_dict = {}
    for column in data_frame.columns:
        if data_frame[column].dtype not in [float, int]:
            continue
        if np.any(
            data_frame[column]
            > (data_frame[column].quantile(0.75) - data_frame[column].quantile(0.25))
            * factor
            + data_frame[column].quantile(0.75)
        ) or np.any(
            data_frame[column]
            < data_frame[column].quantile(0.25)
            - (data_frame[column].quantile(0.75) - data_frame[column].quantile(0.25))
            * factor
        ):
            outliers_dict[column] = True

    return outliers_dict


def quartiles_sorting(series: pd.Series, measurement: str) -> Tuple[pd.Series, ...]:
    """Takes series and returns a tuple of series with quartiles in a first series
    and labels with added measurement e.g. age, kg, cm and etc. in a second series"""
    
    quartile_list = [0, 0.25, 0.5, 0.75, 1.0]
    series_a = pd.qcut(series, quartile_list)
    quartile_labels = [
        f"{series_a.cat.categories[i].left:.0f} - {series_a.cat.categories[i].right:.0f} {measurement}"
        for i in range(len(series_a.cat.categories))
    ]
    series_b = pd.qcut(series, quartile_list, quartile_labels)

    return series_a, series_b


def cut_sorting(
    series: pd.Series, bins: List[float], measurement: str, bin_labels=None
) -> Tuple[pd.Series, ...]:
    """Takes series and returns a tuple of series with pd.cut binned data in a first series
    and labels with added measurement e.g. age, kg, cm and etc. in a second series"""

    series_a = pd.cut(series, bins, right=False)
    if bin_labels is not None:
        bin_labels = bin_labels
    else:
        bin_labels = [
            f"{series_a.cat.categories[i].left:.0f} {measurement}"
            if i != len(series_a.cat.categories) - 1
            else f"{series_a.cat.categories[i].left:.0f} or more {measurement}"
            for i in range(len(series_a.cat.categories))
        ]
    series_b = pd.cut(series, bins, labels=bin_labels, right=False)

    return series_a, series_b


def drop_nan(
    data: pd.DataFrame, target: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    "Drops nan values in a dataframe and also same index in a target series"

    max_col = data.isnull().sum().idxmax()
    drop_idx = data[data[max_col].isnull() == True].index
    data = data.drop(drop_idx)
    target = target.drop(drop_idx)

    return data, target


def drop_nan_features(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    "Drops features with nan values exceeding a threshold in a dataframe "

    mask = ((df.isnull().sum() / df.shape[0]) * 100) > threshold
    cols_to_drop = [col for col in mask.index if mask[col] == True]
    df = df.drop(cols_to_drop, axis=1)

    return df


def agg_numeric_features(
    df: pd.DataFrame, group_var: str, df_name: str
) -> pd.DataFrame:
    """Aggregates the numeric values in a dataframe. This can
    be used to create features for each instance of the grouping variable.
    """

    for col in df:
        if col != group_var and "SK_ID" in col:
            df = df.drop(columns=col)

    numeric_df = df.select_dtypes("number").copy()

    agg = (
        numeric_df.groupby(group_var)
        .agg(["count", "mean", "max", "min", "sum"])
        .reset_index()
    )

    columns = [group_var]
    for var in agg.columns.levels[0]:
        if var != group_var:
            for stat in agg.columns.levels[1][:-1]:
                columns.append(f"{df_name}_{var.lower()}_{stat}")

    agg.columns = columns
    col_to_drop = get_duplicate_columns(agg)
    agg = agg.drop(col_to_drop, axis=1)
    return agg


def count_categorical_features(
    df: pd.DataFrame, group_var: str, df_name: str
) -> pd.DataFrame:
    """Computes counts and normalized counts for each observation
    of 'group_var' of each unique category in every categorical variable.        
    """

    categorical = pd.get_dummies(df.select_dtypes("object").copy())
    categorical[group_var] = df[group_var]
    categorical = categorical.groupby(group_var).agg(["sum", "mean"]).reset_index()

    columns = [group_var]

    for var in categorical.columns.levels[0]:
        if var != group_var:
            for stat in ["count", "count_norm"]:
                columns.append(f"{df_name}_{var.lower()}_{stat}")

    categorical.columns = columns
    col_to_drop = get_duplicate_columns(categorical)
    categorical = categorical.drop(col_to_drop, axis=1)
    return categorical


def mode_categorical_features(
    df: pd.DataFrame, group_var: str, df_name: str
) -> pd.DataFrame:
    "Returns mode of each categorical features from the dataframe"

    for col in df:
        if col != group_var and "SK_ID" in col:
            df = df.drop(columns=col)

    categorical = df.select_dtypes("object").copy()
    categorical[group_var] = df[group_var]
    categorical = (
        categorical.groupby(group_var).agg(lambda x: pd.Series.mode(x)[0]).reset_index()
    )

    columns = [group_var]

    for var in categorical.columns:
        if var != group_var:
            columns.append(f"{df_name}_{var.lower()}_mode")

    categorical.columns = columns
    return categorical


def nunique_categorical_features(
    df: pd.DataFrame, group_var: str, df_name: str
) -> pd.DataFrame:
    "Count unique appearance in each categorical feature from the dataframe"

    for col in df:
        if col != group_var and "SK_ID" in col:
            df = df.drop(columns=col)

    categorical = df.select_dtypes("object").copy()
    categorical[group_var] = df[group_var]
    categorical = categorical.groupby(group_var).agg("nunique").reset_index()

    columns = [group_var]

    for var in categorical.columns:
        if var != group_var:
            columns.append(f"{df_name}_{var.lower()}_unique")

    categorical.columns = columns
    return categorical


def agg_client_level(
    df: pd.DataFrame, group_vars: List[str], df_names: List[str]
) -> pd.DataFrame:
    """Aggregate a dataframe with data first at the loan level and then at the client level"""

    df_agg = agg_numeric_features(df, group_vars[1], df_names[1])

    if any(df.dtypes == "string") or any(df.dtypes == "object"):
        df_counts = count_categorical_features(df, group_vars[1], df_names[1])
        df_by_loan = df_counts.merge(df_agg, how="outer", on=group_vars[1])

        gc.enable()
        del df_agg, df_counts
        gc.collect()

        df_by_loan = df_by_loan.merge(df[group_vars], on=group_vars[1], how="left")
        df_by_loan = df_by_loan.drop(
            df_by_loan[df_by_loan.duplicated(subset=[group_vars[1]]) == True].index
        )
        df_by_loan = df_by_loan.drop(columns=[group_vars[1]])
        df_by_client = agg_numeric_features(df_by_loan, group_vars[0], df_names[0])

    else:
        df_by_loan = df_agg.merge(df[group_vars], on=group_vars[1], how="left")

        gc.enable()
        del df_agg
        gc.collect()

        df_by_loan = df_by_loan.drop(
            df_by_loan[df_by_loan.duplicated(subset=[group_vars[1]]) == True].index
        )
        df_by_loan = df_by_loan.drop(columns=[group_vars[1]])
        df_by_client = agg_numeric_features(
            df_by_loan, group_vars[0], df_name=df_names[0]
        )

    col_to_drop = get_duplicate_columns(df_by_client)
    df_by_client = df_by_client.drop(col_to_drop, axis=1)
    return df_by_client


def reduce_memory_usage(df: pd.DataFrame, verbose: str = True) -> pd.DataFrame:
    """Reduces memory usage of pandas dataframe by casting numeric
    columns to lowest possible int or float type"""
    
    numerics = ["int8", "int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage().sum() / 1024 ** 2
    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if (
                    c_min > np.finfo(np.float16).min
                    and c_max < np.finfo(np.float16).max
                ):
                    df[col] = df[col].astype(np.float16)
                elif (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose:
        print(
            "Mem. usage decreased to {:.2f} Mb ({:.1f}% reduction)".format(
                end_mem, 100 * (start_mem - end_mem) / start_mem
            )
        )
    return df


def reduce_memory_usage_2(df: pd.DataFrame, verbose: str = True) -> pd.DataFrame:
    """Reduces memory usage of pandas dataframe by casting numeric
    columns to lowest possible int or float type"""

    numerics = ["int8", "int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage().sum() / 1024 ** 2

    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            if str(col_type)[:3] == "int":
                df[col] = df[col].astype(np.int32)
            else:
                df[col] = df[col].astype(np.float32)

    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose:
        print(
            "Mem. usage decreased to {:.2f} Mb ({:.1f}% reduction)".format(
                end_mem, 100 * (start_mem - end_mem) / start_mem
            )
        )
    return df


def create_reversed_feature(series: pd.Series) -> pd.Series:
    """From input series creates a new feature of series**(-1)
    and return it as series"""

    rev_series = np.power(series, -1)
    return rev_series


def create_log_feature(series: pd.Series) -> pd.Series:
    """From input series creates a new feature of log(series)
    and return it as series"""

    log_series = np.log((series + 1e-4))
    return log_series


def create_squared_feature(series: pd.Series) -> pd.Series:
    """From input series creates a new feature of series**2
    and return it as series"""

    sq_series = np.power(series, 2)
    return sq_series


def create_cubic_feature(series: pd.Series) -> pd.Series:
    """From input series creates a new feature of series**3
    and return it as series"""

    cub_series = np.power(series, 3)
    return cub_series


def create_product_feature(series_a: pd.Series, series_b: pd.Series) -> pd.Series:
    """From input series A and B creates new feature A*B
    and return it as series"""

    product_series = np.multiply(series_a, series_b)
    return product_series


def create_division_feature(series_a: pd.Series, series_b: pd.Series) -> pd.Series:
    """From input series A and B creates new feature A/B
    and return it as series"""

    div_series = np.divide(series_a, (series_b + 1e-4))
    return div_series


def create_addition_feature(series_a: pd.Series, series_b: pd.Series) -> pd.Series:
    """From input series A and B creates new feature A+B
    and return it as series"""

    add_series = np.add(series_a, series_b)
    return add_series


def create_subtraction_feature(series_a: pd.Series, series_b: pd.Series) -> pd.Series:
    """From input series A and B creates new feature A-B
    and return it as series"""

    sub_series = np.subtract(series_a, series_b)
    return sub_series


def single_column_transformation(
    data_frame: pd.DataFrame, target_series: pd.Series, functions_list: List[Callable]
) -> pd.DataFrame:
    """From input data_frame creates a new features via provided transforming
    function list. Returns a new dataframe with transformed features."""

    df = data_frame.copy()
    new_features_lst = []
    new_features_col_names = []

    for i in range(len(functions_list)):
        for j in range(len(df.columns)):
            if df.iloc[:, j].dtype != float:
                df.loc[:, df.columns[j]] = df.loc[:, df.columns[j]].astype(float)
            new_feature = functions_list[i](df.iloc[:, j])

            original_correlation = np.abs(df.iloc[:, j].corr(target_series))
            new_feature_correlation = np.abs(new_feature.corr(target_series))

            if new_feature_correlation > original_correlation:
                new_feature_name = (
                    f"{df.columns[j]}_{str(functions_list[i]).split('_')[-2]}"
                )
                new_features_lst.append(new_feature)
                new_features_col_names.append(new_feature_name)

    return pd.concat(new_features_lst, axis=1, keys=new_features_col_names)


def pairwise_transformations(
    data_frame: pd.DataFrame, target_series: pd.Series, functions_list: List[Callable]
) -> pd.DataFrame:
    """From input data_frame creates a new features via provided transforming
    function list. Returns a new dataframe with transformed features."""

    df = data_frame.copy()
    new_features_lst = []
    new_features_col_names = []

    for i in range(len(functions_list)):
        for j in range(len(df.columns)):
            for k in range(len(df.columns)):
                if df.columns[j] != df.columns[k]:
                    if df.iloc[:, j].dtype != float or df.iloc[:, k].dtype != float:
                        df.loc[:, df.columns[j]] = df.loc[:, df.columns[j]].astype(
                            float
                        )
                        df.loc[:, df.columns[k]] = df.loc[:, df.columns[k]].astype(
                            float
                        )

                    new_feature = functions_list[i](df.iloc[:, j], df.iloc[:, k])

                    original_correlation1 = np.abs(df.iloc[:, j].corr(target_series))
                    original_correlation2 = np.abs(df.iloc[:, k].corr(target_series))
                    bigest_original_corr = max(
                        original_correlation1, original_correlation2
                    )
                    new_feature_correlation = np.abs(new_feature.corr(target_series))

                    if new_feature_correlation > bigest_original_corr:
                        new_feature_name = f"{df.columns[j]}_{str(functions_list[i]).split('_')[-2]}_{df.columns[k]}"
                        new_features_lst.append(new_feature)
                        new_features_col_names.append(new_feature_name)

    return pd.concat(new_features_lst, axis=1, keys=new_features_col_names)


def get_duplicate_columns(data_frame: pd.DataFrame) -> List[str]:
    """From input dataframe finds columns which are the same
    content wise and returns those columns names as a list."""

    duplicate_column_names = set()

    for i in range(data_frame.shape[1]):
        col = data_frame.iloc[:, i]
        for j in range(i + 1, data_frame.shape[1]):
            other_col = data_frame.iloc[:, j]

            if col.equals(other_col):
                duplicate_column_names.add(data_frame.columns.values[j])

    return list(duplicate_column_names)

def highlight_min(s: pd.Series) -> pd.Series:
    """Takes a row from pandas series and colors a cell if value in a
    cell is min value in an entire row"""

    is_min = s == s.min()
    return ["background-color: steelblue" if v else "" for v in is_min]


def highlight_max(s: pd.Series) -> pd.Series:
    """Takes a row from pandas series and colors a cell if value in a
    cell is max value in an entire row"""

    is_max = s == s.max()
    return ["background-color: steelblue" if v else "" for v in is_max]


def get_state_region(row: pd.Series) -> pd.Series:
    "Helper function for US States binning"

    northeast = [
        "Connecticut",
        "Maine",
        "Massachusetts",
        "New Hampshire",
        "Rhode Island",
        "Vermont",
        "New York",
        "New Jersey",
        "Pennsylvania",
    ]
    midwest = [
        "Illinois",
        "Indiana",
        "Michigan",
        "Ohio",
        "Wisconsin",
        "Iowa",
        "Kansas",
        "Minnesota",
        "Missouri",
        "Nebraska",
        "North Dakota",
        "South Dakota",
    ]
    south = [
        "Delaware",
        "Florida",
        "Georgia",
        "Maryland",
        "North Carolina",
        "South Carolina",
        "Virginia",
        "District of Columbia",
        "West Virginia",
        "Alabama",
        "Kentucky",
        "Mississippi",
        "Tennessee",
        "Arkansas",
        "Louisiana",
        "Oklahoma",
        "Texas",
    ]
    west = [
        "Arizona",
        "Colorado",
        "Idaho",
        "Montana",
        "Nevada",
        "New Mexico",
        "Utah",
        "Wyoming",
        "Alaska",
        "California",
        "Hawaii",
        "Oregon",
        "Washington",
    ]

    if row in northeast:
        return "Northeast"
    elif row in midwest:
        return "Midwest"
    elif row in south:
        return "South"
    else:
        return "West"
    
