import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Union
from matplotlib import pyplot as plt


SCATTERPLOT_SIZE = (10, 10)


def get_not_nans_fraction_by_column(df: pd.DataFrame) -> pd.Series:
    filled_dict = {}
    for column in df.columns:
        filled_dict[column] = df[column].count() / df[column].shape[0]
    return pd.Series(filled_dict)


def num_cat_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Defines which columns are numerical and which are categorical and returns
    numerical and categorical dataframes."""
    categorical = sort_df_columns(df.select_dtypes(include='object'))
    numerical = sort_df_columns(df.select_dtypes(exclude='object'))
    print(f'Numerical dtypes are {list(numerical.dtypes.apply(str).unique())}')
    print(f'Numerical columns are {list(numerical.columns)}')
    print('\n')
    print(f'Categorical dtypes are {list(categorical.dtypes.apply(str).unique())}')
    print(f'Categorical columns are {list(categorical.columns)}')
    return numerical, categorical


def sort_df_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df[sorted(df.columns)]


def true_vs_prediction_scatter_plot(true: Union[pd.Series, np.ndarray],
                                    prediction: Union[pd.Series, np.ndarray]) -> None:
    fig = plt.figure(figsize=SCATTERPLOT_SIZE)
    ax = fig.add_subplot(111, aspect='equal')
    max_value = max(prediction.max(), true.max())
    ax.set_xlim((0, max_value))
    ax.set_ylim((0, max_value))
    plt.plot([0, max_value], [0, max_value], 'k-', color = 'r')
    ax.set_aspect(1)
    ax.grid(b=True, which='major', color='k', linestyle='--')
    ax.scatter(prediction, true)
    plt.show()