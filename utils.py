import pandas as pd
from ml_utils.constants import PLOT_SIZE

def get_not_nans_fraction_by_column(df: pd.DataFrame) -> pd.Series:
    filled_dict = {}
    for column in df.columns:
        filled_dict[column] = df[column].count() / df[column].shape[0]
    return pd.Series(filled_dict)

def barplot(series: pd.Series) -> None:
    f, ax = plt.subplots(figsize=PLOT_SIZE) # set the size that you'd like (width, height)
    # plt.bar(height = perc, x = perc.index)
    series.plot.bar()