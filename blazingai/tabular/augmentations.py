from typing import List

import numpy as np
import pandas as pd


def extract_datetime_features(
    df: pd.DataFrame,
    col_names: List[str],
    is_datetime: bool = True,
    drop_original_cols: bool = False,
) -> pd.DataFrame:
    """Extract datetime features from pd.Series.

    Args:
        df: A DataFrame containing the series we want to augment.
        col_names: The names of the columns we want to augment. These must be
            pandas.Timestamp series.
        is_datetime: Whether the series contain dates or datetimes.
        drop_original_cols: Whether to drop the original columns from the
            returned DataFrame.
    """
    date_features = [
        "year",
        "quarter",
        "month",
        "week",
        "day",
        "day_of_week",
        "day_of_year",
        "is_month_start",
        "is_month_end",
        "is_quarter_start",
        "is_quarter_end",
        "is_year_start",
        "is_year_end",
        "is_leap_year",
        "days_in_month",
    ]
    datetime_features = [
        "time",
        "timetz",
        "hour",
        "minute",
        "second",
        "microsecond",
        "nanosecond",
        "tz",
    ]
    for col_name in col_names:
        for f in date_features:
            df[f] = getattr(df[col_name].dt, f)
        if is_datetime:
            for f in datetime_features:
                df[f] = getattr(df[col_name].dt, f)
    if drop_original_cols:
        df = df.drop(col_names, axis=1)
    return df


def get_cyclical_features(
    df: pd.DataFrame,
    col_names: List[str],
    scaling_factor: int,
    drop_original_cols: bool = False,
) -> pd.DataFrame:
    for col_name in col_names:
        # make sure min value for column is 0 before applying sin/cos
        # transformation
        col_min_value = df[col_name].min()
        if col_min_value > 1:
            raise ValueError(f"Min value for {col_name} is greater than 1.")
        if col_min_value == 1:
            df[col_name] = df[col_name] - 1

        df[f"{col_name}_sin"] = np.sin(
            df[col_name] * (2.0 * np.pi / scaling_factor)
        ).round(2)
        df[f"{col_name}_cos"] = np.cos(
            df[col_name] * (2.0 * np.pi / scaling_factor)
        ).round(2)

    if drop_original_cols:
        df = df.drop(col_names, axis=1)
    return df
