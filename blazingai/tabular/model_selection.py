from typing import Generator, Optional

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_ns_dtype
from sklearn.model_selection._split import _BaseKFold


class TimeSeriesSplit(_BaseKFold):
    def __init__(
        self,
        method: str,
        training_window: int,
        forecasting_window: int,
        forecasting_horizon: int,
        sliding_steps: int,
        n_splits: int = 3,
        date_col_name: str = "date",
    ):
        """
        Args:
            method (str): Either `sliding` or `expanding`.
            training_window (int): Number of days to use for training.
                Note: a training window of 10 days could include features with
                a lookback that extend the training window size.
            forecasting_window (int): Number of days to use for validation.
            forecasting_horizon (int): The length of the buffer between the
                end of the training period and the start of forecasting period.
                This is generally useful to prevent data leakage.
            sliding_steps (int): Number of days between training date start
                between two nearby folds.
            n_splits (int): Number of folds.
            date_col_name (int): Index of the column containing
            date information.
        """
        self.method = self.validate_method(method)
        self.training_window = training_window
        self.forecasting_window = forecasting_window
        if forecasting_horizon < 1:
            raise ValueError("forecasting_horizon must be a positive number.")
        self.forecasting_horizon = forecasting_horizon
        self.sliding_steps = sliding_steps
        self.n_splits = n_splits
        self.date_col_name = date_col_name
        self.train_date_ranges: dict[str, list[str]] = {}
        self.validation_date_ranges: dict[str, list[str]] = {}

    @staticmethod
    def validate_method(method):
        """Checks if the correct method is passed"""
        if method not in ["expanding", "sliding"]:
            raise ValueError
        else:
            return method

    def split(
        self,
        X: pd.DataFrame,
        y: Optional[np.ndarray] = None,
        groups: Optional[str] = None,
    ) -> Generator[
        tuple[
            np.ndarray,
            np.ndarray,
        ],
        None,
        None,
    ]:
        """
        Args:
            X (pandas.DataFrame): Input data.
            method (str): Either `sliding` or `expanding`.
            y: Unused, but required for compatibility with sklearn cross validation API.
            groups: Unused, but required for compatibility with sklearn cross validation API.
        Returns:
            (generator) Indexes for the training and Validation set from the
                data passed
        """
        self._is_valid_df(X)
        arr = self._df_to_array(X)
        self._has_enough_days(arr)
        first = arr.min()

        # yield indexes for train and valid sets
        for step in range(self.n_splits):
            train_start, train_end, valid_start, valid_end = self._get_dates(
                first=first, step=step
            )
            train_mask = (arr >= train_start) & (arr <= train_end)
            valid_mask = (arr >= valid_start) & (arr <= valid_end)

            self.train_date_ranges["period_" + str(step)] = [
                train_start.date().isoformat(),
                train_end.date().isoformat(),
            ]
            self.validation_date_ranges["period_" + str(step)] = [
                valid_start.date().isoformat(),
                valid_end.date().isoformat(),
            ]
            yield (np.where(train_mask)[0], np.where(valid_mask)[0])

    def _get_dates(self, first: pd.Timestamp, step: int) -> list[pd.Timestamp]:
        """Gets the train start, train end, valid start and valid end dates
        Args:
            first: start date of first training set.
            step: the number of split
        Returns:
            list of training and validation start and end dates.
        """
        train_start, train_end = self._get_train_start_end_dates(
            self.method, first, step
        )
        valid_start = train_end + np.timedelta64(self.forecasting_horizon + 1, "D")
        valid_end = valid_start + np.timedelta64(self.forecasting_window - 1, "D")
        return [train_start, train_end, valid_start, valid_end]

    def _is_valid_df(self, X: pd.DataFrame):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input is not a pandas DataFrame")
        self._has_exactly_one_date_column(X)
        self._if_date_otherthan_timestamp(X)

    def _has_exactly_one_date_column(self, X: pd.DataFrame):
        r = [col for col in X.columns if col == self.date_col_name]
        if not len(r) == 1:
            raise ValueError(
                "Input DataFrame has more than one column named "
                f"{self.date_col_name}"
            )

    def _if_date_otherthan_timestamp(self, X: pd.DataFrame):
        if not is_datetime64_ns_dtype(X[self.date_col_name]):
            raise ValueError(
                f"{self.date_col_name} does not have datetime64[ns] format"
            )

    def _df_to_array(self, X: pd.DataFrame):
        # covert to array
        ix = X.columns.get_loc(self.date_col_name)
        # cast dates to np.datetime64
        ar = X.iloc[:, ix].astype("datetime64[us]")
        return ar

    def _has_enough_days(self, ar):
        """To check if the data has enough days depending on the values of the
        arguments passed
        """
        first, last = ar.min(), ar.max()
        days = (last - first).days + 1  # incl. last day
        required_days = ((self.n_splits - 1) * self.sliding_steps) + (
            self.training_window + self.forecasting_horizon + self.forecasting_window
        )
        if required_days > days:
            raise ValueError(
                "Not enough days in the time series, ",
                f"{required_days} > {days}",
            )

    def _get_train_start_end_dates(self, method: str, first: pd.Timestamp, step: int):
        """To get the start and end date for each period.
        Args:
            first: start date for each training period.
            step: number identifying the n_split
        """
        if method == "sliding":
            train_start = first + np.timedelta64(self.sliding_steps * step, "D")
            train_end = train_start + np.timedelta64(self.training_window - 1, "D")
        else:
            train_start = first
            train_end = (
                train_start
                + np.timedelta64(self.sliding_steps * step, "D")
                + np.timedelta64(self.training_window - 1, "D")
            )
        return train_start, train_end
