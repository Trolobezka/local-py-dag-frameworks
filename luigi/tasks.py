import datetime
import enum
import pathlib
import pickle

import pandas
import sklearn.linear_model
import sklearn.metrics

import luigi
from mixins import LoggerMixin, MultiInMixin, SingleInMixin, SingleOutMixin

ROOT = pathlib.Path(__file__).parent.resolve()


class RawDataPoint(luigi.Task):
    date = luigi.DateParameter()

    def output(self) -> luigi.LocalTarget:  # type: ignore
        return luigi.LocalTarget(ROOT / "data" / "raw" / f"{self.date}.xlsx")


class DataPoint(LoggerMixin, SingleInMixin, SingleOutMixin, luigi.Task):
    date = luigi.DateParameter()

    def requires(self) -> RawDataPoint:  # type: ignore
        return RawDataPoint(self.date)

    def run(self) -> None:
        self.log_info(f"Converting '{self.input_path}' to '{self.output_path}'")
        df = pandas.read_excel(self.input_path)
        df.to_parquet(self.output_path)

    def output(self) -> luigi.LocalTarget:  # type: ignore
        return luigi.LocalTarget(ROOT / "data" / "interim" / f"{self.date}.parquet")


class ConcatData(LoggerMixin, MultiInMixin, SingleOutMixin, luigi.Task):
    """"""

    def requires(self) -> list[DataPoint]:  # type: ignore
        excel_files = list((ROOT / "data" / "raw").glob("*.xlsx"))
        dates = [datetime.datetime.strptime(file.stem, "%Y-%m-%d").date() for file in excel_files]
        return [DataPoint(date) for date in dates]

    def run(self) -> None:
        dataframes = [pandas.read_parquet(path) for path in self.input_paths]
        df = pandas.concat(dataframes).reset_index(drop=True)
        df.to_parquet(self.output_path)

    def output(self) -> luigi.LocalTarget:  # type: ignore
        return luigi.LocalTarget(ROOT / "data" / "interim" / f"concat.parquet")


class CleanDataMethod(enum.Enum):
    drop = enum.auto()
    impute = enum.auto()


def drop_columns_with_missing_values(data: pandas.DataFrame, threshold: float) -> pandas.DataFrame:
    column_mask = (data.isna().sum(axis="index") / data.shape[0]) > threshold
    data = data.drop(columns=data.columns[column_mask])
    if data.shape[1] == 0:
        raise ValueError("No columns left after dropping")
    return data


def impute_columns_with_mean(data: pandas.DataFrame) -> pandas.DataFrame:
    data = data.fillna(data.mean(numeric_only=True))
    return data


class CleanData(LoggerMixin, SingleInMixin, SingleOutMixin, luigi.Task):
    method = luigi.EnumParameter(enum=CleanDataMethod)  # drop, impute

    def requires(self) -> ConcatData:  # type: ignore
        return ConcatData()

    def run(self) -> None:
        concat = pandas.read_parquet(self.input_path)
        if self.method == CleanDataMethod.drop:
            clean = drop_columns_with_missing_values(concat, threshold=0.2)
        elif self.method == CleanDataMethod.impute:
            clean = impute_columns_with_mean(concat)
        else:
            raise NotImplementedError(f"Unknown method: '{self.method}'")
        clean.to_parquet(self.output_path)

    def output(self) -> luigi.LocalTarget:  # type: ignore
        return luigi.LocalTarget(ROOT / "data" / "processed" / f"clean_{self.method.name}.parquet")


def adjusted_r2_score(r2: float, n: int, p: int) -> float:
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)


def fit_linear_model(
    X: pandas.DataFrame, y: pandas.DataFrame
) -> tuple[sklearn.linear_model.LinearRegression, float]:
    model = sklearn.linear_model.LinearRegression()
    model.fit(X, y)
    r2 = sklearn.metrics.r2_score(y, model.predict(X))
    r2_adj = adjusted_r2_score(r2=r2, n=X.shape[0], p=X.shape[1])
    return (model, r2_adj)


class LinearRegression(LoggerMixin, SingleInMixin, SingleOutMixin, luigi.Task):
    method = luigi.EnumParameter(enum=CleanDataMethod)  # drop, impute

    def requires(self) -> CleanData:  # type: ignore
        return CleanData(method=self.method)

    def run(self) -> None:
        data = pandas.read_parquet(self.input_path)
        model, r2_adj = fit_linear_model(
            X=data.drop(columns=data.columns[0]),
            y=data.drop(columns=data.columns[1:]),
        )
        self.log_info(f"Adjusted R2: {r2_adj}")
        self.output_path.write_bytes(pickle.dumps(model))

    def output(self) -> luigi.LocalTarget:  # type: ignore
        return luigi.LocalTarget(ROOT / "models" / f"linear_regression_{self.method.name}.pkl")


if __name__ == "__main__":
    pass
