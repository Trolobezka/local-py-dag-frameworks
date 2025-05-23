import enum
import hashlib
from pathlib import Path

import pandas
import sklearn.linear_model
import sklearn.metrics

import prefect
import prefect.cache_policies
import prefect.context


def parameters_to_paths(context: prefect.context.TaskRunContext) -> list[Path]:
    logger = prefect.get_run_logger(context)
    parameters = context.parameters
    if "input_paths" in parameters:
        paths: list[Path] = parameters["input_paths"]  # type: ignore
    elif "input_path" in parameters:
        paths: list[Path] = [parameters["input_path"]]  # type: ignore
    else:
        msg = "No input path(s) found in parameters"
        logger.error(msg)
        raise Exception(msg)
    return sorted(paths)


def hash_mtime(context: prefect.context.TaskRunContext, parameters: dict[str, object]) -> str:
    logger = prefect.get_run_logger(context)
    paths = parameters_to_paths(context)
    _hash = hashlib.blake2b(digest_size=64)
    for path in paths:
        try:
            mtime = path.stat().st_mtime
            logger.debug(f"File '{path}' mtime: {mtime}")
            _hash.update(str(mtime).encode("utf-8"))
        except FileNotFoundError:
            msg = f"File '{path}' not found while checking mtime"
            logger.error(msg)
            raise FileNotFoundError(msg)
    return _hash.hexdigest()


def hash_bytes(context: prefect.context.TaskRunContext, parameters: dict[str, object]) -> str:
    logger = prefect.get_run_logger(context)
    paths = parameters_to_paths(context)
    _hash = hashlib.blake2b(digest_size=64)
    for path in paths:
        try:
            with path.open("rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    _hash.update(chunk)
            logger.debug(f"File '{path}' read")
        except FileNotFoundError:
            msg = f"File '{path}' not found while checking mtime"
            logger.error(msg)
            raise FileNotFoundError(msg)
    return _hash.hexdigest()


MTIME = prefect.cache_policies.CachePolicy.from_cache_key_fn(hash_mtime)
BYTES = prefect.cache_policies.CachePolicy.from_cache_key_fn(hash_bytes)


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


def adjusted_r2_score(r2: float, n: int, p: int) -> float:
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)


def fit_linear_model(
    X: pandas.DataFrame, y: pandas.DataFrame
) -> tuple[sklearn.linear_model.LinearRegression, float]:
    model = sklearn.linear_model.LinearRegression()
    model.fit(X, y)
    r2 = sklearn.metrics.r2_score(y, model.predict(X))
    r2_adj = adjusted_r2_score(r2=r2, n=X.shape[0], p=X.shape[1])  # type: ignore
    return (model, r2_adj)
