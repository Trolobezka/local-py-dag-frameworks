import logging
import re
import typing

import kedro.pipeline
import pandas
import sklearn.linear_model

LoaderType = typing.Callable[[], pandas.DataFrame]
logger = logging.getLogger(__name__)
date_regex = re.compile(r"\d{4}-\d{2}-\d{2}")


def convert(data: dict[str, LoaderType]) -> dict[str, pandas.DataFrame]:
    # In the case of partitioned dataset, the input
    # is a dictionary of filename and load method pairs.
    valid_names = [name for name in data.keys() if date_regex.match(name)]
    logger.info(f"Converting files: {valid_names}")
    return {name: data[name]() for name in valid_names}


def concat(data: dict[str, LoaderType]) -> pandas.DataFrame:
    valid_names = [name for name in data.keys() if date_regex.match(name)]
    logger.info(f"Concatenating files: {valid_names}")
    dfs = [data[name]() for name in valid_names]
    return pandas.concat(dfs).reset_index(drop=True)


from .helpers import drop_columns_with_missing_values, fit_linear_model, impute_columns_with_mean


def clean(concat: pandas.DataFrame, method: str) -> pandas.DataFrame:
    if method == "drop":
        clean = drop_columns_with_missing_values(concat, threshold=0.2)
    elif method == "impute":
        clean = impute_columns_with_mean(concat)
    else:
        raise NotImplementedError(f"Unknown method: '{method}'")
    return clean


def linear_regression(clean: pandas.DataFrame) -> sklearn.linear_model.LinearRegression:
    model, r2_adj = fit_linear_model(
        X=clean.drop(columns=clean.columns[0]),
        y=clean.drop(columns=clean.columns[1:]),
    )
    logger.info(f"Adjusted R2: {r2_adj:.3f}")
    return model


def create_pipeline() -> kedro.pipeline.Pipeline:
    return kedro.pipeline.pipeline(
        [
            kedro.pipeline.node(
                func=convert,
                inputs="raw_data_points",
                outputs="data_points",
                name=convert.__name__ + "_node",
            ),
            kedro.pipeline.node(
                func=concat,
                inputs="data_points",
                outputs="concat_data",
                name=concat.__name__ + "_node",
            ),
            kedro.pipeline.node(
                func=clean,
                inputs={"concat": "concat_data", "method": "params:drop"},
                outputs="clean_drop_data",
                name=clean.__name__ + "_drop" + "_node",
            ),
            kedro.pipeline.node(
                func=clean,
                inputs={"concat": "concat_data", "method": "params:impute"},
                outputs="clean_impute_data",
                name=clean.__name__ + "_impute" + "_node",
            ),
            kedro.pipeline.node(
                func=linear_regression,
                inputs="clean_drop_data",
                outputs="linear_regression_drop_model",
                name=linear_regression.__name__ + "_drop" + "_node",
            ),
            kedro.pipeline.node(
                func=linear_regression,
                inputs="clean_impute_data",
                outputs="linear_regression_impute_model",
                name=linear_regression.__name__ + "_impute" + "_node",
            ),
        ]
    )


def register_pipelines() -> dict[str, kedro.pipeline.Pipeline]:
    pipelines: dict[str, kedro.pipeline.Pipeline] = {}
    pipelines["__default__"] = create_pipeline()
    return pipelines
