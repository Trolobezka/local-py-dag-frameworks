import pathlib
import pickle

import pandas
import sklearn.linear_model
import sklearn.metrics

import dagster

ROOT = pathlib.Path(__file__).parent


def dataframe_to_metadata(
    data: pandas.DataFrame,
) -> dict[str, dagster.TableSchema | dagster.MetadataValue]:
    schema = dagster.TableSchema(
        [
            dagster.TableColumn(name, column_type.name)
            for name, column_type in zip(data.columns, data.dtypes)
        ]
    )
    return {
        "dagster/column_schema": schema,
        "dagster/row_count": dagster.IntMetadataValue(data.shape[0]),
        "dagster/column_count": dagster.IntMetadataValue(data.shape[1]),
        "preview": dagster.MarkdownMetadataValue(data.head().to_markdown()),
    }


def load_excels_from_dir(directory: pathlib.Path) -> list[pandas.DataFrame]:
    excel_files = list(directory.glob("*.xlsx"))
    dataframes = [pandas.read_excel(file) for file in excel_files]
    return dataframes


@dagster.asset(
    code_version="v1",
    description="Concatenated Excel files from the raw data directory.",
)
def concat_data(context: dagster.AssetExecutionContext) -> pathlib.Path:
    dataframes = load_excels_from_dir(ROOT / "data" / "raw")
    output_path = ROOT / "data" / "interim" / "concat.parquet"
    concat = pandas.concat(dataframes).reset_index(drop=True)
    context.add_output_metadata(dataframe_to_metadata(concat))
    concat.to_parquet(output_path)
    return output_path


def drop_columns_with_missing_values(data: pandas.DataFrame, threshold: float) -> pandas.DataFrame:
    column_mask = (data.isna().sum(axis="index") / data.shape[0]) > threshold
    data = data.drop(columns=data.columns[column_mask])
    if data.shape[1] == 0:
        raise ValueError("No columns left after dropping")
    return data


def impute_columns_with_mean(data: pandas.DataFrame) -> pandas.DataFrame:
    data = data.fillna(data.mean(numeric_only=True))
    return data


clean_partition = dagster.StaticPartitionsDefinition(["drop", "impute"])


@dagster.asset(
    code_version="v1",
    partitions_def=clean_partition,  # ["drop", "impute"]
    description="Concatenated data with columns with missing values removed or imputed.",
)
def clean_data(
    context: dagster.AssetExecutionContext,
    concat_data: pathlib.Path,
) -> pathlib.Path:
    method = context.partition_key  # current partition
    concat = pandas.read_parquet(concat_data)
    if method == "drop":
        clean = drop_columns_with_missing_values(concat, threshold=0.2)
    elif method == "impute":
        clean = impute_columns_with_mean(concat)
    else:
        raise NotImplementedError(f"Unknown partition key: '{method}'")
    output_path = ROOT / "data" / "processed" / f"clean_{method}.parquet"
    context.add_output_metadata(dataframe_to_metadata(clean))
    clean.to_parquet(output_path)
    return output_path


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


@dagster.asset(
    code_version="v1",
    partitions_def=clean_partition,  # ["drop", "impute"]
    description="Linear regression model fitted to the clean dataset.",
)
def linear_regression(
    context: dagster.AssetExecutionContext,
    clean_data: pathlib.Path,
) -> pathlib.Path:
    method = context.partition_key
    data = pandas.read_parquet(clean_data)
    model, r2_adj = fit_linear_model(
        X=data.drop(columns=data.columns[0]),
        y=data.drop(columns=data.columns[1:]),
    )
    context.add_output_metadata({"R2": dagster.FloatMetadataValue(r2_adj)})
    output_path = ROOT / "models" / f"linear_regression_{method}.pkl"
    output_path.write_bytes(pickle.dumps(model))
    return output_path


definitions = dagster.Definitions(
    assets=[concat_data, clean_data, linear_regression],
)
