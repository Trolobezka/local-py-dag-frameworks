import pickle
import pprint
from pathlib import Path

import pandas

import prefect
import prefect.artifacts
from helpers import (
    BYTES,
    MTIME,
    CleanDataMethod,
    add_table_to_artifact,
    drop_columns_with_missing_values,
    fit_linear_model,
    impute_columns_with_mean,
)
from prefect.cache_policies import INPUTS, TASK_SOURCE

ROOT = Path(__file__).parent
RAW_DIR = ROOT / "data" / "raw"
INTERIM_DIR = ROOT / "data" / "interim"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"


@prefect.task(cache_policy=INPUTS + TASK_SOURCE + MTIME + BYTES)
def convert(input_path: Path) -> Path:
    data = pandas.read_excel(input_path)
    output_path = INTERIM_DIR / f"{input_path.stem}.parquet"
    data.to_parquet(output_path)
    add_table_to_artifact(data, key="table")
    return output_path


@prefect.task(cache_policy=INPUTS + TASK_SOURCE + MTIME + BYTES)
def concat(input_paths: list[Path]) -> Path:
    dataframes = [pandas.read_parquet(path) for path in input_paths]
    output_path = INTERIM_DIR / "concat.parquet"
    concat = pandas.concat(dataframes).reset_index(drop=True)
    concat.to_parquet(output_path)
    add_table_to_artifact(concat, key="table")
    return output_path


@prefect.task(cache_policy=INPUTS + TASK_SOURCE + MTIME + BYTES)
def clean(input_path: Path, method: CleanDataMethod) -> Path:
    concat = pandas.read_parquet(input_path)
    if method == CleanDataMethod.drop:
        clean = drop_columns_with_missing_values(concat, threshold=0.2)
    elif method == CleanDataMethod.impute:
        clean = impute_columns_with_mean(concat)
    else:
        raise NotImplementedError(f"Unknown method: '{self.method}'")
    output_path = PROCESSED_DIR / f"clean_{method.name}.parquet"
    clean.to_parquet(output_path)
    add_table_to_artifact(clean, key=f"table-{method.name}")
    return output_path


@prefect.task(cache_policy=INPUTS + TASK_SOURCE + MTIME + BYTES)
def linear_regression(input_path: Path, method: CleanDataMethod) -> Path:
    data = pandas.read_parquet(input_path)
    model, r2_adj = fit_linear_model(
        X=data.drop(columns=data.columns[0]),
        y=data.drop(columns=data.columns[1:]),
    )
    output_path = MODELS_DIR / f"linear_model_{method.name}.pkl"
    output_path.write_bytes(pickle.dumps(model))
    prefect.artifacts.create_markdown_artifact(
        f"Adjusted R2: {r2_adj:.3f}",
        key=f"metric-{method.name}",
    )
    return output_path


@prefect.flow()
def workflow(
    excel_paths: list[Path],
    methods: list[CleanDataMethod],
) -> dict[str, list[Path]]:
    parquet_paths = convert.map(input_path=excel_paths)
    concat_path = concat.submit(input_paths=parquet_paths)  # type: ignore
    clean_paths = clean.map(
        input_path=prefect.unmapped(concat_path),
        method=methods,
    )
    linear_regression_paths = linear_regression.map(
        input_path=clean_paths,
        method=methods,
    )
    return {
        convert.name: parquet_paths.result(),
        concat.name: [concat_path.result()],
        clean.name: clean_paths.result(),
        linear_regression.name: linear_regression_paths.result(),
    }


if __name__ == "__main__":
    excel_paths = list(RAW_DIR.glob("*.xlsx"))
    methods = [CleanDataMethod.drop, CleanDataMethod.impute]
    result = workflow(excel_paths=excel_paths, methods=methods)
    pprint.pprint(result, sort_dicts=False, width=120)

    # VisualizationUnsupportedError: `task.map()` is not currently supported by `flow.visualize()`.
    # workflow.visualize(excel_paths=excel_paths, methods=methods)
