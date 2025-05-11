import pandas
from _helpers import drop_columns_with_missing_values, extract, impute_columns_with_mean

inputs, outputs, params, wildcards = extract(snakemake)  # type: ignore
threshold = float(params["threshold"])
method = wildcards["method"]

concat = pandas.read_parquet(inputs[0])
if method == "drop":
    clean = drop_columns_with_missing_values(concat, threshold)
elif method == "impute":
    clean = impute_columns_with_mean(concat)
else:
    raise NotImplementedError(f"Unknown method: '{method}'")
clean.to_parquet(outputs[0])
