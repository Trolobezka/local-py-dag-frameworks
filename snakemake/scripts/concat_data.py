import pandas
from _helpers import extract

inputs, outputs, _, _ = extract(snakemake)  # type: ignore

dataframes = [pandas.read_parquet(file) for file in inputs]
concat = pandas.concat(dataframes).reset_index(drop=True)
concat.to_parquet(outputs[0])
