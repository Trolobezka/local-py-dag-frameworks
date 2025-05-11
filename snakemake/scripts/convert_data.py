import pandas
from _helpers import extract

inputs, outputs, _, _ = extract(snakemake)  # type: ignore

dataframe = pandas.read_excel(inputs[0])
dataframe.to_parquet(outputs[0])
