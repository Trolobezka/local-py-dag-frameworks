import pickle

import pandas
from _helpers import extract, fit_linear_model

inputs, outputs, _, wildcards = extract(snakemake)  # type: ignore
method = wildcards["method"]

data = pandas.read_parquet(inputs[0])
model, r2_adj = fit_linear_model(
    X=data.drop(columns=data.columns[0]),
    y=data.drop(columns=data.columns[1:]),
)
outputs[0].write_bytes(pickle.dumps(model))
