# Setup

```
py -3.11 -m venv .venv
./.venv/Scripts/activate
python -m pip install --upgrade pip
pip install -r ./requirements.txt
```

# Run

```
snakemake --dry-run
snakemake --core 4
snakemake --delete-all-output
```

To create only specific file:
```
snakemake --core 4 data/interim/concat.parquet
snakemake --core 4 models/linear_regression_drop.pkl
```

# Visualization

[Graphviz](https://graphviz.org)'s `dot` command must be available or online/VSCode viewer used.

Must be run in Command Prompt. PowerShell may produce invalid `.dot` file.

```
snakemake --forceall --dag > dag.dot
dot -Tsvg dag.dot > dag.svg
dot -Tpdf dag.dot > dag.pdf
```
