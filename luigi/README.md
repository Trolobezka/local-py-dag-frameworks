# Setup

```
py -3.11 -m venv .venv
./.venv/Scripts/activate
python -m pip install --upgrade pip
pip install -r ./requirements.txt
```

Change path to the `luigi_state.pickle` in the [luigi.cfg](luigi.cfg) configuration file to your own absolute path. The pickle is created on shutdown of the `luigid` process.

# Run using local scheduler

```
luigi --module tasks LinearRegression --local-scheduler --method drop
```

# Run using central scheduler

First terminal:
```
luigid
```

Second terminal:
```
luigi --module tasks LinearRegression --method drop
```
