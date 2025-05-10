# Setup

```
py -3.11 -m venv .venv
./.venv/Scripts/activate
python -m pip install --upgrade pip
pip install -r ./requirements.txt
```

Change path to the [.dagster](.dagster) folder in the `.env` file to your own absolute path.

# Run

```
dagster dev -f .\assets.py
```

Open [http://localhost:3000](http://localhost:3000).

Go to **Assets** (top left) > click **View lineage** (top right) > click **Materialize all...** (top right) > select **All** partitions  > click **Launch backfill** > wait for asset materialization.
