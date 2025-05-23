# Setup

```
py -3.11 -m venv .venv
./.venv/Scripts/activate
python -m pip install --upgrade pip
pip install -r ./requirements.txt
```

# Run

```
prefect server start
```

Open [http://localhost:4200](http://localhost:4200).

```
python flow.py
```

## Cache

To reset the cache, delete the folder [.prefect/storage](.prefect/storage).

# Visualization

> [Graphviz](https://graphviz.org)'s `dot` command must be available or online/VSCode viewer used.
>
> ```
> flow.visualize()
> ```
>
> VisualizationUnsupportedError: `task.map()` is not currently supported by `flow.visualize()`.

The DAG visualization is also inside the web interface. After executing a flow, go to **Flows** (left menu) > click on the **Last run** of your flow (center) > here is the visualization of the executed tasks.
