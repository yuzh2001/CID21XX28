This project contains three repositories.

We use `uv` to manage the dependencies.

## MAPDN

navigate to ./MAPDN, and install the dependencies by:

```bash
uv sync
uv pip install -e . 
apt install freeglut3-dev python3-opengl -y

uv run src/run.py --config-name 0428_coma_ippo
```

If you need to change the run, goes to `src/configs/run/` and copy the `default.yaml` to create a new one.

## Multi-Walker/HARL_based

navigate to ./Multi-Walker/HARL_based, and install the dependencies by:

```bash
uv sync
uv pip install -e . 
```

If you need to change the run, goes to `src/configs/run/` and copy the `default.yaml` to create a new one.
