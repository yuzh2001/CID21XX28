# STAB

This repository is the official implementation of STAB(the paper is under reviewing).

## Project Structure

```
./
├── MAPDN/
├── Multi-Walker/
│   ├── HARL_based/
│   ├── PPO_based/
├── README.md
```
Since our work covers two different environments and seven different algorithms, we split the code into three parts.

We use `uv` and `uv venv` to manage the dependencies and python versions.

For each directory, we provide a `README.md` file to introduce the code structure and how to run the code.

Generally speaking, each repository is a seperated project. You should:
1. navigate inside the directory
2. run `uv venv` to create a new python environment(this command will also use the corresponding python version)
3. run `uv sync` to install the dependencies, plus `uv pip install -e .` to install the algorithm package. 
4. If you need to change the run, goes to `src/configs/run/` and copy the `default.yaml` to create a new one.

```bash
# similar for other two repositories. details see the README.md in each directory.
cd MAPDN
uv venv
uv sync
uv pip install -e .
uv run src/run.py --config-name multiply
```

For detailed instructions, please refer to the `README.md` file in each directory.

## Requirements

- The python versions are not the same for different environments. 
- The requirements are not the same for different environments. 

Must use `uv` to install the dependencies.

## Training & Evaluation

We use an unified interface to train or evaluate the models.

For example, the config file `Multi-Walker/HARL_based/src/runs/train_all.yaml`:

```yaml
defaults:
  - default
  - _self_

run_group: latest
save_group: latest

description: full train and eval process, including all algorithms and environments

steps:
  - type: train
    config_name: config
    multirun: True
    args:
      - algorithm=mappo,happo,hasac
      - environment=raw,angle,disturb_friction_no_obs,disturb_friction_obs,disturb_motor_no_obs,disturb_motor_obs,disturb_package_mass_no_obs,disturb_package_mass_obs
  - type: eval
    config_name: config
    multirun: True
    args:
      - setting=0331-all
  - type: bark
    config_name: default
    multirun: False
    args:
      - finish

```
We use this format to run the training and evaluation in one file.
The config file above should be quite self-explanatory.

We developed such interface with hydra.

For more details, please refer to the [hydra](https://hydra.cc/docs/intro/) official website.

After finishing the config file, you can run the training and evaluation by:

```bash
uv run src/run.py --config-name train_all
```

To make this process faster, we recommend you set an alias for this command.
```bash
alias ur="uv run src/run.py --config-name"

# or, add it to your ~/.bashrc or ~/.zshrc file.
echo "alias ur='uv run src/run.py --config-name'" >> ~/.bashrc
```

Then you can run the training and evaluation by:
```bash
ur train_all
```


