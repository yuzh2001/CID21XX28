"""Train an algorithm."""

import rich.pretty
import wandb
import hydra
from omegaconf import DictConfig
import omegaconf
import rich
from harl.runners import RUNNER_REGISTRY


def _to_dict(cfg1: DictConfig):
    return omegaconf.OmegaConf.to_container(cfg1, resolve=True, throw_on_missing=True)


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    rich.pretty.pprint(cfg, expand_all=True)

    # 1. read the parameters from the config and convert to the format used by harl
    algo_args = cfg.algorithm
    env_args = cfg.environment

    algorithm_name = cfg.algorithm.name
    env_name = cfg.environment.name
    scenario_name = cfg.environment.scenario

    # 2. generate the name using the current time
    from datetime import datetime

    ts = datetime.now().strftime("%m%d-%H%M")
    run_name = f"[{algorithm_name}]<{scenario_name}><{cfg.environment.n_walkers}>"

    run_group = cfg.run_group
    save_group = cfg.save_group
    now_time = datetime.now().strftime("%m%d/%H%M")
    if run_group == "latest":
        run_group = now_time
    if save_group == "latest":
        save_group = now_time

    algo_args.logger.log_dir = f"./results/{save_group}"
    basic_info = {
        "env": env_name,
        "algo": algorithm_name,
        "exp_name": run_name,
    }

    if (
        env_name == "pettingzoo_mw"
        and algo_args.train.get("episode_length") is not None
    ):
        algo_args.train.episode_length = 500

    # 3. initialize wandb
    wandb.init(
        project="HARL",
        config=_to_dict(cfg),
        sync_tensorboard=True,
        name=run_name + f"_{ts}",
        group=run_group,
        job_type="train",
        tags=[
            env_name,
            algorithm_name,
            scenario_name,
            f"wker-{cfg.environment.n_walkers}",
        ],
    )

    # 4. format the parameters and convert to dict to pass to harl
    algo_dict = _to_dict(algo_args)
    del algo_dict["name"]

    env_dict = _to_dict(env_args)
    del env_dict["name"]
    del env_dict["scenario"]

    runner = RUNNER_REGISTRY[algorithm_name](basic_info, algo_dict, env_dict)

    # 5. start training
    runner.run()

    # 6. training completed
    runner.close()

    wandb.finish()


if __name__ == "__main__":
    main()
