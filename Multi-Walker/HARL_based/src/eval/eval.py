from __future__ import annotations

import json
import os
import time

import hydra
import rich
from omegaconf import DictConfig
from rich.panel import Panel
import omegaconf
from copy import deepcopy
import wandb
# from .disturbances import DisturbanceFactory, MultiWalkerEnv
# from .utils.gif import export_gif
# from walker import multiwalker_v9

from harl.runners import RUNNER_REGISTRY
from harl.envs.pettingzoo_mw.pettingzoo_mw_logger import PettingZooMWLogger
import hydra_type
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
from moviepy.editor import VideoFileClip
import imageio
import matplotlib.pyplot as plt
import numpy as np

os.environ["SDL_VIDEODRIVER"] = "dummy"

wandb_results = []
max_cycles = 500


def export_gif(config_name, frames_arr, rewards_arr):
    rich.print(f"Exporting gif for {config_name}")
    gif_dir = os.path.join(
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir, "./videos/"
    )
    gif_folder = os.path.join(gif_dir, f"{config_name}")
    os.makedirs(gif_folder, exist_ok=True)

    # rich.print(frames_arr)
    for i, frames in enumerate(frames_arr):
        # 1. gif generation
        rewards = rewards_arr[i]
        is_negative = rewards < 0
        gif_path = os.path.join(
            gif_folder,
            f"{'fail' if is_negative else 'success'}_{config_name}_{i}_{rewards:.2f}.gif",
        )
        imageio.mimwrite(
            gif_path,
            frames,
            duration=10,
        )

        # 3. video generation
        clip = VideoFileClip(gif_path)
        clip.write_videofile(
            os.path.join(
                gif_folder,
                f"{'fail' if is_negative else 'success'}_{config_name}_{i}_{rewards:.2f}.mp4",
            ),
            codec="libx264",
            logger=None,
        )


def _to_dict(cfg1: DictConfig):
    return omegaconf.OmegaConf.to_container(cfg1, resolve=True, throw_on_missing=True)


def log_wandb():
    print("logging to wandb")
    rich.print(wandb_results)
    angle_intervals = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 5),
        (5, 8),
        (8, 10),
        (10, 15),
        (15, float("inf")),
    ]
    columns = [
        "algo",
        "variant",
        "scenario",
        "n_walkers",
        "terminate_cnt",
        "package_final_x",
        "angle_avg",
    ]
    columns.extend([f"angle-{start}-{end}" for start, end in angle_intervals])
    table_data = []
    for result in wandb_results:
        table_data.append(
            [
                result["algo"],
                result["variant"],
                result["scenario"],
                result["n_walkers"],
                result["terminate_cnt"],
                result["package_x"],
                result["angle_avg"],
                *[
                    result["angle_data"].get(f"angle-{start}-{end}", 0)
                    for start, end in angle_intervals
                ],
            ]
        )

    rich.print(table_data)
    test_table = wandb.Table(data=table_data, columns=columns)
    wandb.log({"test_table": test_table})


def run_evaluations(
    config: hydra_type.EvalConfig, checkpoint: hydra_type.CheckpointConfig
) -> tuple[dict, dict]:
    """执行baseline和扰动测试的评估"""
    gif_dir = os.path.join(
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir, "./videos"
    )
    os.makedirs(gif_dir, exist_ok=True)
    results = []

    # all scenarios
    for scenario in config.scenarios:
        # This can be confusing; remember that the four evals below correspond to four variants of the same algorithm.
        # Therefore, for a disturbed environment, four checkpoints will be tested: trained in original env, trained in angle env, without obs, with obs
        # When the environment is defined as raw, only the first two variants are tested
        # raw, angle
        # + !scenario.is_raw -> obs, no_obs
        raw_results = eval(
            config,
            checkpoint=checkpoint,
            checkpoint_type="raw",
            eval_scenario=scenario,
        )
        results.append(raw_results)

        # angle_results = eval(
        #     config,
        #     checkpoint=checkpoint,
        #     checkpoint_type="angle",
        #     eval_scenario=scenario,
        # )
        # results.append(angle_results)

        if not scenario.is_raw:
            obs_results = eval(
                config,
                checkpoint=checkpoint,
                checkpoint_type=f"disturb_{scenario.name}_obs",
                eval_scenario=scenario,
            )
            results.append(obs_results)
            if config.run.ablation:
                no_obs_results = eval(
                    config,
                    checkpoint=checkpoint,
                    checkpoint_type=f"disturb_{scenario.name}_no_obs",
                    eval_scenario=scenario,
                )
                results.append(no_obs_results)

                _sc = deepcopy(scenario)
                _sc.is_raw = True
                _sc.name = "raw"
                _sc.disturbances = None
                obs_raw_results = eval(
                    config,
                    checkpoint=checkpoint,
                    checkpoint_type=f"disturb_{scenario.name}_obs",
                    eval_scenario=_sc,
                )
                results.append(obs_raw_results)
                no_obs_raw_results = eval(
                    config,
                    checkpoint=checkpoint,
                    checkpoint_type=f"disturb_{scenario.name}_no_obs",
                    eval_scenario=_sc,
                )
                results.append(no_obs_raw_results)

    return results


def eval(
    globalConfig: hydra_type.EvalConfig,
    checkpoint: hydra_type.CheckpointConfig,
    checkpoint_type: str,
    eval_scenario: hydra_type.ScenarioConfig,
):
    start_time = time.time()
    base_checkpoint_path = f"./results/{checkpoint.timestamp}/pettingzoo_mw/multiwalker/{checkpoint.algo}/[{checkpoint.algo}]<{checkpoint_type}>"
    if hasattr(globalConfig, "save_group"):
        base_checkpoint_path = f"./results/{globalConfig.save_group}/pettingzoo_mw/multiwalker/{checkpoint.algo}/[{checkpoint.algo}]<{checkpoint_type}>"
    # if globalConfig.env_tweak.n_walkers != 3:
    if globalConfig.save_group.startswith("03"):
        pass
    else:
        base_checkpoint_path += f"<{globalConfig.env_tweak.n_walkers}>"
    rich.print(os.listdir(base_checkpoint_path))
    seed_folder = next(
        folder
        for folder in os.listdir(base_checkpoint_path)
        if folder.startswith("seed-")
    )
    checkpoint_path = os.path.join(base_checkpoint_path, seed_folder, "models")

    # 1. read the corresponding model
    rich.print(
        Panel(
            f"Checkpoint Path: {checkpoint_path}\nScenario Name: {eval_scenario.name}",
            title="Evaluation Info",
        )
    )

    # 1.1. read the parameters from the config and convert to the format used by harl
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(
            config_name="config",
            overrides=[
                f"algorithm={checkpoint.algo}",
                f"environment={checkpoint_type}",
            ],
        )
        algo_args = cfg.algorithm
        env_args = cfg.environment

        algorithm_name = cfg.algorithm.name
        env_name = cfg.environment.name
        scenario_name = cfg.environment.scenario
        basic_info = {
            "env": env_name,
            "algo": algorithm_name,
            "exp_name": f"testing_<{algorithm_name}>_{scenario_name}",
        }
        if (
            env_name == "pettingzoo_mw"
            and algo_args.train.get("episode_length") is not None
        ):
            algo_args.train.episode_length = 500

        algo_args.train.model_dir = checkpoint_path  # read the model!

        # configure the number of eval episodes
        algo_args.eval.n_eval_rollout_threads = globalConfig.run.eval_threads
        algo_args.eval.eval_episodes = globalConfig.run.eval_episodes

        if globalConfig.render.use_gif:
            algo_args.render.use_render = True
            algo_args.render.render_episodes = globalConfig.run.eval_episodes

        if (
            env_name == "pettingzoo_mw"
            and algo_args.train.get("num_env_steps") is not None
        ):
            algo_args.train.num_env_steps = 1

        env_args.max_cycles = max_cycles

        algo_dict = _to_dict(algo_args)
        del algo_dict["name"]

        env_dict = _to_dict(env_args)
        del env_dict["name"]
        del env_dict["scenario"]
        env_dict["custom"]["eval_disturb"] = _to_dict(eval_scenario)["disturbances"]
        env_dict["custom"]["is_eval"] = True

        env_tweak = globalConfig.env_tweak
        if env_tweak:
            for key, value in _to_dict(env_tweak).items():
                if value is not None:
                    env_dict[key] = value

        runner = RUNNER_REGISTRY[algorithm_name](basic_info, algo_dict, env_dict)

        # runner.run()

        if globalConfig.render.use_gif:
            render_mode = "rgb_array"
            rgb_array, rewards_arr = runner.render(
                render_mode
            )  # [[..rgbarray...], ...]
            # use some method to get the reward, and then record it in config_name
            export_gif(
                config_name=f"[{checkpoint.algo}]<{checkpoint_type}>_{eval_scenario.name}",
                frames_arr=rgb_array,
                rewards_arr=rewards_arr,
            )
        else:
            # choose different eval methods based on whether it is off-policy
            has_logger = hasattr(runner, "logger")
            if has_logger:
                logger: PettingZooMWLogger = runner.logger
                logger.is_testing = True  # identify that it is currently in eval; but eval is used by it, so it can only be called test.
                runner.eval()
            else:
                logger = None
                runner.eval(1)

            if has_logger:
                terminate_arr = logger.test_data["terminate_at"]
                angle_arr = logger.test_data["angle_data"]
            else:
                terminate_arr = runner.eval_episode_lens
                angle_arr = runner.eval_episode_angles
            # start calculating
            # 2.1 calculate the number of times the walker falls before the end
            terminate_cnt = 0
            package_x = []
            for i in range(len(terminate_arr)):
                if (
                    terminate_arr[i] + 2 < max_cycles
                ):  # +2 remove a little margin problem
                    terminate_cnt += 1
                package_x.append(
                    logger.test_data["package_x"][i]
                    if has_logger
                    else runner.episode_xs[i]
                )

            if hasattr(runner, "eval_envs") and runner.eval_envs is not None:
                runner.eval_envs.close()
            runner.close()

            end_time = time.time()
            print(
                f"Evaluate [{checkpoint.algo}]<{checkpoint_type}>_{eval_scenario.name} cost: {end_time - start_time:.2f} seconds"
            )
            return_result = {
                "desc": f"[{checkpoint.algo}]<{checkpoint_type}>_{eval_scenario.name}_{_to_dict(eval_scenario).get('desc', 'original')}",
                "algo": checkpoint.algo,
                "variant": checkpoint_type,
                "scenario": eval_scenario.name,
                "terminate_cnt": terminate_cnt,
                "angle_data": [
                    angle for episode_angles in angle_arr for angle in episode_angles
                ],
                "angle_data_grouped": angle_arr,
                "package_x": sum(package_x) / len(package_x),
            }
            rich.print(return_result["package_x"])
            return return_result


def save_eval_results(results: dict, output_dir: str, name: str) -> str:
    """save the evaluation results to a JSON file"""
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{name}.json"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w") as f:
        json.dump(results, f)
    return filepath


def load_eval_results(filepath: str) -> dict:
    """load the evaluation results from a JSON file, and save a copy to the hydra output directory"""
    with open(filepath, "r") as f:
        data = json.load(f)
    return data


def analyze_eval_results(
    results,
    config: hydra_type.EvalConfig,
    checkpoint: hydra_type.CheckpointConfig,
):
    for res in results:
        angle_intervals = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 5),
            (5, 8),
            (8, 10),
            (10, 15),
            (15, float("inf")),
        ]
        baseline_interval_counts = {
            f"angle-{start}-{end}": sum(
                1 for a in res["angle_data"] if start <= abs(a) < end
            )
            for start, end in angle_intervals
        }
        angle_avg = sum(abs(a) for a in res["angle_data"]) / len(res["angle_data"])

        # plot the line chart (each group has a separate chart)
        angle_groups = res["angle_data_grouped"]
        output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        save_dir = os.path.join(output_dir, "angle_figs")
        os.makedirs(save_dir, exist_ok=True)
        print(save_dir)
        for idx, group in enumerate(angle_groups):
            x = np.arange(len(group))
            y = np.abs(group)
            plt.figure(figsize=(10, 4))
            plt.plot(x, y, label="|angle|")
            plt.xlabel("Step")
            plt.ylabel("Angle (abs)")
            plt.grid(True)
            plt.legend()
            plt.ylim(0, 10)
            plt.xlim(0, 500)
            fname = f"{res['scenario']}_cp_{res['variant']}_{idx}.png"
            fname = fname.replace("/", "_")  # prevent path problem
            plt.savefig(os.path.join(save_dir, fname))
            plt.close()

        wandb_item = {
            "algo": res["algo"],
            "variant": res["variant"],
            "scenario": res["scenario"],
            "n_walkers": config.env_tweak.n_walkers,
            "terminate_cnt": res["terminate_cnt"],
            "package_x": res["package_x"],
            "angle_data": baseline_interval_counts,
            "angle_avg": angle_avg,
        }

        wandb_results.append(wandb_item)

    rich.print(wandb_results)


@hydra.main(
    config_path="./eval_configs",
    config_name="config",
    version_base=None,
)
def main(cfg: hydra_type.SettingConfig):
    # directory for json storage
    json_dir = "./eval_results"
    os.makedirs(json_dir, exist_ok=True)
    timestamp = time.strftime("%m%d-%H:%M")
    GlobalHydra.instance().clear()
    # initialize wandb
    run = wandb.init(
        project="harl_new",
        name=cfg.setting.run_group + "_" + timestamp,
        config=_to_dict(cfg),
        save_code=True,
        group=cfg.setting.run_group,
        job_type="eval",
    )

    def process_checkpoint(checkpoint: hydra_type.CheckpointConfig):
        print(f"Processing checkpoint: {checkpoint.algo} - {checkpoint.desc}")
        should_load_results = cfg.setting.run.load_results
        if should_load_results:
            # load existing results mode
            result_file_name = cfg.setting.run.result_file_name
            if result_file_name == "latest":
                # load the latest version from the latest subdirectory
                latest_dir = os.path.join(json_dir, "latest")
                results = load_eval_results(
                    os.path.join(latest_dir, f"{checkpoint.algo}.json")
                )
            else:
                # load the specified version from the timestamp subdirectory
                timestamp_dir = os.path.join(json_dir, result_file_name)
                results = load_eval_results(
                    os.path.join(timestamp_dir, f"{checkpoint.algo}.json")
                )
        else:
            # execute the evaluation mode
            results = run_evaluations(cfg.setting, checkpoint)

            # create the timestamp subdirectory
            timestamp_dir = os.path.join(json_dir, timestamp)
            os.makedirs(timestamp_dir, exist_ok=True)

            # create the latest subdirectory
            latest_dir = os.path.join(json_dir, "latest")
            os.makedirs(latest_dir, exist_ok=True)

            # save the results to the timestamp subdirectory
            save_eval_results(
                results,
                timestamp_dir,
                f"{checkpoint.algo}",
            )

            save_eval_results(results, latest_dir, f"{checkpoint.algo}")

        # analyze the results and get the charts
        wandb_results = []
        rich.print(wandb_results)
        analyze_eval_results(results, cfg.setting, checkpoint)
        log_wandb()

    for checkpoint in cfg.setting.checkpoints:
        start_time = time.time()
        process_checkpoint(checkpoint)
        end_time = time.time()
        print(f"{checkpoint.algo} 耗时: {end_time - start_time:.2f}秒")

    run.finish()
    # exit()


if __name__ == "__main__":
    main()
