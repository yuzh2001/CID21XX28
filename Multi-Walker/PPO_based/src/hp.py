from __future__ import annotations

import os
import time

import hydra
import supersuit as ss
from omegaconf import DictConfig
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.ppo import MlpPolicy
import rich
import wandb
from walker import multiwalker_v9
from wandb.integration.sb3 import WandbCallback

os.environ["SDL_VIDEODRIVER"] = "dummy"


@hydra.main(config_path="./configs/train", config_name="default", version_base=None)
def train_butterfly_supersuit(cfg: DictConfig):
    rich.print(cfg)
    run = wandb.init(
        project="sb3",
        name=cfg.hp.name + "_" + time.strftime("%Y%m%d-%H%M%S"),
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,
        config=dict(cfg),
    )
    wandb.save("src/*", base_path=".")
    wandb.save("src/walker/*", base_path=".")
    wandb.save("src/walker/multiwalker/*", base_path=".")

    use_angle_reward = cfg.hp.get("use_angle_reward", False)
    use_f_disturbance = cfg.hp.get("use_f_disturbance", False)
    use_f_obs = cfg.hp.get("use_f_obs", False)
    use_motor_obs = cfg.hp.get("use_motor_obs", False)
    use_motor_disturbance = cfg.hp.get("use_motor_disturbance", False)
    use_package_mass_obs = cfg.hp.get("use_package_mass_obs", False)
    use_package_mass_disturbance = cfg.hp.get("use_package_mass_disturbance", False)
    n_walkers = cfg.hp.get("agent_num", 3)
    max_cycles = cfg.hp.get("max_cycles", 500)

    env = multiwalker_v9.parallel_env(
        # 0 angle reward
        use_angle_reward=use_angle_reward,
        # 1 f
        use_f_disturbance=use_f_disturbance,
        use_f_obs=use_f_obs,
        # 2 motor
        use_motor_obs=use_motor_obs,
        use_motor_disturbance=use_motor_disturbance,
        # 3 package mass
        use_package_mass_obs=use_package_mass_obs,
        use_package_mass_disturbance=use_package_mass_disturbance,
        # 4 n_walkers
        n_walkers=n_walkers,
        # 5 max_cycles
        max_cycles=max_cycles,
    )

    env.reset(seed=cfg.hp.seed)

    print(f"Starting training on {str(cfg.hp.name)}.")
    env = ss.frame_stack_v1(env, 3)
    env = ss.pettingzoo_env_to_vec_env_v1(env)

    parallel_envs = 10
    num_cpus = parallel_envs
    env = ss.concat_vec_envs_v1(
        env, parallel_envs, num_cpus, base_class="stable_baselines3"
    )

    # a = 180000
    # b=6000
    # c=10

    # c=5
    # b=12000

    # a=c*b*3

    desired_eval_steps = 400_000
    desired_save_steps = 20_000_000

    # checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=(desired_save_steps // (parallel_envs * 3)),
        save_path=f"./checkpoint_models/training/{cfg.hp.name}_{time.strftime('%Y%m%d-%H%M%S')}",
        name_prefix=f"{cfg.hp.name}_{time.strftime('%Y%m%d-%H%M%S')}",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    eval_callback = EvalCallback(
        env,
        eval_freq=(desired_eval_steps // (parallel_envs * 3)),
        n_eval_episodes=10,
        best_model_save_path=f"./checkpoint_models/training/{time.strftime('%Y%m%d-%H%M%S')}_{cfg.hp.name}_best",
    )
    callback = CallbackList([checkpoint_callback, eval_callback, WandbCallback()])

    model = PPO(
        MlpPolicy,
        env,
        verbose=1,
        n_steps=cfg.hp.n_steps,
        batch_size=cfg.hp.batch_size,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        learning_rate=cfg.hp.learning_rate,
        normalize_advantage=True,
        tensorboard_log=f"logs/agent_3_{time.strftime('%Y%m%d-%H%M%S')}",
        policy_kwargs={
            "net_arch": {
                "pi": cfg.hp.net_arch.pi,
                "vf": cfg.hp.net_arch.vf,
            },
        },
    )
    print(model.device)
    # model = PPO.load(
    #     os.path.join(
    #         "./checkpoint_models",
    #         # "20250215-024717_256net500bslr2e-4(256net500bslr2e-4_20250215-003756).zip",
    #         "20250213-210659_angle_f(angle_f_20250213-185935).zip",
    #     )
    # )
    model.set_env(env)
    model.learn(total_timesteps=cfg.hp.steps, callback=callback)
    model.save(
        f"./checkpoint_models/{time.strftime('%Y%m%d-%H%M%S')}_({run.name}){cfg.hp.name}<{cfg.hp.agent_num}>"
    )
    print(f"Finished training on {str(cfg.hp.name)}.")
    env.close()
    run.finish()


if __name__ == "__main__":
    train_butterfly_supersuit()
