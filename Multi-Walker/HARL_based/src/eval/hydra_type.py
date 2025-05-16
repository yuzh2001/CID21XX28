from dataclasses import dataclass
from typing import List, Any, Optional


@dataclass
class RunConfig:
    result_file_name: Optional[str]
    run_name: str
    seed: int = 0
    eval_episodes: int = 100
    eval_threads: int = 50
    use_custom_disturbance: bool = False
    load_results: bool = False
    ablation: bool = True


@dataclass
class CheckpointConfig:
    algo: str
    timestamp: str
    desc: str = "."


@dataclass
class RenderConfig:
    use_gif: bool = False
    show_reward_curve: bool = False


@dataclass
class DisturbanceConfig:
    name: str
    start_at: int
    end_at: int
    disturbance_args: Any


@dataclass
class ScenarioConfig:
    name: str
    desc: Optional[str]
    disturbances: Optional[List[DisturbanceConfig]]
    is_raw: bool = False


@dataclass
class EnvTweakConfig:
    n_walkers: Optional[int] = 3


@dataclass
class EvalConfig:
    """
    Configuration class for evaluation settings.
    Attributes:
        eval (RunConfig): Detailed configuration for evaluation.
        default_scenarios (List[str]): List of default scenario names to be used in evaluation.
        checkpoints (List[Checkpoint]): List of checkpoints to load for evaluation.
        render (RenderConfig): Configuration for rendering during evaluation.
        disturbances (DisturbancesConfig): Configuration for disturbances applied during evaluation.
    """

    run: RunConfig
    checkpoints: List[CheckpointConfig]
    render: RenderConfig
    scenarios: List[ScenarioConfig]
    env_tweak: EnvTweakConfig
    run_group: str
    save_group: str


@dataclass
class SettingConfig:
    setting: EvalConfig
