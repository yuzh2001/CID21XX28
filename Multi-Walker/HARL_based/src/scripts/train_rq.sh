run_group=$(date +%m%d/%H%M)
save_group=0425/2055
# uv run src/train.py --multirun algorithm=mappo,happo environment=raw,angle,disturb_motor_no_obs,disturb_motor_obs environment.n_walkers=2 run_group=$run_group save_group=$save_group
uv run src/eval/eval.py setting=0425-agent_num setting.env_tweak.n_walkers=2 setting.run_group=$run_group setting.save_group=$save_group