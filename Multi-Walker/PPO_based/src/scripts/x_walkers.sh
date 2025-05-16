# uv run src/eval.py --config-name=0305-weight
# uv run src/eval.py --config-name=0225-motor
# uv run src/eval.py --config-name=0223_f_obs_final



# uv run src/hp.py -m hp=baseline,angle_motor_obs hp.agent_num=2,8
uv run src/hp.py -m hp=angle_motor_obs hp.agent_num=2,8
# uv run src/eval.py --config-name=250513_n_walkers