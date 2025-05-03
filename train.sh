.venv/bin/python polyfish/trainer.py \
 --games 10 --simulations 500 --prefix "" --epochs 10 \
 --cpuct 1.0 --gamma 0.99 --rollouts 50 --temperature 0.7  \
&& clear && tail -f training.log
