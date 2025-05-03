.venv/bin/python polyfish/trainer.py \
 --games 5 --simulations 100 --prefix "" --epochs 10 \
 --cpuct 1.0 --gamma 0.99 --rollouts 50 --temperature 0.7  \
 --mapsize 9 --tribes "Imperius,Imperius" \
&& clear && tail -f training.log
