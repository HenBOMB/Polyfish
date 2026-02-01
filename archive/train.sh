.venv/bin/python polyfish/trainer.py \
 --games 20 --simulations 200 --prefix "" --epochs 20 \
 --cpuct 1.5 --gamma 0.997 --rollouts 20 --temperature 0.9  \
 --mapsize 9 --tribes "Imperius,Imperius" --dirichlet false \
&& clear && tail -f training.log
