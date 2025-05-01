import argparse
import model

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--iterations', type=int, default=1000)
parser.add_argument('-e', '--epochs', type=int, default=100)
parser.add_argument('-r', '--rounds', type=int, default=3)
parser.add_argument('-s', '--simulations', type=int, default=1000)
parser.add_argument('-t', '--temperature', type=float, default=0.7)
parser.add_argument('-c', '--cpuct', type=float, default=1.0)
parser.add_argument('-g', '--gamma', type=float, default=0.997)
parser.add_argument('-d', '--deterministic', action='store_true', default=False)
parser.add_argument('-b', '--batch', type=int, default=16)
parser.add_argument('-p', '--prefix', type=str, default="0.0.0")

args = parser.parse_args()

model.request_train(
    args.iterations,
    args.epochs,
    args.rounds,
    args.simulations,
    args.temperature,
    args.cpuct,
    args.gamma,
    args.deterministic,
    args.batch,
    args.prefix,
)
