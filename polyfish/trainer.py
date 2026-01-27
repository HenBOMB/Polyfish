import argparse
import model

parser = argparse.ArgumentParser()
parser.add_argument('--iterations', type=int, default=1000)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--games', type=int, default=3)
parser.add_argument('--simulations', type=int, default=1000)
parser.add_argument('--temperature', type=float, default=0.7)
parser.add_argument('--cpuct', type=float, default=1.0)
parser.add_argument('--gamma', type=float, default=0.997)
parser.add_argument('--deterministic', action='store_true', default=False)
parser.add_argument('--dirichlet', type=bool, default=True)
parser.add_argument('--rollouts', type=int, default=50)
parser.add_argument('--prefix', type=str, default="0.0.0")
parser.add_argument('--mapsize', type=int, default=11)
parser.add_argument('--tribes', type=str, default="Imperius, Imperius")

args = parser.parse_args()

settings = {
    'size': args.mapsize,
    'tribes': args.tribes
}

# Build a JSON payload and send it as the request body. The previous call passed
# positional arguments which caused a primitive (int) to be sent as JSON and the
# server returned HTTP 400. Send an object instead.
payload = {
    'iterations': args.iterations,
    'epochs': args.epochs,
    'n_games': args.games,
    'n_sims': args.simulations,
    'temperature': args.temperature,
    'cPuct': args.cpuct,
    'gamma': args.gamma,
    'deterministic': args.deterministic,
    'dirichlet': args.dirichlet,
    'rollouts': args.rollouts,
    'prefix': args.prefix,
    'settings': settings,
}

model.request_train(json=payload)
