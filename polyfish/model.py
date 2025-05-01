import numpy as np
from os import path
from net import PolytopiaZeroNet
from requests import post
from random import shuffle
import torch, logging
import torch.nn.functional as F

type Dataset = list[tuple[dict,list[float]]]
                         
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(message)s', 
    datefmt='%H:%M:%S', 
    filename='training.log'
)
logger = logging.getLogger()
config: dict = {}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load(filename: str, config: dict, n_res_blocks: int = 10) -> PolytopiaZeroNet:
    if not filename.endswith('.zip'):
        filename += '.zip'
    net = PolytopiaZeroNet(n_res_blocks, config).to(device)
    if path.exists(filename):
        net.load_state_dict(torch.load(filename, map_location=device))
        logger.info(f"loaded {filename}")
    net.eval()
    return net

def train_network(net: PolytopiaZeroNet, dataset: Dataset, batch_size: int, epochs: int):
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    net.train()
    policy_losses, value_losses = [], []
    for _ in range(epochs):
        shuffle(dataset)
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i : i + batch_size]
            # Build batched observation dict:
            batched_obs = {}
            for key in batch[0][0]:
                batched_obs[key] = torch.tensor(np.array([x[0][key] for x in batch])).to(device).float()
            target_policy = torch.tensor(np.array([x[1] for x in batch])).to(device).float()  # shape: [B, action_dim]
            target_value = torch.tensor(np.array([x[2] for x in batch])).to(device).float()   # shape: [B]
            optimizer.zero_grad()
            policy_logits, value = net(batched_obs)
            # Compute policy loss (cross-entropy between target distribution and network distribution)
            log_probs = F.log_softmax(policy_logits, dim=1)
            policy_loss = -torch.mean(torch.sum(target_policy * log_probs, dim=1))
            # Compute value loss (MSE between network's value and the outcome)
            value_loss = F.mse_loss(value.squeeze(), target_value)
            # dirty fix? hmmm what causes this shape mismatch? [] != [1]
            # value_loss = F.mse_loss(value.view(1), target_value.view(1))
            loss = policy_loss + value_loss
            loss.backward()
            optimizer.step()
            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
    avg_policy_loss = np.mean(policy_losses)
    avg_value_loss = np.mean(value_losses)
    return avg_policy_loss, avg_value_loss

def self_train(net: PolytopiaZeroNet, iterations, n_games: int, epochs: int, n_sims: int, temperature: float, cPuct: float, gamma: float, deterministic: bool, batch_size: int, filename: str | None = None):
    logger.info("training started")
    logger.info(f"iterations={iterations}, n_games={n_games}, epochs={epochs}, n_sims={n_sims}, temperature={temperature}, cPuct={cPuct}, gamma={gamma}, deterministic={deterministic}, batch_size={batch_size}")
    
    for iteration in range(iterations):
        try:
            dataset = request_self_play(n_games, n_sims, temperature, cPuct, gamma, deterministic)
            rewards_data = np.array([x[2] for x in dataset])
            logger.info(f"collected {len(dataset)} plays")
            policy, value = train_network(net, dataset, batch_size, epochs)
            logger.info(f"loss={policy:.4f}, value={value:.4f} min={min(rewards_data):.4f}, max={np.max(rewards_data):.4f}, mean={np.mean(rewards_data):.4f}")
            if filename:
                if iteration % 2 == 0:
                    torch.save(net.state_dict(), f"{filename}-{iteration}.zip")
                else:
                    torch.save(net.state_dict(), f"{filename}-latest.zip")
        except Exception as e:
            logger.error(e)

def request_train(iterations: int, epochs: int, n_games: int, n_sims: int, temperature: float, cPuct: float, gamma: float, deterministic: bool, batch_size: int, prefix: str):
    return post("http://localhost:3000/train", json={
        "iterations": iterations,
        "n_games": n_games,
        "epochs": epochs,
        "n_sims": n_sims, 
        "temperature": temperature, 
        "cPuct": cPuct,
        "gamma": gamma,
        "deterministic": deterministic,
        "batch_size": batch_size,
        "prefix": prefix,
    }).json()

def request_self_play(n_games: int, n_sims: int, temperature: float, cPuct: float, gamma: float, deterministic: bool, game_settings={}) -> Dataset:
    return post("http://localhost:3000/selfplay", json={
        "n_games": n_games,
        "n_sims": n_sims, 
        "temperature": temperature, 
        "cPuct": cPuct,
        "gamma": gamma,
        "deterministic": deterministic,
        "settings": game_settings, 
    }).json()

