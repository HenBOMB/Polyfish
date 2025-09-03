import numpy as np
from os import path
from net import PolytopiaNet # Assuming your PolytopiaNet class is in net.py
from requests import post
from random import shuffle
import torch, logging
import torch.nn as nn # Added for type hinting and nn.functional
import torch.nn.functional as F

# Updated Dataset Type Hint
ObsDict = dict # e.g., {'map': np.ndarray, 'player': np.ndarray}
TargetPoliciesDict = dict # e.g., {'pi_action': np.ndarray, 'pi_option_struct': np.ndarray, ...}
TargetValuesDict = dict   # e.g., {'v_win': float, 'v_econ': float, ...}
MoveTypeStr = str
Dataset = list[tuple[ObsDict, TargetPoliciesDict, TargetValuesDict, MoveTypeStr]]

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S',
    filename='training.log',
    filemode='a' # Append to log file
)
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S'))
logger = logging.getLogger()
if not logger.handlers:
    logger.addHandler(console_handler)
logger.setLevel(logging.INFO)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load(filename: str, config: dict, n_res_blocks: int = 12) -> PolytopiaNet:
    if not filename.endswith('.zip'):
        filename += '.zip'
    net = PolytopiaNet(
        dim_map_channels=config['dim_map_channels'],
        dim_map_size=config['dim_map_size'],
        dim_player=config['dim_player'],
        dim_struct=config['dim_struct'],
        dim_skill=config['dim_ability'],
        dim_unit=config['dim_unit'],
        num_action_types=config['dim_moves'],
        dim_tech=config['dim_tech'],
        num_res_blocks=n_res_blocks,
        num_hidden_channels=config.get('hidden_channels', 128),
        num_player_hidden=config.get('num_player_hidden', 32)
    ).to(device)
    if path.exists(filename):
        try:
            net.load_state_dict(torch.load(filename, map_location=device))
            logger.info(f"Successfully loaded model from {filename}")
        except Exception as e:
            logger.error(f"Error loading model from {filename}: {e}. Starting with a new model.")
    else:
        logger.warning(f"Model file {filename} not found. Starting with a new model.")
    net.eval()
    return net

def train_network(
    net: PolytopiaNet,
    dataset: Dataset,
    batch_size: int,
    epochs: int,
    learning_rate: float = 0.001,
    policy_loss_weights: dict = None,
    value_loss_weights: dict = None,
    gradient_clipping_norm: float = None
):
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    net.train()

    epoch_losses = { 'total': [] }
    try:
        dummy_obs_map = torch.zeros((1, net.initial_conv.in_channels, net.dim_map_size, net.dim_map_size), device=device)
        dummy_obs_player = torch.zeros((1, net.player_fc1.in_features), device=device)
        with torch.no_grad():
            dummy_pred = net({'map': dummy_obs_map, 'player': dummy_obs_player})
        for key in dummy_pred.keys():
            if key.startswith('pi_') or key.startswith('v_'):
                epoch_losses[key] = []
        logger.debug(f"Dynamically determined logging keys: {list(epoch_losses.keys())}")
    except AttributeError as e:
        logger.warning(f"Could not dynamically determine net output keys for logging due to AttributeError: {e}. Using predefined set.")
        predefined_keys = ['pi_action', 'pi_source', 'pi_target',
                           'pi_struct', 'pi_skill', 'pi_unit',
                           'pi_tech', 'pi_reward', 'v_win']
        for key in predefined_keys:
             epoch_losses[key] = []


    if policy_loss_weights is None: policy_loss_weights = {}
    if value_loss_weights is None: value_loss_weights = {}
    default_policy_weight = 1.0

    for epoch_idx in range(epochs):
        shuffle(dataset)
        logger.info(f"Epoch {epoch_idx+1}/{epochs}")
        batch_num = 0
        for i in range(0, len(dataset), batch_size):
            batch_num += 1
            current_batch = dataset[i : i + batch_size]
            if not current_batch: continue

            map_batch = [sample[0]['map'] for sample in current_batch]
            player_batch = [sample[0]['player'] for sample in current_batch]
            batched_obs = {
                'map': torch.tensor(np.array(map_batch), dtype=torch.float32).to(device),
                'player': torch.tensor(np.array(player_batch), dtype=torch.float32).to(device)
            }
            actual_batch_size = batched_obs['map'].size(0)

            target_policies_list = [sample[1] for sample in current_batch]
            target_values_list = [sample[2] for sample in current_batch]
            # move_types_list = [sample[3] for sample in current_batch]

            optimizer.zero_grad()
            predictions = net(batched_obs) # dict of tensors
            batch_total_loss = torch.tensor(0.0).to(device)
            current_batch_losses_log = {key: [] for key in epoch_losses if key != 'total'}

            # Policy Losses
            for sample_idx in range(actual_batch_size):
                sample_target_policies = target_policies_list[sample_idx]

                for net_output_key, batch_pred_logits in predictions.items(): # net_output_key has pi_
                    if not net_output_key.startswith('pi_'): continue
                    if net_output_key not in sample_target_policies:continue

                    sample_pred_logits = batch_pred_logits[sample_idx]
                    sample_target_p_numpy = sample_target_policies[net_output_key]
                    
                    if sample_target_p_numpy is None: continue

                    sample_target_p = torch.tensor(sample_target_p_numpy, dtype=torch.float32).to(device)
                    
                    # Use net_output_key (with _logits) for weight lookup
                    weight = policy_loss_weights.get(net_output_key, default_policy_weight)
                    loss_val_sample = torch.tensor(0.0).to(device)

                    if net_output_key == 'pi_reward':
                        # sample_pred_logits is scalar, sample_target_p should be scalar (0.0 or 1.0)
                        loss_fn = nn.BCEWithLogitsLoss()
                        loss_val_sample = loss_fn(sample_pred_logits.unsqueeze(0), sample_target_p.unsqueeze(0)) * weight
                    else: # Categorical policy heads
                        # Ensure logits and targets are 1D for per-sample calculation
                        if sample_pred_logits.dim() == 0: sample_pred_logits = sample_pred_logits.unsqueeze(0)
                        if sample_target_p.dim() == 0: sample_target_p = sample_target_p.unsqueeze(0)
                        
                        # Handle empty target (e.g., no valid options for this choice)
                        if sample_target_p.numel() == 0 or \
                           (sample_target_p.numel() > 0 and sample_target_p.sum().item() == 0.0):
                            loss_val_sample = torch.tensor(0.0).to(device) # No loss if target is empty or all zeros
                        elif sample_target_p.shape[0] != sample_pred_logits.shape[0]:
                            # logger.warning(f"Shape mismatch for {net_output_key} in sample {sample_idx}. Pred: {sample_pred_logits.shape}, Target: {sample_target_p.shape}. Skipping.")
                            continue
                        else:
                            log_probs_sample = F.log_softmax(sample_pred_logits, dim=-1)
                            loss_val_sample = -torch.sum(sample_target_p * log_probs_sample) * weight
                    
                    if not torch.isnan(loss_val_sample) and not torch.isinf(loss_val_sample):
                        batch_total_loss += loss_val_sample # Add this sample's head loss to batch total
                        if net_output_key in current_batch_losses_log: # Log with no_suffix key
                             current_batch_losses_log[net_output_key].append(loss_val_sample.item())

            # Value Losses
            for net_output_key, batch_pred_v in predictions.items(): # net_output_key is e.g. 'v_win'
                if not net_output_key.startswith('v_'):
                    continue

                valid_target_v_list, valid_pred_v_indices = [], []
                for sample_idx in range(actual_batch_size):
                    sample_target_val = target_values_list[sample_idx].get(net_output_key)
                    if sample_target_val is not None:
                        valid_target_v_list.append(sample_target_val)
                        valid_pred_v_indices.append(sample_idx)
                
                if not valid_target_v_list: continue

                target_v_tensor = torch.tensor(np.array(valid_target_v_list), dtype=torch.float32).to(device).unsqueeze(1)
                pred_v_tensor_for_loss = batch_pred_v[valid_pred_v_indices] # Select predictions for which targets exist

                weight = value_loss_weights.get(net_output_key, 1.0) # Use direct key for weight
                loss_val_batch_head = F.mse_loss(pred_v_tensor_for_loss, target_v_tensor) * weight
                
                if not torch.isnan(loss_val_batch_head) and not torch.isinf(loss_val_batch_head):
                    batch_total_loss += loss_val_batch_head # Add this head's total batch loss
                    if net_output_key in current_batch_losses_log: # Log with direct key
                         current_batch_losses_log[net_output_key].append(loss_val_batch_head.item())


            if actual_batch_size > 0:
                # Average the sum of all losses by the number of samples in the batch
                final_batch_loss = batch_total_loss / actual_batch_size
            else:
                final_batch_loss = torch.tensor(0.0).to(device)

            if final_batch_loss > 0 and final_batch_loss.requires_grad: # Ensure backward is called on a valid graph
                final_batch_loss.backward()
                if gradient_clipping_norm:
                    torch.nn.utils.clip_grad_norm_(net.parameters(), gradient_clipping_norm)
                optimizer.step()

            # Log losses for this batch (average of items if multiple recorded)
            if final_batch_loss.item() > 0 : # Only append if there was a loss
                 epoch_losses['total'].append(final_batch_loss.item())
            for key_log, loss_items_list in current_batch_losses_log.items():
                if loss_items_list and key_log in epoch_losses:
                    epoch_losses[key_log].append(np.mean(loss_items_list))
            
            if batch_num % 50 == 0 and actual_batch_size > 0 : # Print progress
                logger.debug(f"  Batch {batch_num}/{len(dataset)//batch_size}, Avg Batch Loss: {final_batch_loss.item():.4f}")


        # --- End of Epoch ---
        avg_epoch_total_loss = np.mean(epoch_losses['total'][-(len(dataset)//batch_size):]) if epoch_losses['total'] else 0.0
        logger.info(f"Epoch {epoch_idx+1} finished. Avg Total Loss: {avg_epoch_total_loss:.4f}")
        for loss_name_key in epoch_losses.keys():
            if loss_name_key != 'total':
                losses_list_epoch = epoch_losses[loss_name_key]
                # Only log if there were entries for this loss in the current epoch
                current_epoch_loss_entries = losses_list_epoch[-(len(dataset)//batch_size):] if len(losses_list_epoch) >= (len(dataset)//batch_size) else losses_list_epoch
                if current_epoch_loss_entries:
                    avg_loss_head = np.mean(current_epoch_loss_entries)
                    logger.info(f"  Avg {loss_name_key} Loss: {avg_loss_head:.4f}")
        logger.info("-" * 30)


    final_avg_losses_summary = {}
    for loss_name_key, all_losses_for_head in epoch_losses.items():
        if all_losses_for_head: # If any losses were recorded for this head throughout training
            final_avg_losses_summary[loss_name_key] = np.mean(all_losses_for_head)
        else:
            final_avg_losses_summary[loss_name_key] = 0.0 # Or mark as 'N/A'
    logger.info("Training finished.")
    return final_avg_losses_summary

def self_train(
    net: PolytopiaNet,
    iterations: int, n_games: int, epochs: int, n_sims: int,
    temperature: float, cPuct: float, gamma: float, deterministic: bool,
    batch_size: int, dirichlet: bool, rollouts: int, filename: str | None = None,
    settings: dict = {},
    # New training parameters
    learning_rate: float = 0.001,
    policy_loss_weights: dict = None,
    value_loss_weights: dict = None,
    gradient_clipping_norm: float = None
):
    logger.info("Self-training started.")
    logger.info(f"Device: {device}")
    logger.info(f"Iterations: {iterations}, Games/Iter: {n_games}, Epochs/Iter: {epochs}, Sims/Move: {n_sims}")
    logger.info(f"Temperature: {temperature}, cPuct: {cPuct}, Gamma: {gamma}, Deterministic: {deterministic}, Batch Size: {batch_size}")
    logger.info(f"Dirichlet Noise: {dirichlet}, Rollouts: {rollouts}")
    logger.info(f"Learning Rate: {learning_rate}, Grad Clip Norm: {gradient_clipping_norm}")
    logger.info(f"Policy Loss Weights: {policy_loss_weights}")
    logger.info(f"Value Loss Weights: {value_loss_weights}")


    for iteration_idx in range(iterations): # Renamed to iteration_idx
        logger.info(f"--- Iteration {iteration_idx + 1}/{iterations} ---")
        try:
            # CRITICAL ASSUMPTION: request_self_play returns data in the new Dataset format:
            # list[tuple[ObsDict, TargetPoliciesDict, TargetValuesDict, MoveTypeStr]]
            dataset: Dataset = request_self_play(
                n_games, n_sims, temperature, cPuct, gamma, deterministic, dirichlet, rollouts, settings
            )
            if not dataset:
                logger.warning("Received empty dataset from self-play. Skipping training for this iteration.")
                continue

            logger.info(f"Collected {len(dataset)} game states for training.")

            # Extract v_win for logging game outcomes, assuming 'v_win' is a key in TargetValuesDict
            game_outcomes = []
            for _, _, target_vals, _ in dataset:
                if 'v_win' in target_vals:
                    game_outcomes.append(target_vals['v_win'])
            
            if game_outcomes:
                logger.info(f"Game outcomes (v_win): Min={min(game_outcomes):.2f}, Max={np.max(game_outcomes):.2f}, Mean={np.mean(game_outcomes):.2f}")
            else:
                logger.info("No 'v_win' found in target values for outcome logging.")

            # Call the new train_network function
            avg_losses_dict = train_network(
                net, dataset, batch_size, epochs,
                learning_rate=learning_rate,
                policy_loss_weights=policy_loss_weights,
                value_loss_weights=value_loss_weights,
                gradient_clipping_norm=gradient_clipping_norm
            )

            log_message = f"Iteration {iteration_idx + 1} training complete. Avg Losses: "
            for loss_name, avg_loss_val in avg_losses_dict.items():
                log_message += f"{loss_name}={avg_loss_val:.4f}; "
            logger.info(log_message.strip("; "))

            if filename:
                current_filename = f"{filename}-iter{iteration_idx + 1}.zip"
                latest_filename = f"{filename}-latest.zip"
                torch.save(net.state_dict(), current_filename)
                torch.save(net.state_dict(), latest_filename)
                logger.info(f"Saved model to {current_filename} and {latest_filename}")

        except Exception as e:
            logger.exception(f"Exception during iteration {iteration_idx + 1}") # logger.exception includes stack trace

def request_train(*args, **kwargs):
    logger.info(f"Sending training request to server with args: {args}, kwargs: {kwargs}")
    try:
        response = post("http://localhost:3000/train", json=kwargs.get('json', args[0] if args else {}))
        response.raise_for_status()
        logger.info(f"Training request successful. Response: {response.json()}")
        return response.json()
    except Exception as e:
        logger.error(f"Training request failed: {e}")
        return None

def request_self_play(n_games: int, n_sims: int, temperature: float, cPuct: float, gamma: float,
    deterministic: bool, dirichlet: bool, rollouts: int, settings={}
) -> Dataset | list:
    logger.info(f"Requesting {n_games} self-play games from server...")
    payload = {
        "n_games": n_games, "n_sims": n_sims, "temperature": temperature,
        "cPuct": cPuct, "gamma": gamma, "deterministic": deterministic,
        "dirichlet": dirichlet, "rollouts": rollouts, "settings": settings,
    }
    try:
        response = post("http://localhost:3000/selfplay", json=payload)
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, list):
            logger.error(f"Self-play data is not a list: {type(data)}. Payload was: {payload}")
            return []
        logger.info(f"Received {len(data)} items from self-play server.")
        return data
    except Exception as e:
        logger.error(f"Self-play request failed: {e}. Payload was: {payload}")
        return []

