import torch
import torch.nn.functional as F
import random
import math
from typing import List, Dict, Tuple, Any
from gameTypes import *

def coord_to_index(coord: Tuple[int, int], max_size: int) -> int:
    """Converts (x, y) map coordinate to a flat index."""
    x, y = coord
    return y * max_size + x

# We don't strictly need index_to_coord for this logic, but it's good to have
def index_to_coord(index: int, max_size: int) -> Tuple[int, int]:
    """Converts a flat index back to (x, y) map coordinate."""
    y = index // max_size
    x = index % max_size
    return (x, y)

# --- Core Translation Function ---

def translate_prediction_to_move(
    predictions: Dict[str, torch.Tensor],
    valid_moves: List[Dict[str, Any]],
    max_size: int,
    temperature: float = 1.0,
    device: torch.device = torch.device('cpu')
) -> Dict[str, Any] | None:
    """
    Selects a single valid move based on network predictions and temperature.
    Returns:
        The selected Action dictionary, or None if no valid moves are available.
    """
    if not valid_moves:
        print("Warning: No valid moves provided.")
        return None # Or potentially return a default 'EndTurn' action if appropriate

    num_valid_moves = len(valid_moves)
    action_log_probs = torch.full((num_valid_moves,), -float('inf'), dtype=torch.float32, device=device)

    # --- Pre-calculate log probabilities for all components ---
    # Use log_softmax for numerical stability
    # Apply temperature *before* log_softmax
    # Ensure logits are on the correct device BEFORE operations
    t = max(temperature, 1e-9) # Avoid division by zero if T=0

    log_prob_action_type = F.log_softmax(predictions['pi_action_logits'].to(device) / t, dim=-1).squeeze(0) # Assuming batch size 1
    log_prob_actor       = F.log_softmax(predictions['pi_actor_logits'].to(device) / t, dim=-1).squeeze(0)
    log_prob_target      = F.log_softmax(predictions['pi_target_logits'].to(device) / t, dim=-1).squeeze(0)
    log_prob_struct      = F.log_softmax(predictions['pi_option_struct_logits'].to(device) / t, dim=-1).squeeze(0)
    log_prob_skill       = F.log_softmax(predictions['pi_option_skill_logits'].to(device) / t, dim=-1).squeeze(0)
    log_prob_unit        = F.log_softmax(predictions['pi_option_unit_logits'].to(device) / t, dim=-1).squeeze(0)
    log_prob_tech        = F.log_softmax(predictions['pi_tech_logits'].to(device) / t, dim=-1).squeeze(0)

    # --- Calculate log probability for each valid move ---
    for i, move in enumerate(valid_moves):
        current_move_log_prob = 0.0

        # 1. Action Type Probability
        move_type_str = move.get('action', 'None')
        move_type_idx = MOVE_TYPE.get(move_type_str, -1)
        if move_type_idx == -1:
             print(f"Warning: Unknown MoveType '{move_type_str}' in valid_moves. Skipping.")
             continue # Skip this invalid move description
        current_move_log_prob += log_prob_action_type[move_type_idx]

        # --- Add probabilities for required components based on move type ---

        # TODO ensure compatibility with TS simulator
        
        # 2. Actor ('from') Probability
        if move_type_str in ['Step', 'Attack', 'Ability', 'Capture']:
            from_coord = move.get('from')
            if from_coord:
                actor_idx = coord_to_index(tuple(from_coord), max_size)
                current_move_log_prob += log_prob_actor[actor_idx]
            else:
                print(f"Warning: Move '{move_type_str}' requires 'from' but not found. Assigning -inf prob.")
                current_move_log_prob = -float('inf') # Invalid move structure

        # 3. Target ('to') Probability
        if move_type_str in ['Step', 'Attack', 'Ability', 'Summon', 'Harvest', 'Build']:
            to_coord = move.get('to')
            if to_coord:
                target_idx = coord_to_index(tuple(to_coord), max_size)
                current_move_log_prob += log_prob_target[target_idx]
            else:
                print(f"Warning: Move '{move_type_str}' requires 'to' but not found. Assigning -inf prob.")
                current_move_log_prob = -float('inf')

        # 4. Structure Probability
        if move_type_str == 'Build':
            struct_type_str = move.get('struct', 'None')
            struct_idx = BUILD_TYPE.get(struct_type_str, -1)
            if struct_idx != -1:
                current_move_log_prob += log_prob_struct[struct_idx]
            else:
                 print(f"Warning: Move '{move_type_str}' requires valid 'struct' but found '{struct_type_str}'. Assigning -inf prob.")
                 current_move_log_prob = -float('inf')

        # 5. Ability/Skill Probability
        # TODO not all skills are spatial, some require "from", while others "to"
        if move_type_str == 'Ability':
            ability_type_str = move.get('ability', 'None')
            ability_idx = ABILITY_TYPE.get(ability_type_str, -1)
            if ability_idx != -1:
                 current_move_log_prob += log_prob_skill[ability_idx]
            else:
                 print(f"Warning: Move '{move_type_str}' requires valid 'ability' but found '{ability_type_str}'. Assigning -inf prob.")
                 current_move_log_prob = -float('inf')

        # 6. Unit Probability
        if move_type_str == 'Summon':
             unit_type_str = move.get('unit', 'None')
             unit_idx = SUMMON_TYPE.get(unit_type_str, -1)
             if unit_idx != -1:
                 current_move_log_prob += log_prob_unit[unit_idx]
             else:
                 print(f"Warning: Move '{move_type_str}' requires valid 'unit' but found '{unit_type_str}'. Assigning -inf prob.")
                 current_move_log_prob = -float('inf')

        # 7. Technology Probability
        if move_type_str == 'Research':
            tech_type_str = move.get('tech', 'None')
            tech_idx = TECHNOLOGY_TYPE.get(tech_type_str, -1)
            if tech_idx != -1:
                current_move_log_prob += log_prob_tech[tech_idx]
            else:
                 print(f"Warning: Move '{move_type_str}' requires valid 'tech' but found '{tech_type_str}'. Assigning -inf prob.")
                 current_move_log_prob = -float('inf')

        # --- Store the calculated log probability ---
        # Only store if it's not -inf (meaning the move structure was valid)
        if not math.isinf(current_move_log_prob):
             action_log_probs[i] = current_move_log_prob

    # --- Select Move ---
    if torch.all(action_log_probs == -float('inf')):
        print("Warning: All valid moves received -inf probability. Selecting randomly.")
        # Fallback: If all moves somehow got invalid scores, choose randomly
        # This might indicate a problem in the mappings or network outputs
        return random.choice(valid_moves) if valid_moves else None

    if temperature <= 1e-9: # Argmax selection (Greedy)
        best_move_idx = torch.argmax(action_log_probs).item()
    else: # Probabilistic sampling
        # Use torch.multinomial which expects log-probabilities or probabilities
        # It's safer to work with probabilities here after calculating log probs
        action_probs = F.softmax(action_log_probs, dim=0) # Normalize log_probs into probabilities
        # Check for NaNs which can occur if all log_probs were -inf (handled above)
        # or if softmax underflows/overflows (less likely with log_softmax start)
        if torch.isnan(action_probs).any():
            print(f"Warning: NaNs detected in action probabilities after softmax. Log Probs: {action_log_probs}. Falling back to random choice.")
            # Fallback to random choice among those that didn't have -inf log_prob initially
            valid_indices = [i for i, lp in enumerate(action_log_probs.tolist()) if not math.isinf(lp)]
            if not valid_indices: return random.choice(valid_moves) # Should not happen if initial check passed
            chosen_idx = random.choice(valid_indices)
            return valid_moves[chosen_idx]

        best_move_idx = torch.multinomial(action_probs, num_samples=1).item()

    return valid_moves[best_move_idx]

# --- Example Usage ---
if __name__ == '__main__':
    # --- Dummy Data Setup ---
    MAP_SIZE = 5 ** 2
    DIM_ACTION = MOVE_TYPE['_MAX_N']
    DIM_STRUCT = BUILD_TYPE['_MAX_N']
    DIM_SKILL  = ABILITY_TYPE['_MAX_N']
    DIM_UNIT   = SUMMON_TYPE['_MAX_N']
    DIM_TECH   = TECHNOLOGY_TYPE['_MAX_N']
    BATCH_SIZE = 1 # This function assumes batch size 1 for predictions
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Dummy Network Predictions (Logits)
    dummy_predictions = {
        'pi_action_logits': torch.randn(BATCH_SIZE, DIM_ACTION, device=DEVICE),
        'pi_actor_logits': torch.randn(BATCH_SIZE, MAP_SIZE, device=DEVICE),
        'pi_target_logits': torch.randn(BATCH_SIZE, MAP_SIZE, device=DEVICE),
        'pi_option_struct_logits': torch.randn(BATCH_SIZE, DIM_STRUCT, device=DEVICE),
        'pi_option_skill_logits': torch.randn(BATCH_SIZE, DIM_SKILL, device=DEVICE),
        'pi_option_unit_logits': torch.randn(BATCH_SIZE, DIM_UNIT, device=DEVICE),
        'pi_tech_logits': torch.randn(BATCH_SIZE, DIM_TECH, device=DEVICE),
        # Add value heads if needed by other parts, but not used here
        'v_win': torch.randn(BATCH_SIZE, 1, device=DEVICE)
    }

    # 2. Dummy Valid Moves (List of Action dictionaries)
    # Use the string representations defined in the mapping dicts
    dummy_valid_moves = [
        {'action': 'EndTurn'},
        {'action': 'Step', 'from': [1, 1], 'to': [1, 2]},
        {'action': 'Attack', 'from': [1, 1], 'to': [2, 2]},
        {'action': 'Build', 'to': [0, 0], 'struct': 'Farm'},
        {'action': 'Build', 'to': [0, 0], 'struct': 'Mine'},
        {'action': 'Research', 'tech': 'Farming'},
        {'action': 'Summon', 'to': [0, 1], 'unit': 'Warrior'},
        {'action': 'Ability', 'from': [3,3], 'to': [3,4], 'ability': 'Boost'}
        # {'action': 'InvalidActionType', 'from': [0,0], 'to': [0,0]} # Test invalid type
        # {'action': 'Step', 'to': [0,0]} # Test missing 'from'
    ]

    print(dummy_predictions['pi_action_logits'])

    print("Action Logits:", dummy_predictions['pi_action_logits'].shape) # Can be noisy

    selected_move_sample = translate_prediction_to_move(
        dummy_predictions, dummy_valid_moves, MAP_SIZE, temperature=1.0, device=DEVICE
    )
    print(f"Selected Move: {selected_move_sample}")

    # print("\n--- Selecting Move (Temperature = 0.0 - Argmax/Greedy) ---")
    # selected_move_greedy = translate_prediction_to_move(
    #     dummy_predictions, dummy_valid_moves, MAP_WIDTH, MAP_HEIGHT, temperature=0.0, device=DEVICE
    # )
    # print(f"Selected Move: {selected_move_greedy}")

    # print("\n--- Selecting Move (High Temperature = 10.0 - Near Uniform) ---")
    # selected_move_uniform = translate_prediction_to_move(
    #     dummy_predictions, dummy_valid_moves, MAP_WIDTH, MAP_HEIGHT, temperature=10.0, device=DEVICE
    # )
    # print(f"Selected Move: {selected_move_uniform}")

    # print("\n--- Selecting Move (No Valid Moves) ---")
    # selected_move_none = translate_prediction_to_move(
    #     dummy_predictions, [], MAP_WIDTH, MAP_HEIGHT, temperature=1.0, device=DEVICE
    # )
    # print(f"Selected Move: {selected_move_none}")

    print()