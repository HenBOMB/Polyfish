import torch
from train import PolyZeroNet

def test_zero_bug():
    # Initialize the network with dummy dimensions
    # __init__(self, spatial_channels, player_state_dim, map_height, map_width)
    net = PolyZeroNet(spatial_channels=17, player_state_dim=10, map_height=16, map_width=16)
    net.eval()
    
    # Batch size 1
    # State A: Feature 0 is 0.0, Feature 1 is 1.0 (rest are 0 for simplicity)
    state_a = torch.zeros((1, 10))
    state_a[0, 1] = 1.0
    
    # State B: Feature 0 is 1.0, Feature 1 is 0.0 (rest are 0)
    state_b = torch.zeros((1, 10))
    state_b[0, 0] = 1.0
    
    def get_player_tokens(state):
        with torch.no_grad():
            player_tokens = state.unsqueeze(-1) * net.player_feature_embeddings.unsqueeze(0)
            player_tokens = player_tokens + net.player_pos_embeddings.unsqueeze(0)
            player_tokens = net.player_relu(net.player_fc(player_tokens))
            return player_tokens
    
    tokens_a = get_player_tokens(state_a)
    tokens_b = get_player_tokens(state_b)
    
    print("--- Testing Token Identity Collapse (The Zero Bug) ---")
    print(f"Tokens shape: {tokens_a.shape}")
    
    token_0_when_zero = tokens_a[0, 0]
    token_1_when_zero = tokens_b[0, 1]
    
    print(f"Token 0 vector (first 5 elements) when Feature 0 is 0.0: {token_0_when_zero[:5]}")
    print(f"Token 1 vector (first 5 elements) when Feature 1 is 0.0: {token_1_when_zero[:5]}")
    
    diff = torch.abs(token_0_when_zero - token_1_when_zero).sum().item()
    print(f"Total absolute difference between these two token vectors: {diff}")
    
    if diff == 0.0:
        print("\n[RESULT] COLLAPSE CONFIRMED: The network produces identical token vectors for different features when their values are 0.0.")
    else:
        print("\n[RESULT] NO COLLAPSE: The tokens are distinct.")

if __name__ == '__main__':
    test_zero_bug()
