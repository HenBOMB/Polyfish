"""Compare the Rust tch forward against train.py's PolyZeroNet (ground truth)."""
import json, numpy as np, torch
from safetensors.torch import load_file
from train import PolyZeroNet

d = json.load(open("/tmp/parity_rust.json"))
b = d["batch"]
spatial = torch.tensor(d["spatial"], dtype=torch.float32).reshape(b, 161, 11, 11)
player = torch.tensor(d["player"], dtype=torch.float32).reshape(b, 10)

net = PolyZeroNet(161, 10, 11, 11)
net.load_state_dict(load_file("model.safetensors"))
net.eval()
with torch.no_grad():
    policy, values = net(spatial, player)

def cmp(name, py, rust):
    py = py.numpy()
    rust = np.array(rust).reshape(py.shape)
    # compare softmax for policy heads, raw for value
    if name != "value":
        e = lambda z: np.exp(z - z.max(axis=1, keepdims=True))
        sm = lambda z: e(z) / e(z).sum(axis=1, keepdims=True)
        py, rust = sm(py), sm(rust)
    print(f"  {name:8s} max|Δ|={np.abs(py - rust).max():.3e}")

print(f"batch={b}  PyTorch(train.py) vs Rust tch:")
cmp("value", values["win"], d["tch_value"])
cmp("action", policy["action_type"], d["tch_action"])
cmp("source", policy["source_spatial"], d["tch_source"])
cmp("target", policy["target_spatial"], d["tch_target"])
cmp("option", policy["move_option"], d["tch_option"])
