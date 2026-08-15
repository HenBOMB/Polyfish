"""Pure-PyTorch/MPS forward benchmark, measured the SAME way as the integrated
candle/tch eval-server path: batch 128, full model, fresh host->MPS upload every
call, full readback of all outputs to CPU.

Comparable to: candle 71.7ms/forward, tch 15.2ms/forward (integrated, batch ~130).
"""
import time, numpy as np, torch
from safetensors.torch import load_file
from train import PolyZeroNet, _migrate_checkpoint

BATCH = 128
ITERS = 60
dev = torch.device("mps")

net = PolyZeroNet(142, 16, 11, 11)
ckpt = load_file("model.safetensors")
ckpt, _ = _migrate_checkpoint(ckpt, net, 16)
net.load_state_dict(ckpt, strict=False)
net.to(dev).eval()

def one_forward():
    # fresh host data every call (matches eval_server building a Vec<f32> batch)
    sp_host = np.random.rand(BATCH, 142, 11, 11).astype(np.float32)
    pl_host = np.random.rand(BATCH, 16).astype(np.float32)
    # host -> MPS upload (matches Tensor::from_slice(...).to_device(Mps))
    sp = torch.from_numpy(sp_host).to(dev)
    pl = torch.from_numpy(pl_host).to(dev)
    with torch.no_grad():
        policy, values = net(sp, pl)
    # full readback of all 5 outputs to CPU (matches reading value + 4 heads)
    _ = values["win"].cpu().numpy()
    for k in ("action_type", "move_option", "source_spatial", "target_spatial"):
        _ = policy[k].cpu().numpy()

# warmup (shader compile, allocator warmup)
for _ in range(8):
    one_forward()

times = []
for _ in range(ITERS):
    t0 = time.perf_counter()
    one_forward()
    times.append((time.perf_counter() - t0) * 1e3)

times.sort()
n = len(times)
print(f"pure PyTorch/MPS  batch={BATCH}  full model + fresh upload + full readback")
print(f"  median={times[n//2]:.2f}ms  mean={sum(times)/n:.2f}ms  min={times[0]:.2f}ms  p90={times[9*n//10]:.2f}ms")
print()
print(f"  candle (integrated): 71.7 ms/forward")
print(f"  tch    (integrated): 15.2 ms/forward")
print(f"  python (this test):  {times[n//2]:.1f} ms/forward")
