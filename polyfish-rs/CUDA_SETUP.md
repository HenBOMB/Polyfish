# GPU/CUDA Setup Guide

## Issue Fixed
**Problem:** Device mismatch error - network on CUDA but tensors on CPU  
**Solution:** Made tensors use the same device as the network

## Changes Made

### 1. Updated `state_to_tensor` Function
- **File:** `src/ai/features.rs`
- **Change:** Added `device: &Device` parameter
- Now creates tensors on the correct device (CPU or CUDA)

### 2. Added `device()` Helper to Network
- **File:** `src/ai/network.rs`
- **Change:** Added `pub fn device(&self) -> Device`
- Returns the device the network is on by checking layer weights

### 3. Made CUDA Optional
- **File:** `Cargo.toml`
- **Change:** CUDA is now a **feature flag** instead of always-on
- This allows building without CUDA on local machines

## How to Build

### **On Colab (with T4 GPU):**
```bash
cargo build --release --features cuda --bin self_play
cargo run --release --features cuda --bin benchmark
```

### **On Local Machine (CPU only):**
```bash
cargo build --release --bin self_play
cargo run --release --bin benchmark
```

## Updated Files
- ✅ `src/ai/features.rs` - Added device parameter
- ✅ `src/ai/network.rs` - Added device() method
- ✅ `src/ai/mcts_zero.rs` - Uses network.device()
- ✅ `src/bin/self_play.rs` - Uses network.device()
- ✅ `src/bin/benchmark.rs` - Uses device parameter
- ✅ `Cargo.toml` - Made CUDA optional
- ✅ `production.ipynb` - Uses --features cuda
- ✅ `main.ipynb` - Uses --features cuda

## Testing

### Locally (CPU):
```bash
cargo run --release --bin benchmark
# Should print "Using device: Cpu"
```

### On Colab (GPU):
```bash
cargo run --release --features cuda --bin benchmark
# Should print "Using device: Cuda(0)"
```

## Expected Performance

| Device | NN Inference | 80-move Game |
|--------|--------------|--------------|
| CPU | ~17ms | ~71s |
| T4 GPU | ~1-2ms | **~5-10s** |

**Speedup: 10-15x faster on GPU!** 🚀
