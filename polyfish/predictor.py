import threading
import time
import numpy as np
import torch
import torch.nn.functional as F

# 1) Configuration
MAX_BATCH = 64        # max number of obs to batch
MAX_DELAY = 0.001      # max time (s) to wait for a batch

# 2) A simple request object that callers block on
class BatchRequest:
    def __init__(self, obs_tensor):
        self.obs = obs_tensor
        self.event = threading.Event()
        self.result = None  # will hold (pi, v)

# 3) The batcher
class PredictorBatcher:
    def __init__(self, model):
        self.model = model
        self.lock = threading.Lock()
        self.queue = []  # list of BatchRequest
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def predict(self, obs_tensor):
        # Called by main thread for each incoming predict
        req = BatchRequest(obs_tensor)
        with self.lock:
            self.queue.append(req)
            # If we hit max batch size, wake the worker immediately
            if len(self.queue) >= MAX_BATCH:
                # Notify by setting a flag or simply let worker see
                pass
        # Wait for the background worker to fill this req.result
        req.event.wait()
        return req.result

    def _worker(self):
        while True:
            time.sleep(MAX_DELAY)  # small delay to gather requests
            with self.lock:
                if not self.queue:
                    continue
                # Pop up to MAX_BATCH requests
                batch, self.queue = self.queue[:MAX_BATCH], self.queue[MAX_BATCH:]

            # Build a batched input
            batched_map = torch.cat([r.obs['map'] for r in batch], dim=0)       # [B, C, S, S]
            batched_player = torch.cat([r.obs['player'] for r in batch], dim=0) # [B, T]

            # Run one forward pass
            with torch.no_grad():
                output = self.model({
                    'map': batched_map,
                    'player': batched_player
                })
                pi_action = F.softmax(output['pi_action_logits'], dim=-1)
                pi_actor = F.softmax(output['pi_actor_logits'], dim=-1)
                pi_target = F.softmax(output['pi_target_logits'], dim=-1)
                pi_option_struct = F.softmax(output['pi_option_struct_logits'], dim=-1)
                pi_option_skill = F.softmax(output['pi_option_skill_logits'], dim=-1)
                pi_option_unit = F.softmax(output['pi_option_unit_logits'], dim=-1)
                pi_option_tech = F.softmax(output['pi_tech_logits'], dim=-1)
                pi_option_reward = torch.sigmoid(output['pi_reward_logits'])
                v_win = output['v_win'].cpu().numpy()
                v_eco = output['v_eco'].cpu().numpy()
                v_mil = output['v_mil'].cpu().numpy()

            # Scatter results back to requests
            for i, req in enumerate(batch):
                req.result = (
                    [_.item() for _ in pi_action[i]], 
                    [_.item() for _ in pi_actor[i]], 
                    [_.item() for _ in pi_target[i]], 
                    [_.item() for _ in pi_option_struct[i]], 
                    [_.item() for _ in pi_option_skill[i]], 
                    [_.item() for _ in pi_option_unit[i]], 
                    [_.item() for _ in pi_option_tech[i]], 
                    [_.item() for _ in pi_option_reward[i]], 
                    v_win[i].item(),
                    v_eco[i].item(),
                    v_mil[i].item()
                )
                req.event.set()
