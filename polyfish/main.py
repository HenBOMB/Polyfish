import numpy as np
import torch, json
import torch.nn.functional as F
import json, torch, sys, os
import threading
import model
from predictor import PredictorBatcher

with open('data/model/config.json', 'r') as f:
    config = json.load(f)
    config['max_tile_count'] = config['dim_map_size'] ** 2

root_path = 'models/polyfish'
prefix = ''
net = model.load(root_path + '-latest', config)
training = False
train_thread = None
predictor = PredictorBatcher(net)

def reply(data):
    sys.stdout.write(json.dumps(data))
    sys.stdout.flush()

def obs_to_tensor(value: dict):
    return {
        'map': torch.tensor(np.array(value['map'])).to(model.device).unsqueeze(0).float(),
        'player': torch.tensor(np.array(value['player'])).to(model.device).unsqueeze(0).float()
    }

while True:
    line = sys.stdin.readline()
    if not line:
        break

    try:
        data = json.loads(line)
    except:
        model.logger.exception("Invalid JSON: %s", line)
        continue

    cmd = data['cmd']

    if cmd == 'train':
        filepath = root_path +  data.get('prefix', prefix)

        if training:
            reply({ "status": 'busy' })
            continue

        training = True

        def _train_wrapper():
            global training, net
            try:
                model.self_train(
                    net,
                    data.get('iterations', 1000),
                    data.get('n_games', 3),
                    data.get('epochs', 100),
                    data.get('n_sims', 1000),
                    data.get('temperature', 0.7),
                    data.get('cPuct', 1.0),
                    data.get('gamma', 0.997),
                    data.get('deterministic', False),
                    data.get('batch_size', 16),
                    data.get('dirichlet', True),
                    data.get('rollouts', 50),
                    filepath,
                    data.get('settings', {}),
                )
            except Exception as e:
                model.logger.exception("Training thread crashed")
            finally:
                training = False

        train_thread = threading.Thread(target=_train_wrapper, daemon=True)
        train_thread.start()

        reply({ "status": 'success' })

    elif cmd == 'predict':
        output = predictor.predict(obs_to_tensor(data))
        reply({
            'pi_action': output[0],
            'pi_actor': output[1],
            'pi_target': output[2],
            'pi_struct': output[3],
            'pi_skill': output[4],
            'pi_unit': output[5],
            'pi_tech': output[6],
            'pi_reward': output[7],
            'v_win': output[8],
            'v_eco': output[9],
            'v_mil': output[10],
        })
