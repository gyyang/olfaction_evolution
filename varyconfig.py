"""Analyzing the resuts of varying configuration experiment."""

import os
import pickle
import json

import matplotlib.pyplot as plt

save_path = './files/vary_config/'
model_dirs = os.listdir(save_path)

logs = list()
configs = list()

for model_dir in model_dirs:
    # load log
    log_name = os.path.join(save_path, model_dir, 'log.pkl')
    with open(log_name, 'rb') as f:
        log = pickle.load(f)

    # load config
    config_name = os.path.join(save_path, model_dir, 'config.json')
    with open(config_name, 'r') as f:
        config = json.load(f)

    logs.append(log)
    configs.append(config)


# Get final performance and glo-score
final_accs = [log['val_acc'][-1] for log in logs]
glo_scores = [log['glo_score'][-1] for log in logs]

plt.figure()
plt.scatter(final_accs, glo_scores)
plt.xlabel('Final accuracy')
plt.ylabel('Glo score')