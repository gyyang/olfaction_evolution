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

# Analyzing all models, while showing relevant configs
keys = ['sparse_pn2kc', 'train_pn2kc', 'direct_glo', 'sign_constraint']
for config, log in zip(configs, logs):
    title_txt = '\n'.join([key + ' : ' + str(config[key]) for key in keys])
    fig = plt.figure(figsize=(2, 2))
    ax = fig.add_axes([0.2, 0.2, 0.7, 0.7])
    ax.plot(log['epoch'], log['glo_score'])
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Glo score')
    ax.set_title(title_txt)
    plt.ylim([0, 1])