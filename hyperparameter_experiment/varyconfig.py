"""Analyzing the resluts of varying configuration experiment."""

import os
import pickle
import json
import numpy as np

import matplotlib.pyplot as plt

save_path = './files/vary_config/'
model_dirs = os.listdir(save_path)
keys = ['sparse_pn2kc', 'train_pn2kc', 'direct_glo', 'sign_constraint']

logs = list()
configs = list()
coordinates = list()

for model_dir in model_dirs:
    # load log
    log_name = os.path.join(save_path, model_dir, 'log.pkl')
    with open(log_name, 'rb') as f:
        log = pickle.load(f)

    # load config
    config_name = os.path.join(save_path, model_dir, 'config.json')
    with open(config_name, 'r') as f:
        config = json.load(f)

    # append only if there is no direct
    if not config['direct_glo']:
        logs.append(log)
        configs.append(config)
        coordinates.append('_'.join([str(int(config[key])) for key in keys]))

#TODO: rerun experiments after inhibition
# validation accuracy is actually best for no sign constraint, despite low glo score
#glo score is actually higher when pn->kc is not sparse
#training has no effect
glo_scores_timeseries = np.array([log['glo_score'] for log in logs]).transpose()
plt.figure()
plt.plot(glo_scores_timeseries)
plt.legend(coordinates)

val_acc_timeseries = np.array([log['val_acc'] for log in logs]).transpose()
plt.figure()
plt.plot(val_acc_timeseries)
plt.legend(coordinates)
plt.show()

# Get final performance and glo-score
final_accs = [log['val_acc'][-1] for log in logs]
glo_scores = [log['glo_score'][-1] for log in logs]

plt.figure()
plt.scatter(final_accs, glo_scores)
plt.xlabel('Final accuracy')
plt.ylabel('Glo score')

# Analyzing all models, while showing relevant configs
for config, log in zip(configs, logs):
    title_txt = '\n'.join([key + ' : ' + str(config[key]) for key in keys])
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_axes([0.2, 0.2, 0.7, 0.7])
    ax.plot(log['epoch'], log['glo_score'])
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Glo score')
    ax.set_title(title_txt)
    plt.ylim([0, 1])

plt.show()