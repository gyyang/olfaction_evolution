import os
import subprocess
from pathlib import Path

import standard.experiments as experiments
import standard.experiment_controls as experiment_controls
import standard.experiment_metas as experiment_metas
import tools
import settings


use_torch = settings.use_torch


def local_train(config, path=None, **kwargs):
    """Train all models locally."""
    if use_torch:
        import torchtrain as train
        import temp_meta.metatrain as metatrain
    else:
        import train
        import mamlmetatrain as metatrain

    if path is None:
        path = './'

    experiment_name = config.experiment_name
    model_name = config.model_name
    config.save_path = os.path.join(path, 'files', experiment_name, model_name)

    use_metatrain = 'meta_lr' in dir(config)

    if not use_metatrain:
        train.train(config, **kwargs)
    else:
        metatrain.train(config)


def write_jobfile(cmd, jobname, sbatchpath='./sbatch/',
                  nodes=1, ppn=1, gpus=0, mem=16, nhours=3):
    """
    Create a job file.

    Args:
        cmd : str, Command to execute.
        jobname : str, Name of the job.
        sbatchpath : str, Directory to store SBATCH file in.
        scratchpath : str, Directory to store output files in.
        nodes : int, optional, Number of compute nodes.
        ppn : int, optional, Number of cores per node.
        gpus : int, optional, Number of GPU cores.
        mem : int, optional, Amount, in GB, of memory.
        ndays : int, optional, Running time, in days.
        queue : str, optional, Queue name.

    Returns:
        jobfile : str, Path to the job file.
    """

    os.makedirs(sbatchpath, exist_ok=True)
    jobfile = os.path.join(sbatchpath, jobname + '.s')
    # logname = os.path.join('log', jobname)

    with open(jobfile, 'w') as f:
        f.write(
            '#! /bin/sh\n'
            + '\n'
            # + '#SBATCH --nodes={}\n'.format(nodes)
            # + '#SBATCH --ntasks-per-node=1\n'
            # + '#SBATCH --cpus-per-task={}\n'.format(ppn)
            + '#SBATCH --mem-per-cpu={}gb\n'.format(mem)
            # + '#SBATCH --partition=xwang_gpu\n'
            + '#SBATCH --gres=gpu:1\n'
            + '#SBATCH --time={}:00:00\n'.format(nhours)
            # + '#SBATCH --mem=128gb\n'
            # + '#SBATCH --job-name={}\n'.format(jobname[0:16])
            # + '#SBATCH --output={}log/{}.o\n'.format(scratchpath, jobname[0:16])
            + '\n'
            # + 'cd {}\n'.format(scratchpath)
            # + 'pwd > {}.log\n'.format(logname)
            # + 'date >> {}.log\n'.format(logname)
            # + 'which python >> {}.log\n'.format(logname)
            # + '{} >> {}.log 2>&1\n'.format(cmd, logname)
            + cmd + '\n'
            + '\n'
            + 'exit 0;\n'
            )
        print(jobfile)
    return jobfile


def cluster_train(config, path):
    """Train all models locally."""
    experiment_name = config.experiment_name
    model_name = config.model_name

    config.save_path = os.path.join(path, 'files', experiment_name, model_name)
    os.makedirs(config.save_path, exist_ok=True)

    # TEMPORARY HACK
    # Hack: assuming data_dir of form './files/XX'
    config.data_dir = os.path.join(path, config.data_dir[2:])

    tools.save_config(config, config.save_path)

    arg = '\'' + config.save_path + '\''

    use_metatrain = 'meta_lr' in dir(config)

    if use_metatrain:
        if use_torch:
            cmd = r'''python -c "import temp_meta.metatrain as metatrain;metatrain.train_from_path(''' + arg + ''')"'''
        else:
            cmd = r'''python -c "import mamlmetatrain as metatrain;metatrain.train_from_path(''' + arg + ''')"'''
    else:
        if use_torch:
            cmd = r'''python -c "import torchtrain; torchtrain.train_from_path(''' + arg + ''')"'''
        else:
            cmd = r'''python -c "import train; train.train_from_path(''' + arg + ''')"'''

    jobfile = write_jobfile(cmd, jobname=experiment_name + '_' + model_name,
                            mem=12)
    subprocess.call(['sbatch', jobfile])


def train_experiment(experiment, use_cluster=False, path=None,
                     testing=False, n_pn=None, **kwargs):
    """Train model across platforms given experiment name.

    Args:
        experiment: str, name of experiment to be run
            must correspond to a function in experiments.py
        use_cluster: bool, whether to run experiments on cluster
        path: str, path to save models and config
        train_arg: None or str
        testing: bool, whether to test run
    """
    if path is None:
        # Default path
        if use_cluster:
            path = settings.cluster_path
        else:
            path = Path('./')

    print('Training {:s} experiment'.format(experiment))
    experiment_files = [experiments, experiment_controls, experiment_metas]

    experiment_found = False
    for experiment_file in experiment_files:
        if experiment in dir(experiment_file):
            # Get list of configurations from experiment function
            if n_pn is None:
                configs = getattr(experiment_file, experiment)()
            else:
                configs = getattr(experiment_file, experiment)(n_pn=n_pn)
            experiment_found = True
            break
        else:
            experiment_found = False

    if not experiment_found:
        raise ValueError('Experiment not found: ', experiment)

    for config in configs:
        if n_pn is None:
            config.experiment_name = experiment
        else:
            config.experiment_name = experiment + '_pn' + str(n_pn)
        if testing:
            config.max_epoch = 2

        if use_cluster:
            cluster_train(config, path=path)
        else:
            local_train(config, path=path, **kwargs)


def analyze_experiment(experiment, n_pn=None):
    path = './files/' + experiment

    experiment_files = [experiments, experiment_controls, experiment_metas]

    experiment_found = False
    for experiment_file in experiment_files:
        if (experiment + '_analysis') in dir(experiment_file):
            if n_pn is None:
                getattr(experiment_file, experiment + '_analysis')(path)
            else:
                path = path + '_pn' + str(n_pn)
                getattr(experiment_file, experiment + '_analysis')(path, n_pn=n_pn)
            experiment_found = True
            break
        else:
            experiment_found = False

    if not experiment_found:
        print('Analysis not found for experiment', experiment)