import tools
import train
import os
import mamlmetatrain

SBATCHPATH = './sbatch/'
SCRATCHPATH = '/axsys/scratch/ctn/users/gy2259/olfaction_evolution'


def basic_train(experiment, save_path):
    config = experiment()
    config.save_path = save_path
    train.train(config)


def local_train(experiment, save_path, sequential=False, control=False,
                train_arg=None, **kwargs):
    """Train all models locally."""
    for i in range(0, 1000):
        if sequential:
            config = tools.varying_config_sequential(experiment, i)
        elif control:
            config = tools.varying_config_control(experiment, i)
        else:
            config = tools.varying_config(experiment, i)
        if config:
            print('[***] Hyper-parameter: %2d' % i)
            config.save_path = os.path.join(save_path, str(i).zfill(6))

            if train_arg == None:
                train.train(config, **kwargs)
            elif train_arg == 'metalearn':
                mamlmetatrain.train(config)
            else:
                raise ValueError('training function is not recognized by keyword {}'.format(train_arg))


def obsolete_local_sequential_train(experiment, save_path, train_arg = None):
    """Train all models locally."""
    for i in range(0, 1000):
        config = tools.varying_config_sequential(experiment, i)
        if config:
            print('[***] Hyper-parameter: %2d' % i)
            config.save_path = os.path.join(save_path, str(i).zfill(6))

            if train_arg == None:
                train.train(config)
            elif train_arg == 'metalearn':
                mamlmetatrain.train(config)
            else:
                raise ValueError('training function is not recognized by keyword {}'.format(train_arg))

def obsolete_local_control_train(experiment, save_path, train_arg = None):
    '''
    Train each hyper-parameter separately
    '''
    for i in range(0, 1000):
        config = tools.varying_config_control(experiment, i)
        if config:
            print('[***] Hyper-parameter: %2d' % i)
            config.save_path = os.path.join(save_path, str(i).zfill(6))

            if train_arg == None:
                train.train(config)
            elif train_arg == 'metalearn':
                mamlmetatrain.train(config)
            else:
                raise ValueError('training function is not recognized by keyword {}'.format(train_arg))


def write_jobfile(cmd, jobname, sbatchpath=SBATCHPATH, scratchpath=SCRATCHPATH,
                  nodes=1, ppn=1, gpus=0, mem=16, nhours=1):
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
    return jobfile


import subprocess

def cluster_train(experiment, save_path, sequential=False, control=False):
    """Train all models locally."""
    job_name = save_path.split('/')[-1]  # get end of path as job name
    config = tools.varying_config(experiment, 0)
    original_data_dir = config.data_dir[2:]  # HACK

    for i in range(0, 1000):
        if sequential:
            config = tools.varying_config_sequential(experiment, i)
        elif control:
            config = tools.varying_config_control(experiment, i)
        else:
            config = tools.varying_config(experiment, i)

        if config:
            config.save_path = os.path.join(SCRATCHPATH, 'files', job_name, str(i).zfill(6))

            # TEMPORARY HACK
            # TODO: Fix bug when data_dir is not always the same
            config.data_dir = os.path.join(SCRATCHPATH, original_data_dir)
            os.makedirs(config.save_path, exist_ok=True)

            tools.save_config(config, config.save_path)

            arg =  '\'' + config.save_path + '\''

            cmd = r'''python -c "import train; train.train_from_path(''' + arg + ''')"'''

            jobfile = write_jobfile(cmd, job_name + str(i))
            subprocess.call(['sbatch', jobfile])
