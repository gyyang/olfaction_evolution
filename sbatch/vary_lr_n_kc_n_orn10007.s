#! /bin/sh

#SBATCH --mem-per-cpu=16gb
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00

python -c "import train; train.train_from_path('/axsys/scratch/ctn/users/yw2500/olfaction_evolution/files/vary_lr_n_kc_n_orn1000/000007')"

exit 0;
