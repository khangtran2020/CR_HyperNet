#!/bin/bash -l
# The above line must always be first, and must have "-l"
#SBATCH -J hpn_udp
#SBATCH -p datasci
#SBATCH --output=results/userdp_hypernet.out
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
module load python
conda activate hypernet
python userdp_hypernet_cifar10.py --grad_clip 1.0 --delta 0.00001 --noise_scale 1.5 --num_comp_cli 1

