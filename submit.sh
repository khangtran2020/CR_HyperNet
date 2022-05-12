#!/bin/bash -l
# The above line must always be first, and must have "-l"
#SBATCH -J hpn_clean
#SBATCH -p datasci
#SBATCH --output=results/hpn_udp_bt_10_step_10000_noisescale_50_gradclip_0.000001_.out
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
module load python
conda activate hypernet
python main.py --train_mode userdp --grad_clip 0.000001 --udp_delta 0.00001 --noise_scale 50  --num_comp_cli 1 --bt 10 --num_steps 10000 --optim adam --lr 0.0001

