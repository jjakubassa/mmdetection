#!/bin/bash
#SBATCH --partition=gpu_4
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --mem=10G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jonas.jakubassa@students.uni-mannheim.de

#  Prepare software
module load devel/cuda/11.8

# Config
CHECKPOINT_FILE="checkpoints/retinanet_x101_64x4d_fpn_1x_coco_20200130-366f5af1.pth"
CONFIG_FILE="configs/retinanet/retinanet_x101-64x4d_fpn_1x_coco.py"
ALPHA=2.55
STEPS=5
EPSILON=16
ATTACK="gpd"  # "pgd", "fgsm", "cospgd", "none"

# Start job
cd .. 

## Attack
python adv_attack.py --config_file ${CONFIG_FILE} --checkpoint_file ${CHECKPOINT_FILE} --steps $EPSILON --alpha ${ALPHA} --epsilon ${EPSILON}

## Compare performance without attack
python adv_attack.py --config_file ${CONFIG_FILE} --checkpoint_file ${CHECKPOINT_FILE} --steps $EPSILON --alpha ${ALPHA} --epsilon ${EPSILON} --attack 



