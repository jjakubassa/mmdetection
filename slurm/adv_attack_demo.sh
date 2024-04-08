#!/bin/bash
#SBATCH --partition=gpu_4
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
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
STEPS=1
EPSILON=8
ATTACK="pgd"  # "pgd", "fgsm", "cospgd", "none"

cd .. # expects mmdetection/slurm to be the working directory

echo "#########  START ATTACK ###########"
echo "CHECKPOINT_FILE="${CHECKPOINT_FILE}
echo "CONFIG_FILE="${CONFIG_FILE}
echo "ALPHA="${ALPHA}
echo "STEPS="${STEPS}
echo "EPSILON="${EPSILON}
echo "ATTACK="${ATTACK}

python adv_attack.py --config_file ${CONFIG_FILE} --checkpoint_file ${CHECKPOINT_FILE} --steps $EPSILON --alpha ${ALPHA} --epsilon ${EPSILON} --attack ${ATTACK}

echo "#########  END ATTACK ###########"


# echo "#########  START VALIDATION WITHOUT ATTACK ###########"
# ATTACK="none" 
# echo "CHECKPOINT_FILE="${CHECKPOINT_FILE}
# echo "CONFIG_FILE="${CONFIG_FILE}
# echo "ALPHA="${ALPHA}
# echo "STEPS="${STEPS}
# echo "EPSILON="${EPSILON}
# echo "ATTACK="${ATTACK}

# python adv_attack.py --config_file ${CONFIG_FILE} --checkpoint_file ${CHECKPOINT_FILE} --steps $EPSILON --alpha ${ALPHA} --epsilon ${EPSILON} --attack ${ATTACK}

# echo "#########  END VALIDATION WITHOUT ATTACK ###########"


echo "#########  START ATTACK WITH EPSILON = 0 ###########"
ATTACK="pgd" 
EPSILON=0
echo "CHECKPOINT_FILE=" ${CHECKPOINT_FILE}
echo "CONFIG_FILE=" ${CONFIG_FILE}
echo "ALPHA=" ${ALPHA}
echo "STEPS=" ${STEPS}
echo "EPSILON=" ${EPSILON}
echo "ATTACK=" ${ATTACK}

python adv_attack.py --config_file ${CONFIG_FILE} --checkpoint_file ${CHECKPOINT_FILE} --steps $EPSILON --alpha ${ALPHA} --epsilon ${EPSILON} --attack ${ATTACK}

echo "#########  END ATTACK WITH EPSILON = 0 ###########"


