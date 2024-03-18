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

module load devel/cuda/11.8

# python tools/misc/download_dataset.py --dataset-name=coco2017 --unzip
# wget -nc -P checkpoints/ https://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_x101_64x4d_fpn_1x_coco/retinanet_x101_64x4d_fpn_1x_coco_20200130-366f5af1.pth

cd .. 
CHECKPOINT_FILE="checkpoints/retinanet_x101_64x4d_fpn_1x_coco_20200130-366f5af1.pth"
CONFIG_FILE="configs/retinanet/retinanet_x101-64x4d_fpn_1x_coco.py"

python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE}
