# from mmengine.config import Config
import torch
from mmdet.apis import init_detector
from mmdet.datasets import CocoDataset
from mmengine.config import Config
from mmengine import ConfigDict
from mmengine.runner import Runner

def apply_adversarial_attack(image, model):
    perturbed_image = image.clone()  
    #TODO: do a proper attack
    return perturbed_image

# Setup
CHECKPOINT_FILE="checkpoints/retinanet_x101_64x4d_fpn_1x_coco_20200130-366f5af1.pth"
CONFIG_FILE="configs/retinanet/retinanet_x101-64x4d_fpn_1x_coco.py"

cfg = Config.fromfile(CONFIG_FILE)
cfg.work_dir = "./work_dirs/" # needed for runner
model = init_detector(CONFIG_FILE, CHECKPOINT_FILE, device='cuda:0')
runner = Runner.from_cfg(cfg)
data_loader = runner.build_dataloader(cfg.test_dataloader)

original_results = []
perturbed_results = []

# Inference
for i, data in enumerate(data_loader):
    # Run inference on original images
    with torch.no_grad():
        original_result = model(**data)
    original_results.append(original_result)

    # Apply adversarial attack to the images
    perturbed_images = apply_adversarial_attack(data['img'][0], model)
    perturbed_data = dict(img=[perturbed_images], img_metas=data['img_metas'])
    
    # Run inference on perturbed images
    with torch.no_grad():
        perturbed_result = model(**perturbed_data)
    perturbed_results.append(perturbed_result)

# Evaluation
evaluator = runner.build_evaluator(cfg.evaluation.metric)
eval_results_original = evaluator.evaluate(original_results)
evaluator.reset()
eval_results_perturbed = evaluator.evaluate(perturbed_results)
print(f'Original images evaluation results:\n{eval_results_original}')
print(f'Perturbed images evaluation results:\n{eval_results_perturbed}')

