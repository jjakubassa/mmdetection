# from mmengine.config import Config
import torch
from mmdet.apis import init_detector
from mmdet.datasets import CocoDataset
from mmengine.config import Config
from mmengine import ConfigDict
from mmengine.runner import Runner

# TODO: Is from Cospgd Github  -> how to change?
def fgsm_attack(self, perturbed_image, data_grad, orig_image):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image        
    if self.targeted:
        sign_data_grad *= -1
    perturbed_image = perturbed_image.detach() + self.alpha*sign_data_grad
    # Adding clipping to maintain [0,1] range
    if self.norm == 'inf':
        delta = torch.clamp(perturbed_image - orig_image, min = -1*self.epsilon, max=self.epsilon)
    elif self.norm == 'two':
        delta = perturbed_image - orig_image
        delta_norms = torch.norm(delta.view(self.batch_size, -1), p=2, dim=1)
        factor = self.epsilon / delta_norms
        factor = torch.min(factor, torch.ones_like(delta_norms))
        delta = delta * factor.view(-1, 1, 1, 1)
    perturbed_image = torch.clamp(orig_image + delta, 0, 1)
    # Return the perturbed image
    return perturbed_image

def apply_adversarial_attack(image, model, attack_type="fgsm"):
    
    #TODO: do a proper attack
    if attack_type == "fgsm":
        perturbed_image = image.clone().detach().to("cuda")
        perturbed_image.requires_grad = True
        # TODO: look in evaluation file how do they calculate loss?
    elif attack_type == "pgd":
        raise NotImplementedError(f"Attack type {attack_type} not implemented yet")
    elif attack_type == "cospgd":
        raise NotImplementedError(f"Attack type {attack_type} not implemented yet")
    else:
        raise NotImplementedError(f"Attack type {attack_type} not implemented")
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

