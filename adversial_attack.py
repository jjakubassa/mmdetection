# from mmengine.config import Config
import torch
from mmdet.apis import init_detector
from mmdet.evaluation import evaluator
from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.registry import OPTIMIZERS
from torchvision.ops import sigmoid_focal_loss
import torch.optim as optim
from itertools import islice
import cospgd


# TODO: Is from Cospgd Github  -> how to change?


def fgsm_attack(self, perturbed_image, data_grad, orig_image):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    if self.targeted:
        sign_data_grad *= -1
    perturbed_image = perturbed_image.detach() + self.alpha * sign_data_grad
    # Adding clipping to maintain [0,1] range
    if self.norm == "inf":
        delta = torch.clamp(
            perturbed_image - orig_image, min=-1 * self.epsilon, max=self.epsilon
        )
    elif self.norm == "two":
        delta = perturbed_image - orig_image
        delta_norms = torch.norm(delta.view(self.batch_size, -1), p=2, dim=1)
        factor = self.epsilon / delta_norms
        factor = torch.min(factor, torch.ones_like(delta_norms))
        delta = delta * factor.view(-1, 1, 1, 1)
    perturbed_image = torch.clamp(orig_image + delta, 0, 1)
    # Return the perturbed image
    return perturbed_image


# Setup
CHECKPOINT_FILE = "checkpoints/retinanet_x101_64x4d_fpn_1x_coco_20200130-366f5af1.pth"
CONFIG_FILE = "configs/retinanet/retinanet_x101-64x4d_fpn_1x_coco.py"

cfg = Config.fromfile(CONFIG_FILE)
cfg.work_dir = "./work_dirs/"  # needed for runner
model = init_detector(CONFIG_FILE, CHECKPOINT_FILE, device="cuda:0")
runner_original = Runner.from_cfg(cfg)
runner_pertubed = Runner.from_cfg(cfg)
data_loader = runner_original.build_dataloader(cfg.test_dataloader)
# optimizer = OPTIMIZERS.build(cfg.optimizer)

original_results = []
perturbed_results = []

# only use part of data
num_minibatches = 5
debug_data = islice(data_loader, num_minibatches)

attack_steps = 10
attack = "cospgd"
num_classes = 80

EPSILON = 8
ALPHA = 0.01 * 255
TARGETED = False
NORM = "two"

# Inference
for i, data_batch in enumerate(debug_data):
    # don't normalize in transform steps
    images = data_batch["inputs"]
    images = torch.stack(images, dim=0).float()
    images_pertubed = images.clone().detach().to("cuda").requires_grad_(True)

    # labels = data_batch["data_samples"][0].gt_instances.labels
    # bboxes = data_batch["data_samples"][0].gt_instances.bboxes

    # images_pertubed = cospgd.functions.init_l2(
    #     images=images_pertubed, epsilon=EPSILON, clamp_min=0, clamp_max=255
    # )

    data_preprocessed = model.data_preprocessor(data_batch, training=False)

    for _ in range(attack_steps):
        losses = model(**data_preprocessed, mode="loss")
        data_grad = data_preprocessed["in"]

        # Collect the element-wise sign of the data gradient
        sign_data_grad = data_grad.sign()
        # Create the perturbed image by adjusting each pixel of the input image
        if TARGETED:
            sign_data_grad *= -1
        perturbed_image = perturbed_image.detach() + alpha * sign_data_grad
        # Adding clipping to maintain [0,1] range
        if NORM == "inf":
            delta = torch.clamp(
                perturbed_image - orig_image, min=-1 * epsilon, max=epsilon
            )
        elif NORM == "two":
            delta = perturbed_image - orig_image
            delta_norms = torch.norm(delta.view(batch_size, -1), p=2, dim=1)
            factor = EPSILON / delta_norms
            factor = torch.min(factor, torch.ones_like(delta_norms))
            delta = delta * factor.view(-1, 1, 1, 1)
        perturbed_image = torch.clamp(orig_image + delta, 0, 1)

        # Return the perturbed image
        return perturbed_image

        # P = model(images_pertubed)
        # P = model(**data_preprocessed, mode="predict")
        # losses = model(**data_preprocessed, mode="loss")  # avoid second forward pass

        # loss = sigmoid_focal_loss(inputs=images_pertubed, targets=labels)

        # if isinstance(data, dict):
        #     loss = model(**data, mode="loss")
        # elif isinstance(data, (list, tuple)):
        #     loss = model(*data, mode="loss")

        # L_cos = cospgd.functions.cospgd_scale(
        #     predictions=P,
        #     labels=labels,
        #     loss=losses,
        #     num_classes=num_classes,
        #     targeted=False,
        #     one_hot=False,
        # )

        # optimizer.zero_grad()
    #     losses.backward()

    #     images_pertubed = cospgd.functions.step_l2(
    #         perturbed_image=images_pertubed,
    #         epsilon=EPSILON,
    #         data_grad=images.grad,
    #         orig_image=images,
    #         alpha=ALPHA,
    #         targeted=False,
    #         clamp_min=0,
    #         clamp_max=255,
    #     )

    # # Run inference on perturbed images
    # with torch.no_grad():
    #     perturbed_result = model(images_pertubed)
    # perturbed_results.append(perturbed_result)

    #     cospgd.functions.

    #     # init noised image
    #     # add loss, scale pixelwise for cospgd, one-hot = false
    #     perturbed_image = apply_adversarial_attack(image, model) #use pertubed image of previous image
    # runner_original.val_evaluator
    # original_results.append(original_result)

    # # Apply adversarial attack to the images
    # # perturbed_images = apply_adversarial_attack(image, model)
    # # perturbed_data = dict(inputs=[perturbed_images], img_metas=data['img_metas'])

    # # Run inference on perturbed images
    # with torch.no_grad():
    #     perturbed_result = model(perturbed_image)
    # perturbed_results.append(perturbed_result)

# Evaluation
# evaluator = runner.build_evaluator(cfg.val_evaluator)
# eval_results_original = evaluator.compute_metrics(original_results)
# evaluator.reset()
# eval_results_perturbed = evaluator.compute_metrics(perturbed_results)
# print(f'Original images evaluation results:\n{eval_results_original}')
# print(f'Perturbed images evaluation results:\n{eval_results_perturbed}')
