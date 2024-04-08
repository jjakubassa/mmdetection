import argparse
import torch
from mmengine.hooks import Hook
from mmengine.config import Config
from mmengine.runner import Runner
from typing import Dict, List, Optional, Sequence, Union
from mmengine.logging import print_log
from mmengine.registry import LOOPS
from mmengine.runner.base_loop import BaseLoop
from torch.utils.data import DataLoader
from torchvision import transforms
from mmengine.evaluator import Evaluator
import logging
from mmengine.runner.amp import autocast

DATA_BATCH = Optional[Union[dict, tuple, list]]


def pgd_attack(data_batch: dict, runner):
    data_batch_prepro = runner.model.data_preprocessor(data_batch, training=False)

    images = data_batch_prepro.get("inputs")[0].clone().detach().to("cuda")
    images_clone = images.clone().float().detach()
    assert isinstance(images, torch.Tensor)

    # from retinanet_r50_fpn.py. TODO: read from runner.model
    mean = [123.675, 116.28, 103.53]
    std = [58.395, 57.12, 57.375]

    images = denorm(images, mean, std)
    adv_images = images.clone().float().detach().to("cuda")

    if TARGETED:
        raise NotImplementedError

    if RANDOM_START:
        raise NotImplementedError

    for _ in range(STEPS):
        adv_images.requires_grad = True
        data_batch_prepro["inputs"][0] = adv_images
        losses = runner.model(**data_batch_prepro, mode="loss")
        cost, _ = runner.model.parse_losses(losses)

        grad = torch.autograd.grad(
            cost, adv_images, retain_graph=False, create_graph=False
        )
        grad = torch.cat(grad, dim=0)

        adv_images = adv_images.detach() + ALPHA * grad.sign()
        # assert torch.allclose(ALPHA * grad.sign(), torch.zeros_like(grad.sign()))
        delta = torch.clamp(adv_images - images, min=-EPSILON, max=EPSILON)
        # assert torch.allclose(delta, torch.zeros_like(delta))
        adv_images = torch.clamp(images + delta, min=0, max=255).detach()

    adv_images.requires_grad = False
    transforms.Normalize(mean, std, inplace=True)(adv_images)
    data_batch_prepro["inputs"][0] = adv_images

    if ALPHA == 0 or EPSILON == 0 or STEPS == 0:  # for debug
        assert torch.allclose(images_clone, adv_images)
    else:
        assert not torch.allclose(images_clone, adv_images)

    return data_batch_prepro


def fgsm_attack(runner, perturbed_image, data_grad, orig_image):
    data_batch_prepro = runner.model.data_preprocessor(data_batch, training=False)

    images = data_batch_prepro.get("inputs")[0].clone().detach().to("cuda")
    images_clone = images.clone().float().detach()
    assert isinstance(images, torch.Tensor)

    # from retinanet_r50_fpn.py. TODO: read from runner.model
    mean = [123.675, 116.28, 103.53]
    std = [58.395, 57.12, 57.375]

    images = denorm(images, mean, std)
    adv_images = images.clone().float().detach().to("cuda")

    # Get gradient
    adv_images.requires_grad = True
    data_batch_prepro["inputs"][0] = adv_images
    losses = runner.model(**data_batch_prepro, mode="loss")
    cost, _ = runner.model.parse_losses(losses)
    grad = torch.autograd.grad(cost, adv_images, retain_graph=False, create_graph=False)
    grad = torch.cat(grad, dim=0)

    # Collect the element-wise sign of the data gradient
    sign_data_grad = grad.sign()

    # Create the perturbed image by adjusting each pixel of the input image
    if TARGETED:
        sign_data_grad *= -1
    perturbed_image = perturbed_image.detach() + ALPHA * sign_data_grad

    # Adding clipping to maintain [0,1] range
    if NORM == "inf":
        delta = torch.clamp(perturbed_image - orig_image, min=-1 * EPSILON, max=EPSILON)
    elif NORM == "two":
        delta = perturbed_image - orig_image
        batch_size = orig_image.shape()[0]
        delta_norms = torch.norm(delta.view(batch_size, -1), p=2, dim=1)
        factor = EPSILON / delta_norms
        factor = torch.min(factor, torch.ones_like(delta_norms))
        delta = delta * factor.view(-1, 1, 1, 1)
    perturbed_image = torch.clamp(orig_image + delta, 0, 1)
    # Return the perturbed image
    return perturbed_image


# restores the tensors to their original scale
# https://pytorch.org/tutorials/beginner/fgsm_tutorial.html
def denorm(batch, mean=[0.1307], std=[0.3081]):
    """
    Convert a batch of tensors to their original scale.

    Args:
        batch (torch.Tensor): Batch of normalized tensors.
        mean (torch.Tensor or list): Mean used for normalization.
        std (torch.Tensor or list): Standard deviation used for normalization.

    Returns:
        torch.Tensor: batch of tensors without normalization applied to them.
    """
    if isinstance(mean, list):
        mean = torch.tensor(mean).to("cuda")
    if isinstance(std, list):
        std = torch.tensor(std).to("cuda")

    return batch * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Argument Parser for the variables")
    parser.add_argument(
        "--targeted", action="store_true", help="Enable targeted attack"
    )
    parser.add_argument(
        "--random_start", action="store_true", help="Enable random start for attack"
    )
    parser.add_argument(
        "--steps", type=int, default=5, help="Number of steps for the attack"
    )
    parser.add_argument(
        "--alpha", type=float, default=0.01 * 255, help="Alpha value for the attack"
    )
    parser.add_argument(
        "--epsilon", type=float, default=8, help="Epsilon value for the attack"
    )
    parser.add_argument(
        "--attack",
        type=str,
        default="pgd",
        choices=["pgd", "fgsm", "cospgd", "none"],
        help="Type of attack (default: pgd)",
    )
    parser.add_argument(
        "--checkpoint_file",
        type=str,
        default="checkpoints/retinanet_x101_64x4d_fpn_1x_coco_20200130-366f5af1.pth",
        help="Path to the checkpoint file",
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default="configs/retinanet/retinanet_x101-64x4d_fpn_1x_coco.py",
        help="Path to the config file",
    )

    args = parser.parse_args()

    TARGETED = args.targeted
    RANDOM_START = args.random_start
    STEPS = args.steps
    ALPHA = args.alpha
    EPSILON = args.epsilon
    ATTACK = args.attack
    CHECKPOINT_FILE = args.checkpoint_file
    CONFIG_FILE = args.config_file
    NORM = "inf"

    if ATTACK != "none":
        LOOPS.module_dict.pop("ValLoop")

        @LOOPS.register_module()
        class ValLoop(BaseLoop):
            def __init__(
                self,
                runner,
                dataloader: Union[DataLoader, Dict],
                evaluator: Union[Evaluator, Dict, List],
                fp16: bool = False,
            ) -> None:
                super().__init__(runner, dataloader)

                if isinstance(evaluator, (dict, list)):
                    self.evaluator = runner.build_evaluator(evaluator)
                else:
                    assert isinstance(evaluator, Evaluator), (
                        "evaluator must be one of dict, list or Evaluator instance, "
                        f"but got {type(evaluator)}."
                    )
                    self.evaluator = evaluator
                if hasattr(self.dataloader.dataset, "metainfo"):
                    self.evaluator.dataset_meta = getattr(
                        self.dataloader.dataset, "metainfo"
                    )
                    self.runner.visualizer.dataset_meta = getattr(
                        self.dataloader.dataset, "metainfo"
                    )
                else:
                    print_log(
                        f"Dataset {self.dataloader.dataset.__class__.__name__} has no "
                        "metainfo. ``dataset_meta`` in evaluator, metric and "
                        "visualizer will be None.",
                        logger="current",
                        level=logging.WARNING,
                    )
                self.fp16 = fp16

            def run(self) -> dict:
                self.runner.call_hook("before_val")
                self.runner.call_hook("before_val_epoch")
                self.runner.model.eval()
                for idx, data_batch in enumerate(self.dataloader):
                    self.run_iter(idx, data_batch)

                metrics = self.evaluator.evaluate(len(self.dataloader.dataset))  # type: ignore
                self.runner.call_hook("after_val_epoch", metrics=metrics)
                self.runner.call_hook("after_val")
                return metrics

            def run_iter(self, idx, data_batch: Sequence[dict]):
                self.runner.call_hook(
                    "before_val_iter", batch_idx=idx, data_batch=data_batch
                )

                if ATTACK == "pgd":
                    data_batch_prepro = pgd_attack(data_batch, self.runner)
                elif ATTACK == "cospgd":
                    raise NotImplementedError
                elif ATTACK == "fgsm":
                    raise NotImplementedError
                else:
                    raise ValueError

                with torch.no_grad():
                    outputs = self.runner.model(**data_batch_prepro, mode="predict")

                self.evaluator.process(
                    data_samples=outputs, data_batch=data_batch_prepro
                )
                self.runner.call_hook(
                    "after_val_iter",
                    batch_idx=idx,
                    data_batch=data_batch_prepro,
                    outputs=outputs,
                )

    cfg = Config.fromfile(CONFIG_FILE)
    cfg.work_dir = "./work_dirs/"
    cfg.load_from = CHECKPOINT_FILE

    runner = Runner.from_cfg(cfg)
    runner.val()
