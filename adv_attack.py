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
from mmengine.evaluator import Evaluator
import logging

DATA_BATCH = Optional[Union[dict, tuple, list]]


def pgd_attack(data_batch, runner):
    assert isinstance(data_batch, dict)
    images = data_batch.get("inputs")[0].clone().detach().to("cuda")
    assert isinstance(images, torch.Tensor)
    adv_images = images.clone().float().detach().to("cuda")

    if TARGETED:
        raise NotImplementedError

    if RANDOM_START:
        raise NotImplementedError

    for _ in range(STEPS):
        adv_images.requires_grad = True
        data_batch["inputs"][0] = adv_images

        # TODO: do we need to preprocess in every cycle?
        data_batch_prepro = runner.model.data_preprocessor(data_batch, training=False)
        losses = runner.model(**data_batch_prepro, mode="loss")
        cost, _ = runner.model.parse_losses(losses)

        grad = torch.autograd.grad(
            cost, adv_images, retain_graph=False, create_graph=False
        )
        grad = torch.cat(grad, dim=0)

        adv_images = adv_images.detach() + ALPHA * grad.sign()
        delta = torch.clamp(adv_images - images, min=-EPSILON, max=EPSILON)
        adv_images = torch.clamp(images + delta, min=0, max=255).detach()

    adv_images.requires_grad = False
    data_batch["inputs"][0] = adv_images
    data_batch_prepro = runner.model.data_preprocessor(data_batch, training=False)
    return data_batch_prepro


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
        "--epsilon", type=int, default=8, help="Epsilon value for the attack"
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
    ATTACK = not args.no_attack
    CHECKPOINT_FILE = args.checkpoint_file
    CONFIG_FILE = args.config_file

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

                metrics = self.evaluator.evaluate(len(self.dataloader.dataset))
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
                    outputs = self.runner.model.test_step(data_batch_prepro)

                self.evaluator.process(data_samples=outputs, data_batch=data_batch)
                self.runner.call_hook(
                    "after_val_iter",
                    batch_idx=idx,
                    data_batch=data_batch,
                    outputs=outputs,
                )

    cfg = Config.fromfile(CONFIG_FILE)
    cfg.work_dir = "./work_dirs/"
    cfg.load_from = CHECKPOINT_FILE

    runner = Runner.from_cfg(cfg)
    runner.val()
