import torch
from mmengine.hooks import Hook
from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.registry import HOOKS
from typing import Dict, List, Optional, Sequence, Tuple, Union
from mmengine.logging import print_log
from mmengine.registry import LOOPS
from mmengine.runner.amp import autocast
from mmengine.runner.base_loop import BaseLoop
from torch.utils.data import DataLoader
from mmengine.evaluator import Evaluator
import logging

DATA_BATCH = Optional[Union[dict, tuple, list]]


# https://mmengine.readthedocs.io/en/latest/design/hook.html
@HOOKS.register_module()
class CheckInvalidLossHook(Hook):
    def __init__(self) -> None:
        super().__init__()
        print("123456789")

    def before_val_epoch(self, runner: Runner) -> None:
        self._before_epoch(runner, mode="val")
        print("123456789")

    def before_val_iter(
        self, runner, batch_idx: int, data_batch: DATA_BATCH = None
    ) -> None:
        self._before_iter(
            runner, batch_idx=batch_idx, data_batch=data_batch, mode="val"
        )
        # print(batch_idx)

    def after_val_iter(
        self,
        runner,
        batch_idx: int,
        data_batch: DATA_BATCH = None,
        outputs: Optional[Sequence] = None,
    ) -> None:
        self._after_iter(
            runner,
            batch_idx=batch_idx,
            data_batch=data_batch,
            outputs=outputs,
            mode="val",
        )


LOOPS.module_dict.pop("ValLoop")


@LOOPS.register_module()
class ValLoop(BaseLoop):
    """Loop for validation.

    Args:
        runner (Runner): A reference of runner.
        dataloader (Dataloader or dict): A dataloader object or a dict to
            build a dataloader.
        evaluator (Evaluator or dict or list): Used for computing metrics.
        fp16 (bool): Whether to enable fp16 validation. Defaults to
            False.
    """

    def __init__(
        self,
        runner,
        dataloader: Union[DataLoader, Dict],
        evaluator: Union[Evaluator, Dict, List],
        fp16: bool = False,
    ) -> None:
        super().__init__(runner, dataloader)

        if isinstance(evaluator, (dict, list)):
            self.evaluator = runner.build_evaluator(evaluator)  # type: ignore
        else:
            assert isinstance(evaluator, Evaluator), (
                "evaluator must be one of dict, list or Evaluator instance, "
                f"but got {type(evaluator)}."
            )
            self.evaluator = evaluator  # type: ignore
        if hasattr(self.dataloader.dataset, "metainfo"):
            self.evaluator.dataset_meta = self.dataloader.dataset.metainfo
            self.runner.visualizer.dataset_meta = self.dataloader.dataset.metainfo
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
        """Launch validation."""
        self.runner.call_hook("before_val")
        self.runner.call_hook("before_val_epoch")
        self.runner.model.eval()
        for idx, data_batch in enumerate(self.dataloader):
            self.run_iter(idx, data_batch)

        # compute metrics
        metrics = self.evaluator.evaluate(len(self.dataloader.dataset))
        self.runner.call_hook("after_val_epoch", metrics=metrics)
        self.runner.call_hook("after_val")
        return metrics

    @torch.no_grad()
    def run_iter(self, idx, data_batch: Sequence[dict]):
        """Iterate one mini-batch.

        Args:
            data_batch (Sequence[dict]): Batch of data
                from dataloader.
        """
        # TODO: implement attack here
        self.runner.call_hook("before_val_iter", batch_idx=idx, data_batch=data_batch)
        # outputs should be sequence of BaseDataElement
        with autocast(enabled=self.fp16):
            outputs = self.runner.model.val_step(data_batch)
        self.evaluator.process(data_samples=outputs, data_batch=data_batch)
        self.runner.call_hook(
            "after_val_iter", batch_idx=idx, data_batch=data_batch, outputs=outputs
        )


CHECKPOINT_FILE = "checkpoints/retinanet_x101_64x4d_fpn_1x_coco_20200130-366f5af1.pth"
CONFIG_FILE = "configs/retinanet/retinanet_x101-64x4d_fpn_1x_coco.py"

custom_hooks = [dict(type="CheckInvalidLossHook")]
custom_hook = dict(type="CheckInvalidLossHook")

cfg = Config.fromfile(CONFIG_FILE)
cfg.work_dir = "./work_dirs/"
cfg.custom_hooks = custom_hooks
cfg.load_from = CHECKPOINT_FILE

runner = Runner.from_cfg(cfg)
runner.register_hook(CheckInvalidLossHook(), priority="NORMAL")
runner.val()
