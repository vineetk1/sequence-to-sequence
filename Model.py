'''
Vineet Kumar, sioom.ai
'''

from pytorch_lightning import LightningModule
import torch
from logging import getLogger
from sys import exit
from typing import Dict, List, Any, Union, Tuple
import pathlib
from importlib import import_module
import copy
import textwrap

logg = getLogger(__name__)


class Model(LightningModule):
    def __init__(self, model_init: dict, num_classes: int):
        super().__init__()
        # save parameters for future use of "loading a model from
        # checkpoint" or "resuming training from checkpoint"
        self.save_hyperparameters()
        # Trainer('auto_lr_find': True,...) requires self.lr

        if model_init['model'] == "bert":
            from transformers import BertForTokenClassification
            self.model = BertForTokenClassification.from_pretrained(
                model_init['model_type'], num_labels=num_classes)

    def params(self, optz_sched_params: Dict[str, Any]) -> None:
        self.optz_sched_params = optz_sched_params
        # Trainer('auto_lr_find': True...) requires self.lr
        self.lr = optz_sched_params['optz_params']['lr'] if (
            'optz_params' in optz_sched_params) and (
                'lr' in optz_sched_params['optz_params']) else None

    def forward(self):
        logg.debug('')

    def training_step(self, batch: Dict[str, Any],
                      batch_idx: int) -> torch.Tensor:
        loss, _ = self._run_model(batch)
        # logger=True => TensorBoard; x-axis is always in steps=batches
        self.log('train_loss',
                 loss,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True,
                 logger=False)
        return loss

    def training_epoch_end(
            self, training_step_outputs: List[Dict[str,
                                                   torch.Tensor]]) -> None:
        avg_loss = torch.stack([x['loss']
                                for x in training_step_outputs]).mean()
        # on TensorBoard, want to see x-axis in epochs (not steps=batches)
        self.logger.experiment.add_scalar('train_loss_epoch', avg_loss,
                                          self.current_epoch)

    def validation_step(self, batch: Dict[str, Any],
                        batch_idx: int) -> torch.Tensor:
        loss, _ = self._run_model(batch)
        # checkpoint-callback monitors epoch val_loss, so on_epoch=True
        self.log('val_loss',
                 loss,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True,
                 logger=False)
        return loss

    def validation_epoch_end(
            self, val_step_outputs: List[Union[torch.Tensor,
                                               Dict[str, Any]]]) -> None:
        avg_loss = torch.stack(val_step_outputs).mean()
        # on TensorBoard, want to see x-axis in epochs (not steps=batches)
        self.logger.experiment.add_scalar('val_loss_epoch', avg_loss,
                                          self.current_epoch)

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        loss, logits = self._run_model(batch)
        # checkpoint-callback monitors epoch val_loss, so on_epoch=True
        self.log('test_loss_step',
                 loss,
                 on_step=True,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)
        if self.statistics:
            self._statistics_step(actuals=batch['labels'],
                                  predictions=torch.argmax(logits, dim=-1))
        return loss

    def test_epoch_end(
            self, test_step_outputs: List[Union[torch.Tensor,
                                                Dict[str, Any]]]) -> None:
        avg_loss = torch.stack(test_step_outputs).mean()
        # on TensorBoard, want to see x-axis in epochs (not steps=batches)
        self.logger.experiment.add_scalar('test_loss_epoch', avg_loss,
                                          self.current_epoch)
        if self.statistics:
            self._statistics_end()

    def _run_model(self,
                   batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs = self.model(**batch['model_inputs'], labels=batch['labels'])
        loss = outputs[0]
        logits = outputs[1]
        return loss, logits

    def configure_optimizers(self):
        opt_sch_params = copy.deepcopy(self.optz_sched_params)
        _ = opt_sch_params['optz_params'].pop('lr', None)
        if 'optz' in opt_sch_params and opt_sch_params['optz']:
            if 'optz_params' in opt_sch_params and opt_sch_params[
                    'optz_params']:
                if self.lr is not None:
                    optimizer = getattr(import_module('torch.optim'),
                                        opt_sch_params['optz'])(
                                            self.parameters(),
                                            lr=self.lr,
                                            **opt_sch_params['optz_params'])
                else:
                    optimizer = getattr(import_module('torch.optim'),
                                        opt_sch_params['optz'])(
                                            self.parameters(),
                                            **opt_sch_params['optz_params'])
            else:
                if self.lr is not None:
                    optimizer = getattr(import_module('torch.optim'),
                                        opt_sch_params['optz'])(
                                            self.parameters(), lr=self.lr)
                else:
                    optimizer = getattr(import_module('torch.optim'),
                                        opt_sch_params['optz'])(
                                            self.parameters())

        if 'lr_sched' in opt_sch_params and opt_sch_params['lr_sched']:
            if 'lr_sched_params' in opt_sch_params and opt_sch_params[
                    'lr_sched_params']:
                scheduler = getattr(import_module('torch.optim.lr_scheduler'),
                                    opt_sch_params['lr_sched'])(
                                        optimizer=optimizer,
                                        **opt_sch_params['lr_sched_params'])
            else:
                scheduler = getattr(
                    import_module('torch.optim.lr_scheduler'),
                    opt_sch_params['lr_sched'])(optimizer=optimizer)

        # If scheduler is specified then optimizer must be specified
        # If Trainer('resume_from_checkpoint',...), then optimizer and
        # scheduler may not be specified
        if 'optimizer' in locals() and 'scheduler' in locals():
            return {
                'optimizer':
                optimizer,
                'lr_scheduler':
                scheduler,
                'monitor':
                'val_loss'
                if opt_sch_params['lr_sched'] == 'ReduceLROnPlateau' else None
            }
        elif 'optimizer' in locals():
            return optimizer

    def clear_statistics(self) -> None:
        self.statistics = False

    def set_statistics(self, dataset_meta: Dict[str, Any],
                       dirPath: pathlib.Path) -> None:
        self.statistics = True
        self.dataset_meta = dataset_meta
        self.dirPath = dirPath
        num_classes = len(dataset_meta['token-labels -> number:name'])
        self.confusion_matrix = torch.zeros(num_classes,
                                            num_classes,
                                            dtype=torch.int64)

    def _statistics_step(self,
                         predictions: torch.Tensor,
                         actuals: torch.Tensor,
                         example_ids: List[str] = None) -> None:
        for prediction, actual in zip(predictions, actuals):
            for predicted_token_label, actual_token_label in zip(
                    prediction, actual):
                if actual_token_label != -100:
                    self.confusion_matrix[predicted_token_label,
                                          actual_token_label] += 1

    def _statistics_end(self) -> None:
        # Verify
        for i, token_label_count in enumerate(self.confusion_matrix.sum(0)):
            assert self.dataset_meta['test token-labels -> number:count'][
                i] == token_label_count.item()

        # Calculate
        epsilon = 1E-9
        precision = self.confusion_matrix.diag() / (
            self.confusion_matrix.sum(1) + epsilon)
        recall = self.confusion_matrix.diag() / (self.confusion_matrix.sum(0) +
                                                 epsilon)
        f1 = (2 * precision * recall) / (precision + recall + epsilon)
        f1_avg = f1.sum() / f1.shape[0]  # macro average
        f1_wgt = ((f1 * self.confusion_matrix.sum(0)).sum()
                  ) / self.confusion_matrix.sum()

        # Print
        from sys import stdout
        from contextlib import redirect_stdout
        from pathlib import Path
        stdoutput = Path('/dev/null')
        test_results = self.dirPath.joinpath('test-results.txt')
        test_results.touch(exist_ok=False)
        for out in (stdoutput, test_results):
            with out.open("w") as results_file:
                with redirect_stdout(stdout if out ==
                                     stdoutput else results_file):
                    for k, v in self.dataset_meta.items():
                        print(k)
                        print(
                            textwrap.fill(f'{v}',
                                          width=80,
                                          initial_indent=4 * " ",
                                          subsequent_indent=5 * " "))

                    print(
                        '\nConfusion matrix (prediction (rows) vs. actual (columns))='
                    )
                    print(f'{self.confusion_matrix}')

                    print('precision, recall, f1')
                    for x in (precision, recall, f1):
                        print(x)

                    print(f'average f1 = {f1_avg: .4f}')
                    print(f'weighted f1 = {f1_wgt: .4f}')
