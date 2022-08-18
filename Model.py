'''
Vineet Kumar, sioom.ai
'''

from pytorch_lightning import LightningModule
import torch
from logging import getLogger
from typing import Dict, List, Any, Union, Tuple
import pathlib
from importlib import import_module
import copy
import textwrap

logg = getLogger(__name__)


class Model(LightningModule):
    def __init__(self, model_init: dict):
        super().__init__()
        # save parameters for future use of "loading a model from
        # checkpoint"
        self.save_hyperparameters()
        # Trainer('auto_lr_find': True,...) requires self.lr

        if model_init['model'] == "bertEncoderDecoder" and model_init[
                'tokenizer_type'] == "bert" and model_init[
                    "model_type"] == "bert-base-uncased":
            from transformers import EncoderDecoderModel

            self.bertEncDecModel = (
                EncoderDecoderModel.from_encoder_decoder_pretrained(
                    model_init["model_type"], model_init["model_type"])
            )  # initialize Bert2Bert from pre-trained checkpoints

            from transformers import BertTokenizerFast
            tokenizer = BertTokenizerFast.from_pretrained(
                model_init['model_type'])
            self.bertEncDecModel.config.decoder_start_token_id = (
                tokenizer.cls_token_id)
            self.bertEncDecModel.config.pad_token_id = tokenizer.pad_token_id
            self.bertEncDecModel.config.vocab_size = (
                self.bertEncDecModel.config.decoder.vocab_size)
        else:
            strng = (f"unknown model={model_init['model']} or "
                     f"unknown tokenizer_type={model_init['tokenizer_type']} "
                     f"or unknown model_type={model_init['model_type']}")
            logg.critical(strng)
            exit()

    def params(self, optz_sched_params: Dict[str, Any],
               batch_size: Dict[str, int]) -> None:
        self.batch_size = batch_size  # needed to turn off lightning warning
        self.optz_sched_params = optz_sched_params
        # Trainer('auto_lr_find': True...) requires self.lr
        self.lr = optz_sched_params['optz_params']['lr'] if (
            'optz_params' in optz_sched_params) and (
                'lr' in optz_sched_params['optz_params']) else None

    def forward(self):
        logg.debug('')

    def training_step(self, batch: Dict[str, Any],
                      batch_idx: int) -> torch.Tensor:
        tr_loss, _ = self._run_model(batch)
        # logger=True => TensorBoard; x-axis is always in steps=batches
        self.log('train_loss',
                 tr_loss,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True,
                 batch_size=self.batch_size['train'],
                 logger=False)
        return tr_loss

    def training_epoch_end(
            self, training_step_outputs: List[Dict[str,
                                                   torch.Tensor]]) -> None:
        tr_avg_loss = torch.stack([x['loss']
                                   for x in training_step_outputs]).mean()
        # on TensorBoard, want to see x-axis in epochs (not steps=batches)
        self.logger.experiment.add_scalar('train_loss_epoch', tr_avg_loss,
                                          self.current_epoch)

    def validation_step(self, batch: Dict[str, Any],
                        batch_idx: int) -> torch.Tensor:
        v_loss, _ = self._run_model(batch)
        self.log(
            'val_loss',
            v_loss,
            on_step=False,
            on_epoch=True,  # checkpoint-callback monitors epoch
            # val_loss, so on_epoch Must be True
            prog_bar=True,
            batch_size=self.batch_size['val'],
            logger=False)
        return v_loss

    def validation_epoch_end(
            self, val_step_outputs: List[Union[torch.Tensor,
                                               Dict[str, Any]]]) -> None:
        v_avg_loss = torch.stack(val_step_outputs).mean()
        # on TensorBoard, want to see x-axis in epochs (not steps=batches)
        self.logger.experiment.add_scalar('val_loss_epoch', v_avg_loss,
                                          self.current_epoch)

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        ts_loss, logits = self._run_model(batch)
        # checkpoint-callback monitors epoch val_loss, so on_epoch=True
        self.log('test_loss',
                 ts_loss,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=False,
                 batch_size=self.batch_size['test'],
                 logger=True)
        if self.statistics:
            self._statistics_step(predictions=torch.argmax(logits, dim=-1),
                                  batch=batch)
        return ts_loss

    def test_epoch_end(
            self, test_step_outputs: List[Union[torch.Tensor,
                                                Dict[str, Any]]]) -> None:
        if self.statistics:
            self._statistics_end()

    def _run_model(self,
                   batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs = self.bertEncDecModel(**batch['model_inputs'],
                                       labels=batch['labels'])
        return outputs.loss, outputs.logits

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
                       dirPath: pathlib.Path, tokenizer) -> None:
        self.failed_dlgs_file = dirPath.joinpath('dialogs_failed.txt')
        self.failed_dlgs_file.touch()
        if self.failed_dlgs_file.stat().st_size:
            with self.failed_dlgs_file.open('a') as file:
                file.write('\n\n****resume from checkpoint****\n')
        self.test_results = dirPath.joinpath('test-results.txt')
        self.test_results.touch()
        if self.test_results.stat().st_size:
            with self.test_results.open('a') as file:
                file.write('\n\n****resume from checkpoint****\n')

        self.statistics = True
        self.dataset_meta = dataset_meta
        self.dirPath = dirPath
        self.tokenizer = tokenizer
        self.y_true = []
        self.y_pred = []

    def _statistics_step(self, predictions: torch.Tensor,
                         batch: Dict[str, Any]) -> None:
        # write to file the info about failed turns of dialogs
        input = self.tokenizer.batch_decode(batch['model_inputs']['input_ids'],
                                            skip_special_tokens=True)
        label = self.tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)
        output = self.tokenizer.batch_decode(predictions,
                                             skip_special_tokens=True)
        assert output == label
        pass

    def _statistics_end(self) -> None:
        pass
