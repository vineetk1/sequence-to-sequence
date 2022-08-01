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
        predictions = torch.where(batch['labels'] == -100, batch['labels'],
                                  predictions)
        with self.failed_dlgs_file.open('a') as file:
            prev_failed_dlgTurnIdx = None
            wrapper = textwrap.TextWrapper(width=80,
                                           initial_indent="",
                                           subsequent_indent=19 * " ")
            for failed_dlgTurnIdx, failed_elementIdx in torch.ne(
                    batch['labels'], predictions).nonzero():
                input_tokens = self.tokenizer.convert_ids_to_tokens(
                    batch['model_inputs']['input_ids'][failed_dlgTurnIdx])
                failed_token_label_str = (
                    f"{input_tokens[failed_elementIdx]}, "
                    f"{self.dataset_meta['token-labels -> number:name'][batch['labels'][failed_dlgTurnIdx][failed_elementIdx].item()]}, "
                    f"{ self.dataset_meta['token-labels -> number:name'][predictions[failed_dlgTurnIdx][failed_elementIdx].item()]};\t"
                )
                if failed_dlgTurnIdx == prev_failed_dlgTurnIdx:
                    file.write(failed_token_label_str)
                    continue
                input_tokens_str = "Input tokens = " + " ".join(input_tokens)
                dlg_id_str = f"dlg_id = {batch['ids'][failed_dlgTurnIdx]}"
                actual_str = "True labels = "
                predicted_str = "Predicted labels = "
                for i in torch.arange(batch['labels'].shape[1]):
                    if batch['labels'][failed_dlgTurnIdx][i].item() != -100:
                        actual_str = actual_str + self.dataset_meta[
                            'token-labels -> number:name'][batch['labels'][
                                failed_dlgTurnIdx][i].item()] + " "
                        predicted_str = (
                            predicted_str +
                            self.dataset_meta['token-labels -> number:name']
                            [predictions[failed_dlgTurnIdx][i].item()] + " ")
                failed_token_labels_txt = ('Failed token labels: Input token, '
                                           'True label, Predicted label; ....')
                file.write("\n\n")
                for strng in (dlg_id_str, input_tokens_str, actual_str,
                              predicted_str, failed_token_labels_txt):
                    file.write(wrapper.fill(strng))
                    file.write("\n")
                file.write(failed_token_label_str)
                prev_failed_dlgTurnIdx = failed_dlgTurnIdx

        # collect info to later calculate precision, recall, f1, etc.
        for prediction, actual in zip(predictions.tolist(),
                                      batch['labels'].tolist()):
            y_true = []
            y_pred = []
            for predicted_token_label_num, actual_token_label_num in zip(
                    prediction, actual):
                if actual_token_label_num != -100:
                    y_true.append(
                        self.dataset_meta['token-labels -> number:name']
                        [actual_token_label_num])
                    y_pred.append(
                        self.dataset_meta['token-labels -> number:name']
                        [predicted_token_label_num])
            self.y_true.append(y_true)
            self.y_pred.append(y_pred)
            assert len(y_true) == len(y_pred)

    def _statistics_end(self) -> None:

        # Print
        from sys import stdout
        from contextlib import redirect_stdout
        from pathlib import Path
        stdoutput = Path('/dev/null')
        for out in (stdoutput, self.test_results):
            with out.open("a") as results_file:
                with redirect_stdout(stdout if out ==
                                     stdoutput else results_file):
                    for k, v in self.dataset_meta.items():
                        print(k)
                        print(
                            textwrap.fill(f'{v}',
                                          width=80,
                                          initial_indent=4 * " ",
                                          subsequent_indent=5 * " "))

                    from seqeval.scheme import IOB2
                    from seqeval.metrics import accuracy_score
                    from seqeval.metrics import precision_score
                    from seqeval.metrics import recall_score
                    from seqeval.metrics import f1_score
                    from seqeval.metrics import classification_report
                    print('Classification Report=', end="")
                    print(
                        classification_report(self.y_true,
                                              self.y_pred,
                                              mode='strict',
                                              scheme=IOB2))
                    print('Precision=', end="")
                    print(
                        precision_score(self.y_true,
                                        self.y_pred,
                                        mode='strict',
                                        scheme=IOB2))
                    print('Recall=', end="")
                    print(
                        recall_score(self.y_true,
                                     self.y_pred,
                                     mode='strict',
                                     scheme=IOB2))
                    print('F1=', end="")
                    print(
                        f1_score(self.y_true,
                                 self.y_pred,
                                 mode='strict',
                                 scheme=IOB2))
                    print(
                        f'Accuracy={accuracy_score(self.y_true, self.y_pred): .2f}'
                    )
