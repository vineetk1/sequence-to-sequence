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
import pandas as pd
from collections import Counter

logg = getLogger(__name__)


class Model(LightningModule):

    def __init__(self, model_init: dict, tokenizer):
        super().__init__()
        # save parameters for future use of "loading a model from
        # checkpoint"
        self.save_hyperparameters()
        # Trainer('auto_lr_find': True,...) requires self.lr

        if model_init['model'] == "t5EncoderDecoder" and model_init[
                "model_type"] == "google/t5-v1_1-base":
            from transformers import T5ForConditionalGeneration
            self.t5Model = T5ForConditionalGeneration.from_pretrained(
                model_init["model_type"])
            self.t5Model.resize_token_embeddings(len(tokenizer))
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

    def validation_epoch_end(self,
                             val_step_outputs: List[torch.Tensor]) -> None:
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
                 prog_bar=True,
                 batch_size=self.batch_size['test'],
                 logger=True)
        return ts_loss

    def test_epoch_end(self, test_step_outputs: List[torch.Tensor]) -> None:
        pass

    def _run_model(self,
                   batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs = self.t5Model(**batch['model_inputs'], labels=batch['labels'])
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

    def prepare_for_predict(self, dataset_meta: Dict[str, Any],
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

        self.df = pd.read_pickle(dataset_meta['dataset_panda'])
        self.tokenizer = tokenizer

    def on_predict_start(self) -> None:
        self.cntr = Counter()
        self.max_turn_num = 0

    def predict_step(self, batch: Dict[str, Any], batch_idx: int) -> Any:
        batch_output = self.t5Model.generate(
            # parameter = None => replace with self.config.parameter
            max_length=self.tokenizer.max_model_input_sizes['t5-base'],
            min_length=1,
            do_sample=False,
            early_stopping=None,
            num_beams=6,
            #num_beams=None,
            temperature=None,
            top_k=None,
            top_p=None,
            repetition_penalty=1.0,
            bad_words_ids=None,
            length_penalty=1.0,
            no_repeat_ngram_size=None,
            num_return_sequences=1,
            decoder_start_token_id=None,
            use_cache=None,
            # num_beam_groups=None,  # this parameter is not in called program
            # diversity_penalty=None,  # this parametr is not in called program
            prefix_allowed_tokens_fn=None,
            **batch['model_inputs'])

        if self.tokenizer.unk_token_id in batch_output:
            print(self.tokenizer.batch_decode(batch_output))
            print(self.tokenizer.batch_decode(batch['labels']))
            logg.critical("UNK detected in batch_output")
            exit()

        for batch_idx in range(len(batch['ids'])):
            self.max_turn_num = max(self.max_turn_num,
                                    batch["ids"][batch_idx][1])
            self.cntr['num_turns'] += 1
            # (label != output) is True even when there are extra spaces in
            # either label or output; but self._diff(label, output) ignores
            # extra spaces
            if ((label := self.tokenizer.decode(batch['labels'][batch_idx],
                                                skip_special_tokens=True)) !=
                (output := self.tokenizer.decode(
                    batch_output[batch_idx, 1:],
                    skip_special_tokens=True))) and self._diff(label, output):
                # dlg-turn failed
                self.cntr['num_turns_fail'] += 1
                self.cntr[f'num_trn{batch["ids"][batch_idx][1]}_fail'] += 1
                self._failed_dlg_to_file(input=self.tokenizer.decode(
                    batch['model_inputs']['input_ids'][batch_idx],
                    skip_special_tokens=True),
                                         label=label,
                                         output=output,
                                         id=batch['ids'][batch_idx])
            else:
                self.cntr['num_turns_pass'] += 1
                self.cntr[f'num_trn{batch["ids"][batch_idx][1]}_pass'] += 1

    def _failed_dlg_to_file(self, input: torch.Tensor, label: torch.Tensor,
                            output: torch.Tensor, id: List[int]) -> None:
        wrapper = textwrap.TextWrapper(width=80,
                                       initial_indent="",
                                       subsequent_indent=11 * " ")
        pd_dlg_trn = self.df.loc[(self.df['dlg_ids'] == id[0])
                                 & (self.df['turns'] == id[1])]
        dlg_id_str = f'dlg_id:   {id[0]}, {id[1]}'
        pd_input = (pd_dlg_trn["in_seq2seq_frames"] + " " +
                    pd_dlg_trn["sentences"]).item()
        pd_input_str = f'pd input: {pd_input}'
        md_input_str = f'md input: {input}'
        diff_input = self._diff(pd_input, input)
        diff_input_str = f'diff input: {diff_input}'
        pd_label = pd_dlg_trn["out_seq2seq_frames"].item()
        pd_label_str = f'pd label: {pd_label}'
        md_label_str = f'md label: {label}'
        diff_label = self._diff(pd_label, label)
        diff_label_str = f'diff label: {diff_label}'
        predict_str = f'predict:  {output}'
        diff_mdLabel_output = self._diff(label, output)
        diff_mdLabel_output_str = f'diff md_label predict: {diff_mdLabel_output}'
        with self.failed_dlgs_file.open('a') as file:
            for strng in (dlg_id_str, pd_input_str, md_input_str,
                          diff_input_str, pd_label_str, md_label_str,
                          diff_label_str, predict_str,
                          diff_mdLabel_output_str):
                file.write(wrapper.fill(strng))
                file.write("\n")
            file.write("\n\n")

    def _diff(self, str1: str, str2: str) -> str:
        chars_to_replace = {'[': ' ', ']': ' ', ',': ' '}
        str1 = str1.translate(str.maketrans(chars_to_replace)).split()
        str2 = str2.translate(str.maketrans(chars_to_replace)).split()
        strng = ""
        if str1 != str2:
            for elem1, elem2 in zip(str1, str2):
                if elem1 != elem2:
                    cnt1, cnt2 = self._get_counts(elem1, elem2)
                    strng += f'({elem1}({cnt1}), {elem2}({cnt2})) '
        return strng

    def _get_counts(self, elem1: str, elem2: str) -> Tuple[int, int]:
        cnt1, cnt2 = 0, 0
        try:
            elem1 = str(int(float(elem1)))
            elem2 = str(int(float(elem2)))
            # string is a number
            for sentence in self.df['out_seq2seq_frames']:
                if elem1 in sentence:
                    # multiple occurrences of elem1 are counted as 1 occurrence
                    cnt1 += 1
                if elem2 in sentence:
                    cnt2 += 1
        except ValueError:
            # string is not a number
            for out_frame in self.df['out_seq2seq_frames']:
                if elem1 in out_frame:
                    cnt1 += 1
                if elem2 in out_frame:
                    cnt2 += 1
        return cnt1, cnt2

    def on_predict_end(self) -> None:
        from sys import stdout
        from contextlib import redirect_stdout
        stdoutput = pathlib.Path('/dev/null')
        for out in (stdoutput, self.test_results):
            with out.open("a") as results_file:
                with redirect_stdout(stdout if out ==
                                     stdoutput else results_file):
                    print(f"# of turns = {self.cntr['num_turns']}")
                    strng = (
                        f"(#, %) of turns passed = "
                        f"{self.cntr['num_turns_pass']}, "
                        f"{((self.cntr['num_turns_pass'] * 100) / self.cntr['num_turns']): .2f}"
                    )
                    print(strng)
                    strng = (
                        f"(#, %) of turns failed = "
                        f"{self.cntr['num_turns_fail']}, "
                        f"{((self.cntr['num_turns_fail'] * 100) / self.cntr['num_turns']): .2f}"
                    )
                    print(strng)
                    for turn_num in range(self.max_turn_num + 1):
                        num_turns_pass = self.cntr[f'num_trn{turn_num}_pass']
                        num_turns_fail = self.cntr[f'num_trn{turn_num}_fail']
                        num_turns_total = num_turns_pass + num_turns_fail
                        strng = (
                            f"(#, %) of turn {turn_num} that passed = "
                            f"{num_turns_pass}, "
                            f"{((num_turns_pass * 100) / num_turns_total): .2f}"
                        )
                        print(strng)
                        strng = (
                            f"(#, %) of turn {turn_num} that failed = "
                            f"{num_turns_fail}, "
                            f"{((num_turns_fail * 100) / num_turns_total): .2f}"
                        )
                        print(strng)
