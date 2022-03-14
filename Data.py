'''
Vineet Kumar, sioom.ai
'''

from pytorch_lightning import LightningDataModule
import torch
from torch.utils.data import Dataset, RandomSampler, DataLoader
from logging import getLogger
from typing import List, Dict, Tuple, Any
import pandas as pd
from sklearn.model_selection import train_test_split

logg = getLogger(__name__)


class Data(LightningDataModule):
    def __init__(self, batch_size: dict):
        super().__init__()
        for batch_size_key in ('train', 'val', 'test'):
            if batch_size_key not in batch_size or not isinstance(
                    batch_size[batch_size_key],
                    int) or batch_size[batch_size_key] == 0:
                batch_size[batch_size_key] = 1
        self.batch_size = batch_size
        # Trainer('auto_scale_batch_size': True...) requires self.batch_size
        self.batch_size = batch_size['train']

    def prepare_data(self, dataset_path: str) -> None:
        load_dataset(dataset_path)

    def setup(self, dataset_split: Dict[str, int], no_training: bool,
              no_testing: bool):
        for dataset_split_key in ('train', 'val', 'test'):
            if dataset_split_key not in dataset_split or not isinstance(
                    dataset_split[dataset_split_key], int):
                dataset_split[dataset_split_key] = 0
        self.dataset_metadata, train_data, val_data, test_data =\
            _get_trainValTest_data(
                    batch_size=self.batch_size, split=dataset_split)
        if not no_training:
            self.train_data = Data_set(train_data)
            self.valid_data = Data_set(val_data)
        if not no_testing:
            self.test_data = Data_set(test_data)
        if no_training and no_testing:
            logg.debug('No Training and no Testing')

    @staticmethod
    def app_specific_params() -> Tuple[Dict[Any, Any], Dict[Any, Any]]:
        app_specific_init, app_specific = {}, {}
        app_specific_init['num_classes'] = 8
        app_specific_init['imbalanced_classes'] = [
            0.3030, 0.2167, 0.1460, 0.1061, 0.0925, 0.0650, 0.0567, 0.0140
        ]

        app_specific['num_classes'] = 8
        return app_specific_init, app_specific

    def get_dataset_metadata(self) -> Dict[str, Any]:
        return self.dataset_metadata

    def put_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size['train'],
            shuffle=False,
            sampler=RandomSampler(self.train_data),
            batch_sampler=None,
            #num_workers=6,
            num_workers=0,
            collate_fn=self._bert_collater,
            pin_memory=True,
            drop_last=False,
            timeout=0)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.valid_data,
            batch_size=self.batch_size['val'],
            shuffle=False,
            sampler=RandomSampler(self.valid_data),
            batch_sampler=None,
            #num_workers=6,
            num_workers=0,
            collate_fn=self._bert_collater,
            pin_memory=True,
            drop_last=False,
            timeout=0)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size['test'],
            shuffle=False,
            sampler=RandomSampler(self.test_data),
            batch_sampler=None,
            #num_workers=6,
            num_workers=0,
            collate_fn=self._bert_collater,
            pin_memory=True,
            drop_last=False,
            timeout=0)

    def predict_dataloader(self) -> DataLoader:
        pass

    def _bert_collater(self, examples: List[Dict[str, str]]) -> Dict[str, Any]:
        batch_texts, batch_labels = [], []
        for example in examples:
            batch_texts.append(example['sentence'])
            batch_labels.append(example['label'])

        batch_model_inputs = self.tokenizer(text=batch_texts,
                                            padding=True,
                                            truncation=True,
                                            return_tensors='pt',
                                            return_token_type_ids=True,
                                            return_attention_mask=True)

        return {
            'model_inputs': {
                'input_ids':
                batch_model_inputs['input_ids'].type(torch.LongTensor),
                'attention_mask':
                batch_model_inputs['attention_mask'].type(torch.FloatTensor),
                'token_type_ids':
                batch_model_inputs['token_type_ids'].type(torch.LongTensor)
            },
            'labels':
            (torch.LongTensor(batch_labels)).view(len(batch_labels), 1)
        }


class Data_set(Dataset):
    # example = sentence_id plus text plus label
    def __init__(self, examples: List[Dict[str, str]]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        return (self.examples[idx])


def load_dataset(dataset_path: str) -> None:
    df = pd.read_csv(dataset_path, encoding='unicode_escape')

    # remove unneeded data
    df = df.drop(columns=['POS'])
    df = df[df.Tag.isin(["B-art", "I-art", "B-eve", "I-eve", "B-nat", "I-nat"])
            == False]

    # get a list of unique word-labels
    unique_labels = list(df.Tag.unique())
    # convert NaN, of sentence # column, to previous sentence #
    df = df.fillna(method='ffill')

    # group words and tags of a sentence
    # create a new column called "sentence" which groups words of a sentence
    df['sentence'] = df[['Sentence #', 'Word']].groupby(
        ['Sentence #'])['Word'].transform(lambda x: ' '.join(x))
    df = df.drop(columns=['Word'])
    # create a new column called "word_labels" which groups tags of a sentence
    df['word_labels'] = df[['Sentence #', 'Tag']].groupby(
        ['Sentence #'])['Tag'].transform(lambda x: ' '.join(x))
    df = df.drop(columns=['Tag'])
    df = df.drop(columns=['Sentence #'])
    # drop duplicate rows and reset the index
    df = df.drop_duplicates().reset_index(drop=True)

    # convert word_labels to token_labels
    df['token_labels'] = pd.Series(
        map(word_labels_to_token_labels, df['sentence'], df['word_labels']))
    df = df.drop(columns=['word_labels'])


def _get_trainValTest_data(
    batch_size: Dict[str, int], split: Dict[str, int]
) -> Tuple[Dict[str, Any], List[Dict[str, str]], List[Dict[str, str]],
           List[Dict[str, str]]]:
    assert split['train'] + split['val'] + split['test']

    # Split dataset into train, val, test
    if not split['train'] and split['test']:
        # testing a dataset on a checkpoint file; no training
        df_train, df_val, df_test, split['val'], split[
            'test'] = None, None, df, 0, 100
    else:
        df_train, df_temp = train_test_split(df,
                                             shuffle=True,
                                             stratify=None,
                                             train_size=(split['train'] / 100),
                                             random_state=42)
        df_val, df_test = train_test_split(
            df_temp,
            shuffle=True,
            stratify=None,
            test_size=(split['test'] / (split['val'] + split['test'])),
            random_state=42)
        assert len(df) == len(df_train) + len(df_val) + len(df_test)

    dataset_metadata = {
        'batch_size': batch_size,
        'dataset_info': {
            'split': (split['train'], split['val'], split['test']),
            'lengths': (len(df), len(df_train) if df_train is not None else 0,
                        len(df_val) if df_val is not None else 0,
                        len(df_test) if df_test is not None else 0),
        },
    }

    return dataset_metadata, df_train.to_dict(
        'records') if df_train is not None else 0, df_val.to_dict(
            'records') if df_val is not None else 0, df_test.to_dict(
                'records') if df_test is not None else 0


def word_labels_to_token_labels(sentence: str, word_labels: str) -> str:
    for word in sentence:
        pass
    return 1
