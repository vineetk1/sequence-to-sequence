'''
Vineet Kumar, sioom.ai
'''

from logging import getLogger
from typing import List, Dict, Tuple, Any
import pandas as pd
from sklearn.model_selection import train_test_split
import pathlib

logg = getLogger(__name__)


class Dialog_dataset():
    def __init__(self):
        self.irrelevant_punctuations = [',', '.']

    def load_dataset(self, tokenizer, dataset_path: str) -> None:
        self.tokenizer = tokenizer
        dirName = pathlib.Path(dataset_path).resolve(strict=True).parents[0]
        fileName_noSuffix = pathlib.Path(dataset_path).stem
        filePath = dirName.joinpath(f'{fileName_noSuffix}.df')
        '''
        if filePath.exists():
            logg.info(f'Loaded already existing {filePath}')
            self.df = pd.read_pickle(filePath)
            return
        '''
        self.df = pd.read_csv(dataset_path, encoding='unicode_escape')

        # remove unneeded data
        self.df = self.df.drop(columns=['POS'])
        self.df = self.df[self.df.Tag.isin(
            ["B-art", "I-art", "B-eve", "I-eve", "B-nat", "I-nat"]) == False]

        # get a list of unique word-labels
        self.unique_labels = list(self.df.Tag.unique())
        # convert NaN, of sentence # column, to previous sentence #
        self.df = self.df.fillna(method='ffill')

        # group words and tags of a sentence
        # create new column called "sentence" which groups words of a sentence
        self.df['sentence'] = self.df[['Sentence #', 'Word']].groupby(
            ['Sentence #'])['Word'].transform(lambda x: ' '.join(x))
        self.df = self.df.drop(columns=['Word'])
        # create new column called "word_labels" which groups tags of sentence
        self.df['word_labels'] = self.df[['Sentence #', 'Tag']].groupby(
            ['Sentence #'])['Tag'].transform(lambda x: ' '.join(x))
        self.df = self.df.drop(columns=['Tag'])
        self.df = self.df.drop(columns=['Sentence #'])

        # ***************check for problems in the dataset********************
        # no duplicate sentences; first occurrence is not duplicate
        #assert not self.df.duplicated(subset=['sentence'], keep='first').any()
        # total number of duplicate sentences
        #assert self.df.duplicated(subset=['sentence'], keep='first').sum()
        #self.df['sentence'].duplicated().sum()
        # show all occurrences of duplicate sentences
        #self.df[self.df.duplicated(subset=['sentence'], keep=False)]

        # drop duplicate rows and reset the index
        #self.df = self.df.drop_duplicates().reset_index(drop=True)
        self.df.drop_duplicates(subset=['sentence'],
                                keep='first',
                                inplace=True)
        self.df.reset_index(drop=True, inplace=True)

        # pre-tokenization processing of sentences
        self.df['sentence'] = pd.Series(
            map(self._pre_tokenization, self.df['sentence']))

        # convert word_labels to token_labels
        self.df['token_labels'] = pd.Series(
            map(self._word_labels_to_token_labels, self.df['sentence'],
                self.df['word_labels']))
        self.df = self.df.drop(columns=['word_labels'])

        # remove all rows that have an empty list for token_label

        self.df.to_pickle(filePath)

    def _pre_tokenization(self, sentence: str) -> List:
        '''
        ************************************************************
        sentence.split() or tokenizer fail when sentence has both
        apostrope and quotes without the escape characer; Test it out; Use
        try-except
        ************************************************************
        '''
        # split sentence along white spaces into words
        sentence = sentence.split()
        assert sentence
        return sentence

    def _word_labels_to_token_labels(self, sentence: List,
                                     words_labls: str) -> List:
        '''
        ************************************************************
        sentence.split() or tokenizer fail when sentence has both
        apostrope and quotes without the escape characer; Test it out; Use
        try-except
        ************************************************************
        '''
        words_labels = words_labls.split()
        if len(sentence) != len(words_labels):
            print(f'\n{sentence}\n')
            return []
        assert words_labels
        assert len(sentence) == len(words_labels)
        tokens = self.tokenizer.convert_ids_to_tokens(
            self.tokenizer(sentence, is_split_into_words=True)['input_ids'])
        words_idxs = self.tokenizer(sentence,
                                    is_split_into_words=True).word_ids()
        assert len(tokens) == len(words_idxs)

        tokens_labels = []
        for token_idx in range(len(words_idxs)):
            # token = tokens[token_idx]
            # word_idx = words_idxs[token_idx]
            # word = sentence[word_idx]
            # word_label = words_labels[word_idx]
            # Assume no indexing problem in the following two cases:
            # (1) words_idxs[token_idx] != words_idxs[token_idx-1] => first
            # token of a word
            # (2) words_idxs[token_idx] != words_idxs[token_idx+1] => last
            # token of  a word
            if words_idxs[token_idx] is None:
                # special-token (e.g. CLS, SEP, PAD) gets a label of -100
                tokens_labels.append("-100")
            elif words_idxs[token_idx] != words_idxs[token_idx - 1]:
                # no indexing error because words_idxs[token_idx-1=0] is
                # always None and this case is handled by "if statement"
                # first token of a word gets the label of that word
                tokens_labels.append(words_labels[words_idxs[token_idx]])
            else:
                # if not first token then remaining tokens get label of "-100"
                tokens_labels.append("-100")

        assert len(tokens_labels) == len(tokens)
        return tokens_labels

    def split_dataset(
        self, split: Dict[str, int]
    ) -> Tuple[Dict[str, Any], List[Dict[str, str]], List[Dict[str, str]],
               List[Dict[str, str]]]:
        assert split['train'] + split['val'] + split['test'] == 100

        # Split dataset into train, val, test
        if not split['train'] and split['test']:
            # testing a dataset on a checkpoint file; no training
            df_train, df_val, df_test, split['val'], split[
                'test'] = None, None, self.df, 0, 100
        else:
            df_train, df_temp = train_test_split(self.df,
                                                 shuffle=True,
                                                 stratify=None,
                                                 train_size=(split['train'] /
                                                             100),
                                                 random_state=42)
            df_val, df_test = train_test_split(
                df_temp,
                shuffle=True,
                stratify=None,
                test_size=(split['test'] / (split['val'] + split['test'])),
                random_state=42)
            assert len(self.df) == len(df_train) + len(df_val) + len(df_test)

        dataset_metadata = {
            'dataset_info': {
                'split': (split['train'], split['val'], split['test']),
                'lengths':
                (len(self.df), len(df_train) if df_train is not None else 0,
                 len(df_val) if df_val is not None else 0,
                 len(df_test) if df_test is not None else 0),
            },
        }

        return dataset_metadata, df_train.to_dict(
            'records') if df_train is not None else 0, df_val.to_dict(
                'records') if df_val is not None else 0, df_test.to_dict(
                    'records') if df_test is not None else 0
