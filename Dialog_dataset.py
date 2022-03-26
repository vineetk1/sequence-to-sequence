'''
Vineet Kumar, sioom.ai
'''

from logging import getLogger
from typing import List, Dict, Tuple, Any
import pandas as pd
from sklearn.model_selection import train_test_split
import pathlib
import pickle

logg = getLogger(__name__)


def prepare_dataset(tokenizer, dataset_path: str) -> None:
    dataset_raw_file = pathlib.Path(dataset_path).resolve(strict=True)
    dirName = pathlib.Path(dataset_path).resolve(strict=True).parents[0]
    fileName_noSuffix = pathlib.Path(dataset_path).stem
    dataset_file = dirName.joinpath(f'{fileName_noSuffix}.df')
    dataset_meta_file = dirName.joinpath(f'{fileName_noSuffix}.meta')

    if dataset_file.exists() and dataset_meta_file.exists() and (
            dataset_raw_file.stat().st_mtime < dataset_file.stat().st_mtime):
        logg.info(f'Already existing {dataset_file}')
        return

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
    # create new column called "sentence" which groups words of a sentence
    df['sentence'] = df[['Sentence #', 'Word']].groupby(
        ['Sentence #'])['Word'].transform(lambda x: ' '.join(x))
    df = df.drop(columns=['Word'])
    # create new column called "word_labels" which groups tags of sentence
    df['word_labels'] = df[['Sentence #', 'Tag']].groupby(
        ['Sentence #'])['Tag'].transform(lambda x: ' '.join(x))
    df = df.drop(columns=['Tag'])
    df = df.drop(columns=['Sentence #'])

    # ***************check for problems in the dataset********************
    # no duplicate sentences; first occurrence is not duplicate
    #assert not df.duplicated(subset=['sentence'], keep='first').any()
    # total number of duplicate sentences
    #assert df.duplicated(subset=['sentence'], keep='first').sum()
    # show all occurrences of duplicate sentences
    #df[df.duplicated(subset=['sentence'], keep=False)]

    # drop duplicate rows and reset the index
    df.drop_duplicates(subset=['sentence'], keep='first', inplace=True)
    df.reset_index(drop=True, inplace=True)

    # pre-tokenization processing of sentences
    def _pre_tokenization(sentence: str) -> List:
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

    df['sentence'] = pd.Series(map(_pre_tokenization, df['sentence']))

    # convert word_labels to token_labels_names
    def _word_labels_to_token_labels_names(sentence: List,
                                           words_labls: str) -> List:
        '''
        ************************************************************
        (1) sentence.split() or tokenizer fail when sentence has both
        apostrope and quotes without the escape characer; Test it out; Use
        try-except
        (2) truncation if sentence (not history) is longer than max length;
            note: only sentence gets labels and not the history
        (3) make sure beginning of history is truncated; how about doing a
            summary of history
        ************************************************************
        '''
        words_labels = words_labls.split()
        if len(sentence) != len(words_labels):
            print(f'\n{sentence}\n')
            return None
        assert words_labels
        assert len(sentence) == len(words_labels)
        tokens = tokenizer.convert_ids_to_tokens(
            tokenizer(sentence, is_split_into_words=True)['input_ids'])
        words_idxs = tokenizer(sentence, is_split_into_words=True).word_ids()
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

    df['token_labels'] = pd.Series(
        map(_word_labels_to_token_labels_names, df['sentence'],
            df['word_labels']))
    df = df.drop(columns=['word_labels'])

    # remove all rows that have None
    df = df.dropna()
    df.reset_index(drop=True, inplace=True)

    # convert token-label names to numbers
    def _token_labels_names2numbers(token_labels_names: List) -> List:
        return [
            unique_labels.index(token_label_name)
            if token_label_name != '-100' else -100
            for token_label_name in token_labels_names
        ]

    df['token_labels'] = pd.Series(
        map(_token_labels_names2numbers, df['token_labels']))

    df.to_pickle(dataset_file)
    with dataset_meta_file.open('wb') as dmF:
        pickle.dump(unique_labels, dmF, protocol=pickle.HIGHEST_PROTOCOL)


def split_dataset(
    dataset_path: str, split: Dict[str, int]
) -> Tuple[Dict[str, Any], List[Dict[str, str]], List[Dict[str, str]],
           List[Dict[str, str]]]:
    assert split['train'] + split['val'] + split['test'] == 100

    # retrieve data files
    dirName = pathlib.Path(dataset_path).resolve(strict=True).parents[0]
    fileName_noSuffix = pathlib.Path(dataset_path).stem
    dataset_file = dirName.joinpath(f'{fileName_noSuffix}.df')
    dataset_meta_file = dirName.joinpath(f'{fileName_noSuffix}.meta')
    if (not dataset_file.exists()) or (not dataset_meta_file.exists()):
        strng = ('Either one or both of following files do not exist: '
                 '{dataset_file}, {dataset_meta_file}')
        logg.critical(strng)
        exit()
    df = pd.read_pickle(dataset_file)
    with dataset_meta_file.open('rb') as dmF:
        unique_labels = pickle.load(dmF)

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
        'dataset_info': {
            'split': (split['train'], split['val'], split['test']),
            'lengths': (len(df), len(df_train) if df_train is not None else 0,
                        len(df_val) if df_val is not None else 0,
                        len(df_test) if df_test is not None else 0),
            'num_classes':
            len(unique_labels)
        },
    }

    return dataset_metadata, df_train.to_dict(
        'records') if df_train is not None else 0, df_val.to_dict(
            'records') if df_val is not None else 0, df_test.to_dict(
                'records') if df_test is not None else 0
