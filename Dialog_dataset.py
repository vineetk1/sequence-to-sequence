'''
Vineet Kumar, sioom.ai
'''

from logging import getLogger
from typing import List, Dict, Tuple, Any
import pandas as pd
from sklearn.model_selection import train_test_split
import pathlib
import pickle
import random

logg = getLogger(__name__)


def prepare_dataset(tokenizer, dataset_path: str) -> None:
    dataset_raw_file = pathlib.Path(dataset_path).resolve(strict=True)
    dirName = pathlib.Path(dataset_path).resolve(strict=True).parents[0]
    fileName_noSuffix = pathlib.Path(dataset_path).stem
    dataset_file = dirName.joinpath(f'{fileName_noSuffix}.df')
    dataset_meta_file = dirName.joinpath(f'{fileName_noSuffix}.meta')

    #if dataset_file.exists() and dataset_meta_file.exists() and (
    #        dataset_raw_file.stat().st_mtime < dataset_file.stat().st_mtime):
    #    logg.info(f'Already existing {dataset_file}')
    #    return

    df = pd.read_csv(dataset_path, encoding='unicode_escape')

    # give a name to 'Unnamed: 0' column; remove unneeded columns
    df.rename(columns={'Unnamed: 0': 'dlg_id'}, inplace=True)
    df = df.drop(columns=[
        'title_status', 'lot', 'state', 'country', 'condition', 'vin'
    ])

    def _generate_sentence_wordLabels(
            *entities: List[Any]) -> Tuple[List, List]:
        entities_labels = ['price', 'brand', 'model', 'year', 'mileage', 'color']
        price_sen_seg = [
            'price under {}', 'under {}', 'over {}', '{}', 'less than {}',
            'more than {}', 'between {} and {}'
        ]
        brand_sen_seg = [
            'looking for {}'
            'brand {}', '{}', 'brands {}', '{} and {} or {}'
            '{}', '{} {} {}'
        ]
        model_sen_seg = [
            'looking for {}'
            'model {}', '{}', 'model {}', '{} and {} or {}'
            '{}', '{} {} {}'
        ]
        year_sen_seg = [
            'year under {}', 'under {}', 'over {}', '{}', 'less than {}',
            'more than {}', 'between {} and {}'
        ]
        mileage_sen_seg = [
            'mileage under {}', 'under {}', 'over {}', '{}', 'less than {}',
            'more than {}', 'between {} and {}'
        ]
        color_sen_seg = [
            'looking for {}'
            'color {}', '{}', 'color {}', '{} and {} or {}'
            '{}', '{} {} {}'
        ]
        entities_sentence_segments = {
            'price': price_sen_seg,
            'brand': brand_sen_seg,
            'model': model_sen_seg,
            'year': year_sen_seg,
            'mileage': mileage_sen_seg,
            'color': color_sen_seg,
        }

        # (1a) generate sentence from entities
        entities_idxs = random.sample(range(len(entities)), len(entities))
        sentence = ""
        for entity_idx in entities_idxs:
            sentence += f'{entities[entity_idx]}, '
        '''
        ************************************************************
        sentence.split() or tokenizer fail when sentence has both
        apostrope and quotes without the escape characer; How about using
        three quotes? Test it out; Use try-except
        ************************************************************
        '''
        # (1b) pre-tokenization processing of sentence
        # remove all punctuations except $#@%{} Also split sentence along
        # white spaces into words
        sentence_split_into_words = sentence.translate(
            str.maketrans('', '', '!"&\'()*+,-./:;<=>?[\\]^_`|~')).split()

        # (1c) generate word-labels from words in sentence
        # ****NOTE: IMP** word-labels are not created from sentence but from entities*******
        words_labels = []
        for entity_label_idx in rndm_smpl:
            words = entities[entity_label_idx]
            # remove punctuations here ???
            if not isinstance(words, str):
                words_labels.append(f'B-{entity_labels[entity_label_idx]}')
            else:
                first_word = True
                for word in words.split():
                    if first_word:
                        first_word = False
                        words_labels.append(
                            f'B-{entity_labels[entity_label_idx]}')
                    else:
                        words_labels.append(
                            f'I-{entity_labels[entity_label_idx]}')
        if len(sentence_split_into_words) != len(words_labels):
            strng = (
                f'\n{len(sentence_split_into_words)} != {len(words_labels)}, '
                f'{sentence_split_into_words}, {words_labels}')
            logg.critical(strng)
            exit()
        return sentence_split_into_words, words_labels

    # (1) create "sentence_split_into_words_word_labels" column with
    # sentence_split_into_words and word_labels
    df["sentence_split_into_words_word_labels"] = pd.Series(
        map(_generate_sentence_wordLabels, df["price"], df["brand"],
            df["model"], df["year"], df["mileage"], df["color"]))
    df = df.drop(
        columns=["price", "brand", "model", "year", "mileage", "color"])

    # (2) from "sentence_split_into_words_word_labels" column, create two
    # columns with sentence_split_into_words and word_labels
    for i, col in enumerate(["sentence_split_into_words", "word_labels"]):
        df[col] = df["sentence_split_into_words_word_labels"].apply(
            lambda sentence_split_into_words_word_labels:
            sentence_split_into_words_word_labels[i])
    df = df.drop(columns=["sentence_split_into_words_word_labels"])

    def _word_labels_to_token_labels_names(sentence: List,
                                           words_labels: List) -> List:
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
        if len(sentence) != len(words_labels):
            strng = (f'\n{len(sentence)} != {len(words_labels)}, '
                     f'{sentence}, {words_labels}')
            logg.critical(strng)
            exit()
        assert words_labels
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
            # token of a word
            word_idx = words_idxs[token_idx]
            word_label = words_labels[
                word_idx] if word_idx is not None else None
            prev_word_idx = words_idxs[token_idx - 1]
            if word_idx is None:
                # special-token (e.g. CLS, SEP, PAD) gets a label of -100
                tokens_labels.append("-100")
            elif word_idx != prev_word_idx:
                # first token of a word gets the label of that word;
                # no indexing error because words_idxs[token_idx-1=0] is
                # always None and this case is handled by "if statement"
                tokens_labels.append(word_label)
            else:  # word_idx == prev_word_idx
                # if not first token then a remaining token of that word
                if word_label[0] == 'O':
                    tokens_labels.append('O')
                else:  # word_label[0] == 'B' or word_label[0] == 'I'
                    tokens_labels.append(f'I{word_label[1:]}')

        assert len(tokens_labels) == len(tokens)
        return tokens_labels

    # (3) convert word_labels to token_labels_names
    df['token_labels'] = pd.Series(
        map(_word_labels_to_token_labels_names,
            df['sentence_split_into_words'], df['word_labels']))
    df = df.drop(columns=['word_labels'])

    # (4) create a list of unique BIO2 labels
    unique_bio2_label_names = []
    for token_labels in df['token_labels']:
        for token_label in token_labels:
            if (token_label not in unique_bio2_label_names) and (token_label !=
                                                                 "-100"):
                unique_bio2_label_names.append(token_label)

    def _token_labels_names2numbers(token_labels_names: List) -> List:
        return [
            unique_bio2_label_names.index(token_label_name)
            if token_label_name != '-100' else -100
            for token_label_name in token_labels_names
        ]

    # (5) convert token-label names to numbers
    df['token_labels'] = pd.Series(
        map(_token_labels_names2numbers, df['token_labels']))

    # (6) save data-frame to disk
    df.to_pickle(dataset_file)
    with dataset_meta_file.open('wb') as dmF:
        pickle.dump(unique_bio2_label_names,
                    dmF,
                    protocol=pickle.HIGHEST_PROTOCOL)


def split_dataset(
    dataset_path: str, split: Dict[str, int]
) -> Tuple[Dict[str, Any], List[List[List[Any]]], List[List[List[Any]]],
           List[List[List[Any]]]]:
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
        unique_bio2_label_names = pickle.load(dmF)

    # Split dataset into train, val, test; Note: parts of dialogs may get split
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

    train_data, val_data, test_data = (
        df_train.values.tolist() if df_train is not None else None,
        df_val.values.tolist() if df_val is not None else None,
        df_test.values.tolist() if df_test is not None else None)

    # create meta-data for the datasets
    from collections import Counter

    def token_labels_count(dataset):
        if dataset is None:
            return None
        count = Counter()
        for example in dataset:
            for token_label in example[2]:
                count[token_label] += 1
        return dict(count)

    trainValTest_tokenLabels_count = [
        token_labels_count(dataset)
        for dataset in (train_data, val_data, test_data)
    ]

    trainValTest_tokenLabels_unseen = [
        set(range(len(unique_bio2_label_names))).difference(
            set(sub_dataset_tokenLabels_count.keys()))
        for sub_dataset_tokenLabels_count in (trainValTest_tokenLabels_count)
    ]

    dataset_metadata = {
        'dataset splits': {
            'train': split['train'],
            'val': split['val'],
            'test': split['test']
        },
        'dataset lengths': {
            'original': len(df),
            'train': len(df_train) if df_train is not None else 0,
            'val': len(df_val) if df_val is not None else 0,
            'test': len(df_test) if df_test is not None else 0
        },
        'token-labels -> number:name':
        {k: v
         for k, v in enumerate(unique_bio2_label_names)},
        'train token-labels -> number:count':
        trainValTest_tokenLabels_count[0],
        'val token-labels -> number:count': trainValTest_tokenLabels_count[1],
        'test token-labels -> number:count': trainValTest_tokenLabels_count[2],
        'train unseen token-labels': trainValTest_tokenLabels_unseen[0],
        'val unseen token-labels': trainValTest_tokenLabels_unseen[1],
        'test unseen token-labels': trainValTest_tokenLabels_unseen[2]
    }

    return dataset_metadata, train_data, val_data, test_data
