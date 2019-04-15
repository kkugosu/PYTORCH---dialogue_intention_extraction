import numpy as np
import re
import json
import torch.utils.data
from torch.utils.data import DataLoader
import io,os
from torch import nn
from gensim.models import word2vec
from torch.nn.utils.rnn import pad_sequence
from torch import optim
from torch.autograd import Variable
from torchtext import data


def batchload(dataset, repeat, batchsize, data_seq):
    """

    load data as much as batch

    Args:

        dataset:
            data to load
        repeat:
            True if repeat load data
        batchsize:
            batchsize
        data_seq:
            order of data to load

    Yields:

        Batch data

    :param dataset:
    :param repeat:
    :param batchsize:
    :param data_seq:
    :return:

    """

    while True:
        i = batchsize

        while i <= len(data_seq):
            print(i)
            batch = []
            batch_seq = 0
            batchnum = data_seq[i - batchsize:i]

            while batch_seq < batchsize:
                batch.append(dataset[batchnum[batch_seq]])
                batch_seq = batch_seq + 1
            # print("batchnum = ",i)
            yield batch
            i = i + batchsize

        if not repeat:
            break


class Example(object):
    @classmethod
    def fromdict(cls, data, fields):
        ex = cls()
        for key, vals in fields.items():
            if key not in data:
                raise ValueError("Specified key {} was not found in "
                                 "the input data".format(key))
            if vals is not None:
                if not isinstance(vals, list):
                    vals = [vals]
                for val in vals:
                    name, field = val
                    setattr(ex, name, field.preprocess(data[key]))
        return ex


class Dataset(torch.utils.data.Dataset):
    sort_key = None

    def __init__(self, examples, fields, filter_pred=None):
        self.examples = examples

        self.fields = dict(fields)

        # Unpack field tuples
        for n, f in list(self.fields.items()):
            if isinstance(n, tuple):
                self.fields.update(zip(n, f))
                del self.fields[n]
        self.pp = tuple(d for d in self.examples if d is not None)

    @classmethod
    def splits(cls, path=None, root='.data', train=None, **kwargs):

        train_data = cls(os.path.join(path, train), **kwargs)
        # print(train_data.examples) #여기엔 field example둘다 들어있음
        # print(tuple(d for d in train_data if d is not None)) #여기엔 example만 나열된 튜플이됨
        return tuple(d for d in train_data if d is not None)

    def split(self, split_ratio=0.7, stratified=False, strata_field='label',
              random_state=None):

        train_ratio, test_ratio, val_ratio = check_split_ratio(split_ratio)
        # For the permutations

        rnd = RandomShuffler(random_state)
        if not stratified:
            train_data, test_data, val_data = rationed_split(self.examples, train_ratio,
                                                             test_ratio, val_ratio, rnd)
        else:
            if strata_field not in self.fields:
                raise ValueError("Invalid field name for strata_field {}"
                                 .format(strata_field))
            strata = stratify(self.examples, strata_field)
            train_data, test_data, val_data = [], [], []
            for group in strata:
                # Stratify each group and add together the indices.
                group_train, group_test, group_val = rationed_split(group, train_ratio,
                                                                    test_ratio, val_ratio,
                                                                    rnd)
                train_data += group_train
                test_data += group_test
                val_data += group_val

        splits = tuple(Dataset(d, self.fields)
                       for d in (train_data, val_data, test_data) if d)

        # In case the parent sort key isn't none
        if self.sort_key:
            for subset in splits:
                subset.sort_key = self.sort_key
        return splits

    def __getitem__(self, i):
        return self.examples[i]

    def __len__(self):
        try:
            return len(self.examples)
        except TypeError:
            return 2 ** 32

    def __iter__(self):
        for x in self.examples:
            yield x

    def __getattr__(self, attr):
        if attr in self.fields:
            for x in self.examples:
                yield getattr(x, attr)


class MyTabularDataset(Dataset):

    def __init__(self, path,  fields,  **kwargs):

        with open(path, encoding="utf8") as f:
            for line in f:
                examples = [Example.fromdict(per_data, fields) for per_data in json.loads(line)]

        if isinstance(fields, dict):
            fields, field_dict = [], fields
            for field in field_dict.values():
                if isinstance(field, list):
                    fields.extend(field)
                else:
                    fields.append(field)

        super(MyTabularDataset, self).__init__(examples, fields, **kwargs)