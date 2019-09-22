from __future__ import absolute_import, division, print_function

import numpy as np
import pandas as pd


def read_data_and_process(filename, sep="\t"):
    col_names = ["user", "item", "rate", "st"]
    df = pd.read_csv(filename, sep=sep, header=None, names=col_names, engine='python')
    df["user"] -= 1
    df["item"] -= 1
    for col in ("user", "item"):
        df[col] = df[col].astype(np.int32)
    df["rate"] = df["rate"].astype(np.float32)
    return df


class ShuffleDataIterator(object):
    """
    generate batches data randomly
    """
    # initialization
    def __init__(self, inputs, batch_size=10):
        self.inputs = inputs
        self.batch_size = batch_size
        self.num_cols = len(self.inputs)
        self.len = len(self.inputs[0])
        self.inputs = np.transpose(np.vstack([np.array(self.inputs[i])
                                              for i in range(self.num_cols)]))

    # total sample
    def __len__(self):
        return self.len

    def __iter__(self):
        return self

    # get the next batch of data
    def __next__(self):
        return self.__next__()

    # generate the subscripts of batch_size randomly and get the sample
    def next(self):
        ids = np.random.randint(0, self.len, (self.batch_size,))
        out = self.inputs[ids, :]
        return [out[:, i] for i in range(self.num_cols)]


class OneEpochDataIterator(ShuffleDataIterator):
    """
    generate a epoch of data orderly and use it in test
    """

    def __init__(self, inputs, batch_size=10):
        super(OneEpochDataIterator, self).__init__(inputs, batch_size=batch_size)
        if batch_size > 0:
            self.idx_group = np.array_split(np.arange(self.len), np.ceil(self.len / batch_size))
        else:
            self.idx_group = [np.arange(self.len)]
        self.group_id = 0

    def next(self):
        if self.group_id >= len(self.idx_group):
            self.group_id = 0
            raise StopIteration
        out = self.inputs[self.idx_group[self.group_id], :]
        self.group_id += 1
        return [out[:, i]for i in range(self.num_cols)]

