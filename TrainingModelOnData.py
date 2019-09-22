"""
Predict function
𝑦𝑝𝑟𝑒𝑑[𝑢,𝑖]=𝑏𝑖𝑎𝑠𝑔𝑙𝑜𝑏𝑎𝑙+𝑏𝑖𝑎𝑠𝑢𝑠𝑒𝑟[𝑢]+𝑏𝑖𝑎𝑠𝑖𝑡𝑒𝑚[𝑖]+<𝑒𝑚𝑏𝑒𝑑𝑑𝑖𝑛𝑔𝑢𝑠𝑒𝑟[𝑢], 𝑒𝑚𝑏𝑒𝑑𝑑𝑖𝑛𝑔𝑖𝑡𝑒𝑚[𝑖] >

we need minimized the loss (add regularize terms)
∑𝑢,𝑖| 𝑦𝑝𝑟𝑒𝑑[𝑢,𝑖] − 𝑦𝑡𝑟𝑢𝑒[𝑢,𝑖] |2 + 𝜆(|  𝑒𝑚𝑏𝑒𝑑𝑑𝑖𝑛𝑔𝑢𝑠𝑒𝑟[𝑢] | 2 + | 𝑒𝑚𝑏𝑒𝑑𝑑𝑖𝑛𝑔𝑖𝑡𝑒𝑚[𝑖] | 2)
"""

import time
import numpy as np
import tensorflow as tf
import pandas as pd

from collections import deque
from six import next
from tensorflow.core.framework import summary_pb2

np.random.seed(13575)

# size of a batch of data
BATCH_SIZE = 1000
# user number
USER_NUM = 6040
# factor dimension
DIM = 15
# movie number
ITEM_NUM = 3952
# max number of iterate
EPOCH_MAX = 200
# use CPU to train data
DEVICE = "/cpu:0"


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
        return [out[:, i] for i in range(self.num_cols)]


# use matrix decomposition to build the grid structure
def inference_svd(user_batch, item_batch, user_num, item_num, dim=5, device="/cpu:0"):
    # use CPU
    with tf.device("/cpu:0"):
        # initial bias
        global_bias = tf.get_variable("global_bias", shape=[])
        w_bias_user = tf.get_variable("embd_bias_user", shape=[user_num])
        w_bias_item = tf.get_variable("embd_bias_item", shape=[item_num])

        # initial bias vectors
        bias_user = tf.nn.embedding_lookup(w_bias_user, user_batch, name="bias_user")
        bias_item = tf.nn.embedding_lookup(w_bias_item, item_batch, name="bias_item")
        w_user = tf.get_variable("embd_user", shape=[user_num, dim],
                                 initializer=tf.truncated_normal_initializer(stddev=0.02))
        w_item = tf.get_variable("embd_user", shape=[item_num, dim],
                                 initializer=tf.truncated_normal_initializer(stddev=0.02))

        # item vector and user vector
        embd_user = tf.nn.embedding_lookup(w_user, user_batch, name="embedding_user")
        embd_item = tf.nn.embedding_lookup(w_item, user_batch, name="embedding_item")

    with tf.device(device):
        # according to the formulation to calculate the inner product of user vector and item vector
        infer = tf.reduce_sum(tf.multiply(embd_user, embd_item), 1)
        # add some bias terms
        infer = tf.add(infer, global_bias)
        infer = tf.add(infer, bias_user)
        infer = tf.add(infer, bias_item, name="svd_inference")

        # add regularization terms
        regularization_term = tf.add(tf.nn.l2_loss(embd_user), tf.nn.l2_loss(embd_item), name="svd_regularizer")

    return infer, regularization_term


# iterate optimization part
def optimization(infer, regularizer, rate_batch, learning_rate=0.001, reg=0.1, device="/cpu:0"):
    global_step = tf.train.get_global_step()
    assert global_step is not None
    # find the appropriate optimizer to optimize this model
    cost_l2 = tf.nn.l2_loss(tf.subtract(infer, rate_batch))
    penalty = tf.constant(reg, dtype=tf.float32, shape=[], name="l2")
    cost = tf.add(cost_l2, tf.multiply(regularizer, penalty))  # add a regularizer to avoid it has overfitting
    # this is not convex model so GradientDescentOptimizer is not suitable here
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost, global_step=global_step)

    return cost, train_op


# cut off function
def clip(x):
    return np.clip(x, 1.0, 5.0)


# this summary is used to visualized the Tensorboard
def make_scalar_summary(name, val):
    return summary_pb2.Summary(value=[summary_pb2.Summary.Value(tag=name, simple_value=val)])


def read_data_and_process(filename, sep="\t"):
    col_names = ["user", "item", "rate", "st"]
    df = pd.read_csv(filename, sep=sep, header=None, names=col_names, engine='python')
    df["user"] -= 1
    df["item"] -= 1
    for col in ("user", "item"):
        df[col] = df[col].astype(np.int32)
    df["rate"] = df["rate"].astype(np.float32)
    return df


# using the functions above to get train data
def get_data():
    df = read_data_and_process("./movielens/ml-1m/ratings.dat", sep="::")
    rows = len(df)
    df = df.iloc[np.random.permutation(rows)].reset_index(drop=True)
    split_index = int(rows * 0.9)
    df_train = df[0:split_index]
    df_test = df[split_index:].reset_index(drop=True)
    print(df_train.shape, df_test.shape)
    return df_train, df_test


# the real train process
def svd(train, test):
    sample_per_batch = len(train) // BATCH_SIZE

    # one batch of data used to train
    iter_train = ShuffleDataIterator([train["user"], train["item"], train["rate"]], batch_size=BATCH_SIZE)

    # test data
    iter_test = OneEpochDataIterator([test["user"], test["item"], test["rate"]], batch_size=-1)

    # user and item batch
    user_batch = tf.placeholder(tf.int32, shape=[None], name="id_user")
    item_batch = tf.placeholder(tf.int32, shape=[None], name="id_item")
    rate_batch = tf.placeholder(tf.float32, shape=[None])

    # build graph and train
    infer, regularizer = inference_svd(user_batch, item_batch, user_num=USER_NUM,
                                       item_num=ITEM_NUM, dim=DIM, device=DEVICE)
    global_step = tf.contrib.framework.get_or_create_global_step()
    _, train_op = optimization(infer, regularizer, rate_batch, learning_rate=0.001, reg=0.05, device=DEVICE)

    # initial all the variables
    init_op = tf.global_variables_initializer()

    # Start iterate processing
    with tf.Session() as sess:
        sess.run(init_op)
        summary_writer = tf.summary.FileWriter(logdir="/tmp/svd/log", graph=sess.graph)
        print("{} {} {} {}".format("epoch", "train_error", "val_error", "elapsed_time"))
        errors = deque(maxlen=sample_per_batch)
        start = time.time()
        for i in range(EPOCH_MAX * sample_per_batch):
            users, items, rates = next(iter_train)
            _, pred_batch = sess.run([train_op, infer], feed_dict={user_batch: users,
                                                                   item_batch: items, rate_batch:rates})
            pred_batch = clip(pred_batch)
            errors.append(np.power(pred_batch - rates, 2))
            if i % sample_per_batch == 0:
                train_err = np.sqrt(np.mean(errors))
                test_err2 = np.array([])
                for users, items, rates, in iter_test:
                    pred_batch = sess.run(infer, feed_dict={user_batch: users,
                                                            item_batch: items})
                    pred_batch = clip(pred_batch)
                    test_err2 = np.append(test_err2, np.power(pred_batch - rates, 2))

                end = time.time()
                test_err = np.sqrt(np.mean(test_err2))
                print("{:3d} {:f} {:f} {:f}(s)".format(i // sample_per_batch, train_err,
                                                       test_err, end - start))
                train_err_summary = make_scalar_summary("training_error", train_err)
                test_err_summary = make_scalar_summary("test_error", test_err)
                summary_writer.add_summary(train_err_summary, i)
                summary_writer.add_summary(test_err_summary, i)
                start = end


# get data
df_train, df_test = get_data()
# finish the train
svd(df_train, df_test)

