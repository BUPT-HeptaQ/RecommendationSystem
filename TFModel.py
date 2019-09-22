import tensorflow as tf


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

