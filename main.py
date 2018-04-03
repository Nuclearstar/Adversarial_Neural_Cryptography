import tensorflow as tf

BATCH_SIZE = 4096
MSG_SIZE = 16
KEY_SIZE = 16

N = MSG_SIZE + KEY_SIZE


def separation(o_message, dist):
    """Distance from Alice where the message is sent upto the point of receiving end"""
    return tf.reduce_sum(
        tf.abs(tf.subtract(o_message, dist)),
        reduction_indices=1
    )

def eve_retribution(o_message, decipher_eve):
    return tf.square(MSG_SIZE/2 - separation(o_message, decipher_eve)) / (MSG_SIZE/2) ** 2


def cipher_loss(o_message, decipher_bob, decipher_eve):
    """The loss function of Alice and Bob. Compute from the original plain text
    o_message, the deciphered decipher_bob of Bob and the deciphered decipher_eve of Eve."""
    return separation(o_message, decipher_bob)/MSG_SIZE + eve_retribution(o_message, decipher_eve)

def attacker_loss(o_message, decipher_eve):
    return separation(o_message, decipher_eve)

def weights(shape, name):
    initial = tf.truncated_normal(shape, stddev=1)
    return tf.Variable(initial, name=name)

def bias(shape, name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)

def build_cipher_network(encrypt_msg, name, initial_size):
    net = tf.matmul(encrypt_msg, weights([initial_size, N], name="{}/fc_layer".format(name)))
    net = tf.reshape(net, [BATCH_SIZE, N, 1])
    net = tf.sigmoid(tf.nn.conv1d(net, weights([4, 1, 2], name="{}/conv1d-1".format(name)), stride=1, padding="SAME"))
    net = tf.sigmoid(tf.nn.conv1d(net, weights([2, 2, 4], name="{}/conv1d-2".format(name)), stride=2, padding="SAME"))
    net = tf.sigmoid(tf.nn.conv1d(net, weights([1, 4, 4], name="{}/conv1d-3".format(name)), stride=1, padding="SAME"))
    net = tf.tanh(tf.nn.conv1d(net, weights([1, 4, 1], name="{}/conv1d-4".format(name)), stride=1, padding="SAME"))
    net = tf.reshape(net, [BATCH_SIZE, MSG_SIZE])
    return net

if __name__ == '__main__':
    sess = tf.InteractiveSession()

    # Here we generate a shared key between Alice and Bob for security
    key = 2 * tf.random_uniform([BATCH_SIZE, KEY_SIZE], minval=0, maxval=2, dtype=tf.int32) - 1
    key = tf.to_float(key)

    # Alice input: generate plaintext or message that has to be sent to Bob
    o_message = 2 * tf.random_uniform([BATCH_SIZE, MSG_SIZE], minval=0, maxval=2, dtype=tf.int32) - 1
    o_message = tf.to_float(o_message)
    encrypt_msg = tf.concat([o_message, key], 1)

    # Alice sents output and Bob and Eve tries to receive it as input: cipher text
    cipher_text = tf.placeholder(tf.float32, shape=[BATCH_SIZE, MSG_SIZE])

    # Bob output: deciphered text that should be equal to o_message
    decipher_bob = tf.placeholder(tf.float32, shape=[BATCH_SIZE, MSG_SIZE])

    # Eve output: deciphered text that should be equal to o_message
    decipher_eve = tf.placeholder(tf.float32, shape=[BATCH_SIZE, MSG_SIZE])

    cipher_text = build_cipher_network(encrypt_msg, name="cipher/a", initial_size=N)
    decipher_bob = build_cipher_network(tf.concat([cipher_text, key], 1), name="cipher/b", initial_size=N)
    decipher_eve = build_cipher_network(cipher_text, name="attacker", initial_size=MSG_SIZE)


    gain = tf.train.AdamOptimizer(0.0008)

    cipher_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "cipher/")
    attacker_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "attacker/")

    cipher_training_step = gain.minimize(cipher_loss(o_message, decipher_bob, decipher_eve), var_list=cipher_vars)
    attacker_training_step = gain.minimize(attacker_loss(o_message, decipher_eve), var_list=attacker_vars)

    cipher_accuracy = tf.reduce_mean(separation(o_message, decipher_bob))
    attacker_accuracy = tf.reduce_mean(attacker_loss(o_message, decipher_eve))

    sess.run(tf.initialize_all_variables())
    for q in range(20000):
      if q % 100 == 0:
        accuracy_of_training = cipher_accuracy.eval(), attacker_accuracy.eval()
        print("step {}, training accuracy {}".format(q, accuracy_of_training))
      cipher_training_step.run()
      attacker_training_step.run()
      attacker_training_step.run()
