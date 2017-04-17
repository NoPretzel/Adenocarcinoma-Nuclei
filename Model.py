import tensorflow as tf


class Model:

    def __init__(self, output_classes, batch_size, histology):
        self.output_classes = output_classes
        self.batch_size = batch_size
        self.histology = histology

        self.y = tf.placeholder('float')

    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def maxpool2d(self, x):
        return tf.nn.max_pool(x, ksize=[1, 1, 1, 1], strides=[1, 1, 1, 1], padding='SAME')

    def cnn(self, x):
        weights = {'W_conv1': tf.Variable(tf.random_normal([1, 1, 1, 25])),
                   'W_conv2': tf.Variable(tf.random_normal([1, 1, 25, 64])),
                   'W_fc': tf.Variable(tf.random_normal([7 * 7 * 64, 1024])),
                   'out': tf.Variable(tf.random_normal([1024, self.output_classes]))}

        biases = {'b_conv1': tf.Variable(tf.random_normal([32])),
                  'b_conv2': tf.Variable(tf.random_normal([64])),
                  'b_fc': tf.Variable(tf.random_normal([1024])),
                  'out': tf.Variable(tf.random_normal([self.output_classes]))}

        x = tf.reshape(x, shape=[-1, 28, 28, 1])

        conv1 = tf.nn.relu(self.conv2d(x, weights['W_conv1']) + biases['b_conv1'])
        conv1 = self.maxpool2d(conv1)

        conv2 = tf.nn.relu(self.conv2d(conv1, weights['W_conv2']) + biases['b_conv2'])
        conv2 = self.maxpool2d(conv2)

        fc = tf.reshape(conv2, [-1, 7 * 7 * 64])
        fc = tf.nn.relu(tf.matmul(fc, weights['W_fc']) + biases['b_fc'])
        fc = tf.nn.dropout(fc, 0.8)

        output = tf.matmul(fc, weights['out']) + biases['out']

        return output

    def train(self, x, batch_size):
        prediction = self.cnn(x)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, self.y))
        optimizer = tf.train.AdamOptimizer().minimize(cost)

        hm_epochs = 10
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())

            for epoch in range(hm_epochs):
                epoch_loss = 0
                for _ in range(self.histology.size() / batch_size):
                    epoch_x, epoch_y = self.histology.next_batch()
                    _, c = sess.run([optimizer, cost], feed_dict={'x': epoch_x, 'y': epoch_y})
                    epoch_loss += c

                print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)

            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(self.y, 1))

            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
