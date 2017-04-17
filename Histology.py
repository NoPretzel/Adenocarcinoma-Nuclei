import tensorflow as tf
import scipy.io
import numpy as np


class Histology:

    def __init__(self, image_directory, image_file_type):
        self.histology_queue = tf.train.string_input_producer(
            tf.train.match_filenames_once("{}*{}".format(image_directory, image_file_type)))
        self.image_file_type = image_file_type
        self.image_reader = tf.WholeFileReader()

    def size(self):
        self.histology_queue.size()

    def next_batch(self, batch_size):
        file_name, image_file = self.image_reader.read(self.histology_queue)
        image = tf.image.decode_jpeg(image_file)
        num_preprocess_threads = 1
        min_queue_examples = 256

        return tf.train.shuffle_batch(
            [image, self.get_labels_for_image_tensor(image, file_name.replace("{}".format(self.image_file_type), '.mat'))],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples)

    @staticmethod
    def get_labels_for_image_tensor(image_tensor, classification_file):
        ml = scipy.io.loadmat(classification_file)
        labels = np.zeros(len(image_tensor), len(image_tensor))
        for nucleus in ml['detection']:
            labels[int(nucleus[0])][int(nucleus[1])] = 1
        return labels

