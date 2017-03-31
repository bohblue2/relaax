import tensorflow as tf


class Subgraph(object):
    def __init__(self, *args, **kwargs):
        with tf.variable_scope(type(self).__name__):
            self.__node = self.build(*args, **kwargs)

    @property
    def node(self):
        return self.__node
