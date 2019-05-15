import keras
from keras import backend as K

class SimpleRNNCell(keras.layers.Layer):

    def __init__(self, n_emb, n_hidden, **kwargs):
        self.units = n_hidden
        self.state_size = n_emb
        super(SimpleRNNCell, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(input_shape[-1], self.units),
                                      initializer='uniform',
                                      name='kernel')
        self.recurrent_kernel = self.add_weight(
            shape=(self.state_size, self.units),
            initializer='uniform',
            name='recurrent_kernel')
        self.built = True

    def call(self, inputs, states):
        prev_output = states[0]
        print(prev_output.shape, self.recurrent_kernel.shape, inputs.shape, self.kernel.shape)
        h = K.dot(inputs, self.kernel)
        output = h + K.dot(prev_output, self.recurrent_kernel)
        return output, [output]