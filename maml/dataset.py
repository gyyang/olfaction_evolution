""" Code for loading data. """
import numpy as np

class DataGenerator(object):
    """
    Data Generator capable of generating batches of sinusoid or Omniglot data.
    A "class" is considered a class of omniglot digits or a particular sinusoid function.
    """
    def __init__(self, batch_size, meta_batch_size):
        """
        Args:
            batch_size: num samples in one learning batch
            meta_batch_size: size of meta batch size (e.g. number of functions)
        """
        self.meta_bs = meta_batch_size
        self.batch_size = batch_size
        self.num_classes = 1  # by default 1 (only relevant for classification problems)

        self.dim_input = 1
        self.dim_output = 1

    def generate(self):
        """Generate one meta-batch.

        Returns:
            inputs: array, (meta_batch_size, n_samples_per_episode, dim_input)
            outputs: array, (meta_batch_size, n_samples_per_episode, dim_output)
        """
        amp = np.random.uniform(0.1, 5.0, [self.meta_bs])
        phase = np.random.uniform(0, np.pi, [self.meta_bs])

        inputs = np.zeros([self.meta_bs, self.batch_size, self.dim_input])
        outputs = np.zeros([self.meta_bs, self.batch_size, self.dim_output])

        for func in range(self.meta_bs):
            inputs[func] = np.random.uniform(-5.0, 5.0, [self.batch_size, 1])
            outputs[func] = amp[func] * np.sin(inputs[func]-phase[func])
        return inputs, outputs
