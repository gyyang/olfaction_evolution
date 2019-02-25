""" Code for loading data. """
import numpy as np

import task

class ReferenceDataGenerator(object):
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


class DataGenerator(object):
    def __init__(self, batch_size, meta_batch_size):
        train_x, train_y, val_x, val_y = task.load_data(
            'proto', './datasets/proto/standard')

        self.meta_bs = meta_batch_size
        self.batch_size = batch_size

        self.train_x = train_x
        self.train_y = train_y

        # dictionary mapping class to odor indices
        unique_y = np.unique(train_y)
        self.ind_dict = {y: np.where(train_y==y)[0] for y in unique_y}
        self.unique_y = unique_y

    def generate(self):
        """Generate one meta-batch.

        Returns:
            inputs: array, (meta_batch_size, n_samples_per_class, dim_input)
            outputs: array, (meta_batch_size, n_samples_per_class, dim_output)
        """
        # TODO: don't manually set here
        n_sample_per_class = 5
        n_class_per_batch = 5
        assert n_sample_per_class * n_class_per_batch * 2 == self.batch_size

        inputs = np.zeros([self.meta_bs, self.batch_size, self.train_x.shape[-1]])
        outputs = np.zeros([self.meta_bs, self.batch_size])

        for i in range(self.meta_bs):
            # randomly select several classes to train on
            classes = np.random.choice(
                self.unique_y, size=n_class_per_batch, replace=False)
            # relabel them
            new_labels = np.random.randint(0, 3, len(classes))

            # for each class, sample some odors
            j = 0
            for _ in range(2):  # repeat twice for lossa and lossb
                for c, l in zip(classes, new_labels):
                    ind = np.random.choice(
                        self.ind_dict[c], n_sample_per_class, replace=False)
                    inputs[i, j:j+n_sample_per_class, :] = self.train_x[ind, :]
                    outputs[i, j:j+n_sample_per_class] = l
                    j += n_sample_per_class

        return inputs, outputs
