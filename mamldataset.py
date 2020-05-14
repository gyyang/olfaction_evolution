""" Code for loading data. """
import os
import shutil

import numpy as np

import tools
from configs import input_ProtoConfig
import configs
import task


class DataGenerator(object):
    def __init__(
            self,
            dataset,
            batch_size,
            meta_batch_size,
            num_samples_per_class=1,
            num_class=2,
            dim_output=2,
    ):
        train_x, train_y, val_x, val_y = task.load_data(dataset)

        self.meta_bs = meta_batch_size
        self.batch_size = batch_size
        self.num_samples_per_class = num_samples_per_class
        self.num_class = num_class
        self.dim_output = dim_output

        self.train_x = train_x
        self.train_y = train_y

        if np.ndim(train_y) == 2: #hack for multi_class_one_hot
            class_y, valence_y = train_y[:,0], train_y[:,1]
            self.train_y = class_y
            self.valence_y = valence_y
            self.n_valence_class = np.unique(valence_y).size

        else:
            class_y = train_y
            self.train_y = class_y
            self.n_valence_class = None

        # dictionary mapping class to odor indices
        unique_y = np.unique(class_y)
        self.ind_dict = {y: np.where(class_y==y)[0] for y in unique_y}

        if np.ndim(train_y) == 2:
            self.metatrain_classes = unique_y
            self.metaval_classes = unique_y
        else:
            self.metatrain_classes = unique_y[:int(0.5 * len(unique_y))]
            self.metaval_classes = unique_y[int(0.5 * len(unique_y)):]


        if np.mod(num_class, dim_output) != 0:
            raise ValueError('Now only supporting num_class multiples of dim_output')

    def generate(self, dataset_type='train'):
        """Generate one meta-batch.

        Args:
            dataset_type: str, 'train' or 'val'

        Returns:
            inputs: array, (meta_batch_size, n_samples_per_class, dim_input)
            outputs_head1: array, (meta_batch_size, n_samples_per_class, dim_output)
        """
        if dataset_type == 'train':
            all_classes = self.metatrain_classes
        elif dataset_type == 'val':
            all_classes = self.metaval_classes
        else:
            raise ValueError('Unknown dataset type: ' + str(dataset_type))

        n_sample_per_class = self.num_samples_per_class
        n_class_per_batch = self.num_class
        assert n_sample_per_class * n_class_per_batch * 2 == self.batch_size

        inputs = np.zeros([self.meta_bs, self.batch_size, self.train_x.shape[-1]])
        outputs_head1 = np.zeros([self.meta_bs, self.batch_size, self.dim_output])
        if self.n_valence_class is not None:
            outputs_head2 = np.zeros([self.meta_bs, self.batch_size, self.n_valence_class])

        for i in range(self.meta_bs):
            # randomly select several classes to train on
            classes = np.random.choice(
                all_classes, size=n_class_per_batch, replace=False)
            # relabel them
            # new_labels = np.random.randint(0, n_valence, len(classes))
            # TODO: what to do when n_class_per_batch different from dim_output?
            new_labels = (list(range(self.dim_output)) *
                          (self.num_class//self.dim_output))

            # for each class, sample some odors
            j = 0
            for _ in range(2):  # repeat twice for lossa and lossb
                for c, l in zip(classes, new_labels):
                    ind = np.random.choice(
                        self.ind_dict[c], n_sample_per_class, replace=False)
                    inputs[i, j:j+n_sample_per_class, :] = self.train_x[ind, :]
                    outputs_head1[i, j:j+n_sample_per_class, l] = 1.0  # one-hot

                    if self.n_valence_class:
                        outputs_head2[i,j:j+n_sample_per_class, self.valence_y[ind]] = 1.0
                    j += n_sample_per_class

        if self.n_valence_class is not None:
            outputs_head = np.concatenate((outputs_head1, outputs_head2), axis=2)
        else:
            outputs_head = outputs_head1

        # n_orn = 50
        # noise = .01
        # for i in range(self.meta_bs):
        #     prototypes = np.random.uniform(0, 1, (n_class_per_batch, n_orn))
        #
        #     labels = np.random.random_integers(0, n_class_per_batch-1, self.batch_size)
        #     odor_noise = np.random.uniform(0, noise, (self.batch_size, n_orn))
        #     odor_base = prototypes[labels,:]
        #     odors = odor_base + odor_noise
        #
        #     labels = task._get_labels(prototypes, odors, percent_generalization=100)
        #     inputs[i,:,:] = odors
        #     outputs_head1[i, np.arange(labels.size), labels] = 1
        outputs_head = outputs_head1

        return inputs, outputs_head


def _generate_meta_proto():
    config = configs.MetaConfig()
    config.label_type = 'multi_head_one_hot'
    config.data_dir = './datasets/proto/multi_head'
    num_samples_per_class = config.meta_num_samples_per_class
    num_class = config.N_CLASS
    meta_batch_size = config.meta_batch_size
    dim_output = config.meta_output_dimension
    dataset = config.data_dir
    data_generator = DataGenerator(
        dataset= dataset,
        batch_size=num_samples_per_class * num_class * 2,  # 5 is # classes
        meta_batch_size= meta_batch_size,
        num_samples_per_class=num_samples_per_class,
        num_class=num_class,
        dim_output=dim_output,
    )
    inputs, outputs = data_generator.generate()
    return inputs, outputs


def save_proto(config=None, seed=0, folder_name=None):
    """Save dataset in numpy format."""

    if config is None:
        config = configs.input_ProtoConfig()

    # make and save data
    train_x, train_y = _generate_meta_proto()

    folder_path = os.path.join(config.path, folder_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    else:
        shutil.rmtree(folder_path)
        os.makedirs(folder_path)

    vars = [train_x.astype(np.float32), train_y.astype(np.float32)]
    varnames = ['train_x', 'train_y']
    for result, name in zip(vars, varnames):
        np.save(os.path.join(folder_path, name), result)

    #save parameters
    tools.save_config(config, folder_path)
    return folder_path


def load_data(data_dir):
    """Load dataset."""
    names = ['train_x', 'train_y']
    return [np.load(os.path.join(data_dir, name + '.npy')) for name in names]


if __name__ == '__main__':
    save_proto(folder_name='meta_proto')