import os


class BaseConfig(object):
    def __init__(self):
        pass

    def update(self, new_config):
        self.__dict__.update(new_config.__dict__)

    def __str__(self):
        return str(self.__dict__)


class input_ProtoConfig(BaseConfig):
    def __init__(self):
        super(input_ProtoConfig, self).__init__()
        self.path = os.path.join(os.getcwd(), 'datasets', 'proto')
        self.hallem_path = os.path.join(os.getcwd(),'datasets','hallem')

        self.n_train = 1000000
        self.n_val = 8192

        self.N_CLASS = 100
        self.N_ORN = 50

        # label type can be either combinatorial, one_hot, sparse
        self.label_type = 'sparse'

        self.percent_generalization = 100
        self.n_combinatorial_classes = 20
        self.combinatorial_density = .3
        # If True, the enclidean distance used for nearest neighbor is
        # computed in a distorted space
        self.distort_input = False
        # If True, shuffle the train and validation labels
        self.shuffle_label = False

        # If relabel is True, then randomly relabel the classes
        # The number of true classes (pre-relabel) is n_trueclass
        # The number of classes post relabeling is N_CLASS
        self.relabel = False
        self.n_trueclass = 1000

        # if True, concentration is varied independently of the odor identity
        self.vary_concentration = False


class SingleLayerConfig(BaseConfig):
    def __init__(self):
        super(SingleLayerConfig, self).__init__()
        self.dataset = 'repeat'
        self.model = 'singlelayer'
        self.lr = .001
        self.max_epoch = 100
        self.batch_size = 256
        self.save_path = './files/peter_tmp'


# class FullConfig(input_ProtoConfig):
class FullConfig(BaseConfig):
    def __init__(self):
        super(FullConfig, self).__init__()
        self.dataset = 'proto'
        self.data_dir = './datasets/proto/_100_generalization_onehot_s0'
        self.model = 'full'
        self.save_path = './files/test'
        self.save_every_epoch = False

        self.lr = .001  # learning rate
        self.max_epoch = 10
        self.batch_size = 256
        self.target_acc = None  # target accuracy

        # Overall architecture
        # If False, ORNs are already replicated in the dataset
        self.replicate_orn_with_tiling = True
        self.N_ORN_DUPLICATION = 10
        self.N_PN = 50
        self.N_KC = 2500

        #noise model
        #model for noise: can be 'additive'. 'multiplicative', or None
        self.NOISE_MODEL = 'additive'
        self.ORN_NOISE_STD = 0.

        # ORN--> PN connections

        # Initialization method for pn2kc: can take values uniform, random, or normal
        self.initializer_orn2pn = 'normal'
        # If True, ORN --> PN connections are positive
        self.sign_constraint_orn2pn = True
        # If True, PN --> KC connections are trainable
        self.train_orn2pn = True
        # If True, train a direct glomeruli-like connections
        self.direct_glo = False
        # PN normalization before non_linearity
        self.pn_norm_pre = None
        # PN normalization after non_linearity
        self.pn_norm_post = None
        # If True, skip the ORN --> PN connections
        self.skip_orn2pn = False


        # PN --> KC connections

        # Initialization method for pn2kc: can take values uniform, random, or normal
        self.initializer_pn2kc = 'constant'
        # Initial value of pn2kc weights. if it is set to 0, network will initialize according to sparsity
        self.initial_pn2kc = 0
        # If True, ORN --> PN connections are positive
        self.sign_constraint_pn2kc = True
        # If True, PN --> KC connections are trainable
        self.train_pn2kc = False
        # If True, PN --> KC connections are sparse
        self.sparse_pn2kc = True
        # If True, PN --> KC connections are mean-subtracted (sum of all connections onto every KC is 0)
        self.mean_subtract_pn2kc = False
        # If True, KC biases are trainable
        self.train_kc_bias = True
        # initial KC bias
        self.kc_bias = -1
        # If True, PN --> KC connection weights are uniform
        self.uniform_pn2kc = False
        # If True, have loss on KC weights
        self.kc_loss = False
        # Parameters for KC loss. Only used if kc_loss = True.
        # alpha = loss strength.
        self.kc_loss_alpha = 1
        # beta = when to apply loss. higher the value, the smaller the weight in which loss will be applied
        self.kc_loss_beta = 10
        # KC normalization before non_linearity
        self.kc_norm_pre = None
        # KC normalization after non_linearity
        self.kc_norm_post = None
        # If True, add dropout to KC layer
        self.kc_dropout = True
        self.kc_dropout_rate = 0.5
        # If True, skip the PN --> KC connections
        self.skip_pn2kc = False
        # number of inputs onto KCs
        self.kc_inputs = 7

