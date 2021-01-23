import os


class BaseConfig(object):
    def __init__(self):
        self.experiment_name = None
        self.model_name = None

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
        #TODO: this name should really be N_OR. fix without breaking code
        self.N_ORN = 50
        self.n_or_per_orn = 1

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

        # If label_type == 'multi_head_sparse', the second head is valence
        self.n_class_valence = 3
        # If has_special_odors is True, then some odors will activate single ORs
        self.has_special_odors = True
        self.special_odor_activation = 1.
        # the number of prototypes that leads to each non-neutral response
        self.n_proto_valence = 5

        # If tuple[0] = True, each odor will have an ORN response probability sampled from a distribution
        # tuple[1] = the degree of masking, varies from (0, 1]. Defines the bimodality of the prob dist.
        self.mask_orn_activation_row = (False, 8)

        # If tuple[0] = True, every orn will have an odor response probability sampled from a distribution
        # tuple[1] = the degree of masking, varies from (0, 1]. Defines the bimodality of the prob dist.
        self.mask_orn_activation_column = (False, 0)

        # If tuple[0] = True, total orn activity becomes more spread out as defined by a distribution
        # tuple[1] = Spread, varies from (0, 1]. Defines the bimodality of the prob dist.
        self.spread_orn_activity = (False, .5)

        # Whether to have correlation between ORNs
        self.orn_corr = None


class InputAutoEncode(BaseConfig):
    def __init__(self):
        super(InputAutoEncode, self).__init__()
        self.path = os.path.join(os.getcwd(), 'datasets', 'autoencode')

        self.n_train = 1000000
        self.n_val = 8192

        self.n_class = 100
        self.n_orn = 50

        self.proto_density = 0.5
        self.p_flip = 0.2


class SingleLayerConfig(BaseConfig):
    def __init__(self):
        super(SingleLayerConfig, self).__init__()
        self.dataset = 'proto'
        self.data_dir = './datasets/proto/standard'
        self.model = 'singlelayer'
        self.lr = .001
        self.max_epoch = 100
        self.batch_size = 256
        self.save_path = './files/test'


class FullConfig(BaseConfig):
    def __init__(self):
        super(FullConfig, self).__init__()
        self.dataset = 'proto'
        # self.data_dir = './datasets/proto/standard'
        # New standard dataset is relabel
        self.data_dir = './datasets/proto/relabel_200_100'
        #model can be full, normmlp, or singlelayer
        self.model = 'full'
        self.save_path = './files/test'
        self.save_every_epoch = False
        self.save_log_only = False
        self.save_epoch_interval = 1

        # self.lr = .001  # learning rate
        self.lr = 5e-4  # new default for relabel dataset
        self.decay_steps = 1e8  # learning rate decay steps
        self.decay_rate = 1.  # learning rate decay rate, default to no decay
        self.max_epoch = 100
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

        # Receptor --> ORN connections

        # If True, create an receptor layer
        self.receptor_layer = False
        # Initialization method for or2orn: can take values uniform, random, or normal
        self.initializer_or2orn = 'uniform'
        # If True, OR --> ORN connections are positive
        self.sign_constraint_or2orn = True
        # If True, normalize by or2orn weight matrix by L1 norm (sum of weights onto every ORN add up to 1)
        self.or2orn_normalization = True
        # If True, add bias to receptor weights
        self.or_bias = False
        # ORN normalization. orn never experiences nonlinearity so no distinction between pre and post
        self.orn_norm = None
        # If True, set ORN weights manually
        self.orn_manual = False
        # Varies from 0-1. Controls the level of randomness in ORN-PN weights.
        self.orn_random_alpha = 0

        # ORN--> PN connections
        # whether to dropout at ORN layer
        self.orn_dropout = False  # TODO: If True, now applied POST tiling, but consider PRE tiling
        self.orn_dropout_rate = 0.1

        # Initialization method for pn2kc: can take values uniform, random, or normal
        self.initializer_orn2pn = 'uniform'
        # If True, ORN --> PN connections are positive
        self.sign_constraint_orn2pn = True
        # If True, PN --> KC connections are trainable
        self.train_orn2pn = True
        # If True, train a direct glomeruli-like connections
        self.skip_orn2pn = False  # Implements direct_glo in TF version
        # PN normalization before non_linearity
        self.pn_norm_pre = 'batch_norm'  # Default to BatchNorm
        # PN normalization after non_linearity
        self.pn_norm_post = None
        # If True, skip the ORN --> PN connections
        # self.skip_orn2pn = False  # TODO: Completely remove
        # dropout for pn
        self.pn_dropout = False
        # dropout rate for pns
        self.pn_dropout_rate = .2
        # If True, normalize orn2pn weight matrix by L1 norm (sum of weights onto every PN add up to 1)
        self.orn2pn_normalization = False  # TODO: Check if works

        self.pn_prune_weak_weights = False
        self.pn_prune_threshold = .05
        self.initial_orn2pn = 0


        # PN --> KC connections

        # Initialization method for pn2kc: can take values uniform, random, or normal
        self.initializer_pn2kc = 'uniform'
        # Initial value of pn2kc weights. if it is set to 0, network will initialize according to sparsity
        self.initial_pn2kc = 0  # TODO: Need revisit, try use 0 for all
        # If True, ORN --> PN connections are positive
        self.sign_constraint_pn2kc = True
        # If True, PN --> KC connections are trainable
        self.train_pn2kc = True
        # If True, PN --> KC connections are sparse
        self.sparse_pn2kc = False
        # If True, PN --> KC connections are mean-subtracted (sum of all connections onto every KC is 0)
        self.mean_subtract_pn2kc = False
        # If True, KC biases are trainable
        self.train_kc_bias = True
        # initial KC bias
        self.kc_bias = -1
        # KC normalization before non_linearity
        self.kc_norm_pre = None
        # KC normalization after non_linearity
        self.kc_norm_post = None
        # If True, add dropout to KC layer
        self.kc_dropout = True
        self.kc_dropout_rate = 0.5  # new default for relabel dataset
        # If True, skip the PN --> KC connections
        self.skip_pn2kc = False
        # number of inputs onto KCs
        self.kc_inputs = 7
        # multiplicative noise on PN to KC connectivity
        self.pn2kc_noise = False
        self.pn2kc_noise_value = 0.2
        # noise onto KCs
        self.kc_noise = False
        self.kc_noise_std = 0.2
        # coding level on kcs
        self.coding_level = None

        # Whether to prune weak KC weights
        self.kc_prune_weak_weights = False
        self.kc_prune_threshold = 0.02

        # Whether to do feedforward or recurrent inhibition on KC
        self.kc_ffinh = False
        self.kc_ffinh_coeff = 0
        self.kc_recinh = False
        self.kc_recinh_coeff = 0
        self.kc_recinh_step = 10

        #separate optimizer
        # TODO: Remove in the future
        # self.separate_optimizer = False
        # self.separate_lr = 0.001

        # New layer after KC
        # TODO: Remove in the future
        # self.extra_layer = False
        # self.extra_layer_neurons = 200

        # TODO: Remove in the future
        # Output connections
        self.output_bias = True
        # If True, set the output weights to be the oracle (pattern-matching)
        self.set_oracle = False
        # Scale the oracle weights
        self.oracle_scale = 1.0

        # Computing loss
        # Only meaningful for multi_head configuration
        self.train_head1 = True
        self.train_head2 = True


class RNNConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        self.dataset = 'proto'
        self.data_dir = './datasets/proto/standard'

        self.model = 'rnn'
        self.save_path = './files/test'
        self.save_every_epoch = False
        self.save_log_only = False
        self.save_epoch_interval = 1

        self.lr = .001  # learning rate
        self.decay_steps = 1e8  # learning rate decay steps
        self.decay_rate = 1.  # learning rate decay rate, default to no decay
        self.max_epoch = 30
        self.batch_size = 256
        self.target_acc = None  # target accuracy

        # Overall architecture
        # If False, ORNs are already replicated in the dataset
        self.N_ORN_DUPLICATION = 10
        self.N_PN = 50
        self.NEURONS = 2500

        #noise model
        #model for noise: can be 'additive'. 'multiplicative', or None
        self.NOISE_MODEL = 'additive'
        self.ORN_NOISE_STD = 0.

        # Recurrent steps
        self.TIME_STEPS = 2
        # Initialization method for pn2kc: can take values uniform, random, or normal
        self.initializer_rec = 'uniform'
        # Initial value of rec weights. if it is set to 0, network will
        # initialize according to sparsity
        self.initial_rec = 0
        # If True, ORN --> PN connections are positive
        self.sign_constraint_rec = True
        # Normalization before non_linearity
        self.rec_norm_pre = 'batch_norm'  # Default to BatchNorm
        # Normalization after non_linearity
        self.rec_norm_post = None
        # dropout for pn
        self.rec_dropout = True
        # dropout rate for pns
        self.rec_dropout_rate = .0
        # Initialize rec weight as diagonal matrix
        self.diagonal = False
        # Weight dropout
        self.weight_dropout = False
        self.weight_dropout_rate = 0.
        # Whether to prune weak KC weights
        self.prune_weak_weights = False
        self.prune_threshold = 0.
        # Whether to prevent neurons from being reactivated
        self.allow_reactivation = True


class MetaConfig(FullConfig):
    def __init__(self):
        super(MetaConfig, self).__init__()
        # data directory
        self.data_dir = './datasets/proto/standard'
        # model type
        self.model = 'full'
        # how many points for input generation
        self.meta_n_dataset = 1000 * 32
        # number of classes
        self.N_CLASS = 4
        # number of labels per class
        self.meta_labels_per_class = 1
        # number of metatraining iterations
        self.metatrain_iterations = 100000
        # number of tasks sampled per meta-update (outer batch size)
        self.meta_batch_size = 16
        # the base learning rate of the generator
        self.meta_lr = .001
        # number of inner gradient updates during training
        self.meta_num_updates = 1
        # step size alpha for inner gradient update
        self.meta_update_lr = .3
        # number of examples used for inner gradient update (K for K-shot learning)
        self.meta_num_samples_per_class = 8
        # batch_norm, layer_norm, or None
        self.meta_norm = 'None'
        # if True, do not use second derivatives in meta-optimization (for speed)
        self.meta_stop_grad = False
        # label type for the meta dataset
        self.label_type = 'one_hot'
        # saving / printing epoch interval
        self.meta_print_interval = 250
        # maximum learning rate for the KC-output layer
        self.output_max_lr = 0.2
        # trainable lr?
        self.meta_trainable_lr = False

