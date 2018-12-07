import os

class input_ProtoConfig(object):
    def __init__(self):
        self.path = os.path.join(os.getcwd(), 'datasets', 'proto')
        self.hallem_path = os.path.join(os.getcwd(),'datasets','hallem')

        self.n_train = 1000000
        self.n_val = 8192

        self.N_CLASS = 100
        self.N_ORN = 50
        self.N_PN = 50
        self.ORN_NOISE_STD = 0.
        self.N_KC = 2500

        self.percent_generalization = 100
        self.use_combinatorial = False
        self.n_combinatorial_classes = 20
        self.combinatorial_density = .3
        # If True, the enclidean distance used for nearest neighbor is
        # computed in a distorted space
        self.distort_input = False
        # If True, shuffle the train and validation labels
        self.shuffle_label = False

        # If False, ORNs are already replicated in the dataset
        self.replicate_orn_with_tiling = True
        self.N_ORN_DUPLICATION = 10


class SingleLayerConfig(input_ProtoConfig):
    def __init__(self):
        super(SingleLayerConfig, self).__init__()
        self.dataset = 'repeat'
        self.model = 'singlelayer'
        self.lr = .001
        self.max_epoch = 100
        self.batch_size = 256
        self.save_path = './files/peter_tmp'


class FullConfig(input_ProtoConfig):
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


        # ORN--> PN connections
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
        # KC normalization before non_linearity
        self.kc_norm_pre = None
        # KC normalization after non_linearity
        self.kc_norm_post = None
        # If True, add dropout to KC layer
        self.kc_dropout = True
        # If True, skip the PN --> KC connections
        self.skip_pn2kc = False
        # number of inputs onto KCs
        self.kc_inputs = 7

        # label type can be either combinatorial, one_hot, sparse
        self.label_type = 'one_hot'

