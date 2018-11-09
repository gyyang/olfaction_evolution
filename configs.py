import os

class input_ProtoConfig(object):
    def __init__(self):
        self.path = os.path.join(os.getcwd(), 'datasets', 'proto')

        self.n_train = 1000000
        self.n_val = 8192

        self.N_CLASS = 100  # TODO: make it easier to change this parameter
        self.N_ORN = 50
        self.N_ORN_DUPLICATION = 1
        self.N_PN_PER_ORN = 1
        self.ORN_NOISE_STD = 0 #make sure this param is set to zero if N_ORN_PER_PN = 1
        self.N_KC = 2500

        self.percent_generalization = 100
        self.use_combinatorial = False
        self.n_combinatorial_classes = 20
        self.combinatorial_density = .3


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
        # If True, ORN --> PN connections are positive
        self.sign_constraint_pn2kc = True
        # If True, PN --> KC connections are trainable
        self.train_pn2kc = False
        # If True, PN --> KC connections are sparse
        self.sparse_pn2kc = True
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

        # label type can be either combinatorial, one_hot, sparse
        self.label_type = 'one_hot'

