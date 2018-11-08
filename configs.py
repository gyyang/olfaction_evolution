import os

class input_ProtoConfig(object):
    def __init__(self):
        self.path = os.path.join(os.getcwd(), 'datasets', 'proto')

        self.n_train = 1000000
        self.n_val = 8192

        self.N_CLASS = 50  # TODO: make it easier to change this parameter
        self.N_ORN = 50
        self.N_ORN_PER_PN = 1
        self.N_PN_PER_ORN = 1
        self.ORN_NOISE_STD = 0 #make sure this param is set to zero if N_ORN_PER_PN = 1

        self.percent_generalization = 50
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
        self.data_dir = './datasets/proto/_100_generalization_onehot'
        self.model = 'full'
        self.save_path = './files/test'
        self.N_GLO = self.N_ORN * self.N_PN_PER_ORN
        self.N_KC = 2500

        self.lr = .001
        self.max_epoch = 10
        self.batch_size = 256
        # Whether PN --> KC connections are sparse
        self.sparse_pn2kc = True
        # Whether PN --> KC connections are trainable
        self.train_pn2kc = False
        # Whether to have direct glomeruli-like connections
        self.direct_glo = False
        # Whether the coefficient of the direct glomeruli-like connection
        # motif is trainable
        self.train_direct_glo = True
        # Whether to tradeoff the direct and random connectivity
        self.tradeoff_direct_random = False
        # Whether to impose all cross area connections are positive
        self.sign_constraint = True
        # Whether to have PN norm before non_linearity
        self.pn_norm_pre_nonlinearity = None
        self.norm_factor = .5
        # Whether to have PN norm after non_linearity
        self.pn_norm_post_nonlinearity = None

        # Whether to have KC norm before non_linearity
        self.kc_norm_pre_nonlinearity = None
        # Whether to have KC norm after non_linearity
        self.kc_norm_post_nonlinearity = None
        # dropout
        self.kc_dropout = True
        # label type can be either combinatorial, one_hot, sparse
        self.label_type = 'one_hot'

