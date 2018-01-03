import sys
import argparse

def str2bool(v):
  '''
  used to create special type for argparse
  '''
  return v.lower() in ('yes', 'true', 't', '1')

class Parameters():
    '''
    Keeps all the variables related to the present run of the experiment -
    which are usually set up with flags.
    '''

    def __init__(self):
        '''
        Sets up all the default values and then calls argsparse to get the
        flags set by the user.
        '''

        # Let us now update the values by parsing command line flags
        args = self._parse_flags()
        self._update_params(args)

    def _parse_flags(self):
        '''
        Provide flags interface to set each of the variables above.
        '''
        parser = argparse.ArgumentParser()

        parser.register('type', 'Bool', str2bool)

        parser.add_argument('-v', '--verbose', help='increase output \
                            verbosity',default=False,type='Bool')
        
        parser.add_argument('--induced_eer', help='',default=False,type='Bool')
        parser.add_argument('--vgan', help='',default=False,type='Bool')
        parser.add_argument('--aegan', help='',default=False,type='Bool')
        parser.add_argument('--wgan', help='',default=False,type='Bool')
        parser.add_argument('--prob_attack', help='',default=False,type='Bool')
        parser.add_argument('--attack_clusters', help='',default=1,type=int)
        parser.add_argument('--attack_num_tries', help='',default=100,type=int)
        parser.add_argument('--digraph_attack', help='',default=False,type='Bool')
        parser.add_argument('--digraph_attack_type', help='',
                    default='gaussian_mixture',type=str)
        parser.add_argument('--prob_then_kmeans', help='',
                    default=True,type='Bool')
        parser.add_argument('--prob_kmeans_sample_size', help='',
                    default=20000,type=int)
        parser.add_argument('--gaussian_components', help='',
                    default=2,type=int)

        parser.add_argument('--median_classifiers',
            help='Have multiple classifiers at FAR thresholds',default=False,type='Bool')

        parser.add_argument('--exclude_trained_samples', help='',default=True,type='Bool')
        # Classifiers
        parser.add_argument('--manhattan', help='use manhattan \
                        classifier',default=False,type='Bool')
        parser.add_argument('--scaled_manhattan', help='use scaled manhattan \
                        classifier',default=False,type='Bool')

        parser.add_argument('--svm', help='use svm classifier',
                             default=False,type='Bool')
        parser.add_argument('--ae', help='use ae classifier',
                            default=False,type='Bool')
        parser.add_argument('--var_ae', '--vae', help='use varae classifier',
                            default=False,type='Bool')
        parser.add_argument('--con_ae', '--cae', help='use ae classifier',
                            default=False,type='Bool')
        parser.add_argument('--random_forests', '--rf', help='use rf classifier',
                            default=False,type='Bool')
        parser.add_argument('--rf_trees', help='num of trees for rf classifier',
                            default=100, type=int)

        parser.add_argument('--pohmm', help='use ae classifier',
                            default=False,type='Bool')
        parser.add_argument('--gaussian', help='use gaussian classifier',
                            default=False,type='Bool')
        parser.add_argument('--knc', help='use KNN classifier',
                            default=False,type='Bool')
        parser.add_argument('--knc_n', help='num of trees for rf classifier',
                            default=2, type=int)
        parser.add_argument('--nearest_neighbors', help='use nearest neighbor classifier',
                            default=False,type='Bool')
        parser.add_argument('--gaussian_mixture', '--gmm', help='use this one',
                            default=False, type='Bool')

        parser.add_argument('--fc_net', help='use all given classifiers',
                            default=False,type='Bool')

        parser.add_argument('--ensemble', help='use all given classifiers',
                            default=False,type='Bool')

        parser.add_argument('--dbn',help='use dbn', default=False, type='Bool')
        parser.add_argument('--deng', help='use dbn', default=False, type='Bool')
        

        #TODO: Add support for --all flag.
        parser.add_argument('--quick_test', '--qt', help='run classifiers on only \
                            one user', default=False,type='Bool')

        parser.add_argument('--quick_test_num', '--qtn', help='run classifiers on only \
                            one user', default=5,type=int)

        parser.add_argument('--extract_attack_vectors', '--eav', help='extract\
                            attack vectors and their normalized versions in \
                            nice format', default=False,type='Bool')

        # Datasets
        # FIXME: Ask them to include full path
        parser.add_argument('--dataset', help='specify name of dataset file',
                             default='cmu.csv')

        parser.add_argument('--pickle', help='use ae classifier',
                             default=True,type='Bool')
        parser.add_argument('--swipes_num_users', help='num of android swipe \
                            users', default=100, type=int)
        parser.add_argument('--swipe_features', help='0: all features; 1: \
                            touch features; 2: gravity features.', default=0, type=int)
        parser.add_argument('--skip_fing_area', help='Skip mean fing \
                            area feature', default=False,type='Bool')
        parser.add_argument('--swipes_test_size', help='percentage of users \
                for test set', default=0.5,type=float)
         
        parser.add_argument('--eer_threshold', help='use eer threshold or not',
                             default=True,type='Bool')

        parser.add_argument('--mean_threshold', help='use eer threshold or not',
                             default=True,type='Bool')
        parser.add_argument('--median_threshold', help='use eer threshold or not',
                             default=True,type='Bool')

        parser.add_argument('--skip_inputs', help='', default=False,type='Bool')
        parser.add_argument('--feature_norm', help='Pass in arbitrary value \
                if you dont want to use stddev or minmax', default='stddev')

        parser.add_argument('--score_norm', help='', default='stddev')
        parser.add_argument('--seed', help='', default=1234)
        parser.add_argument('--kmeans_attack', help='',
                        default=False,type='Bool')
        parser.add_argument('--gan_attack', help='',
                        default=False,type='Bool')

        parser.add_argument('--attack_great_users', help='', default=True,type='Bool')
        parser.add_argument('--attack_bad_users', help='', default=True,type='Bool')
        parser.add_argument('--attack_ok_users', help='', default=True,type='Bool')

        parser.add_argument('--cracker', help='', default=False,type='Bool')
        parser.add_argument('--robustness_check', help='', default=False,type='Bool')
        parser.add_argument('--complete_check', help='', default=False,type='Bool')
        parser.add_argument('--shuffle_training_data', help='', default=False,type='Bool')

        parser.add_argument('--kmeans_cluster_size', '--kcs', help='', default=8,
                type=int)
        parser.add_argument('--kmeans_impostors', help='', default=50)
        parser.add_argument('--kmeans_impostor_samples', help='', default=400)

        # parser.add_argument('--impostor_samples', help='', default=200)

        args = parser.parse_args()        
        return args

    def _update_params(self, args):
        '''
        Lame way to assign all params to correct class objects - not as
        easy as I had hoped ugh. Mostly just an identity function.

        FIXME: Get rid of this stuff by using type cleverly in args itself.
        Then can return args instead of doing this crap - and that will
        essentially become self.params.
        '''

        self.pickle = args.pickle
        self.dataset = args.dataset

        self.android = False
        self.keystrokes = False
        self.mouse = False
        
        #TODO: This can be relegated to an update args function.
        if 'android' in self.dataset:
            self.android = True
            self.feature_norm = False
        elif 'cmu' in self.dataset or 'mturk' in self.dataset:
            self.keystrokes = True
        elif 'mouse' in self.dataset:
            self.mouse = True
        else:
            assert False, 'unrecognized dataset'
        
        # These are used for the digraph attack. Each element represents a key for which we need
        # timing samples.
        if self.dataset == 'mturk_mustang.csv':
            self.password_keys = ['m', 'mu', 'u', 'us', 's', 'st', 't', 'ta', 'a', 'an',
                            'n', 'ng', 'g']
            self.digraph_attack_file = 'data/prob_attack_data.json'
        elif self.dataset == 'mturk_password.csv':
            self.password_keys = ['p', 'pa', 'a', 'as', 's' , 'ss', 's', 'sw', 'w', 'wo', 'o', 'or', 'r', 'rd', 'd']
            self.digraph_attack_file = 'data/attack_password2.json' 
        elif self.dataset == 'mturk_letmein.csv':
            self.password_keys = ['l', 'le', 'e', 'et', 't', 'tm', 'm', 'me',
            'e', 'ei', 'i', 'in', 'n']
            self.digraph_attack_file = 'data/attack_letmein.json' 
        elif self.dataset == 'mturk_abc123.csv':
            self.password_keys = ['a', 'ab', 'b', 'bc', 'c', 'c1', '1', '12',
            '2', '23', '3']
            self.digraph_attack_file = 'data/attack_abc123.json' 
        elif self.dataset == 'mturk_123456789.csv':
            self.password_keys = ['1', '12', '2', '23', '3', '34', '4', '45',
            '5', '56', '6', '67', '7', '78', '8', '89', '9']
            self.digraph_attack_file = 'data/attack_123456789.json' 
        elif self.dataset == 'cmu.csv':
            self.password_keys = ['dot', 'dott', 't', 'ti', 'i', 'ie', 'e',
            'e5', '5', '5Shift', 'Shift', 'Shiftr', 'r', 'ro', 'o', 'oa', 'a',
            'an', 'n', 'nl', 'l']
            self.digraph_attack_file = 'data/attack_cmu.json' 
        else:
            self.password_keys = None

        self.swipe_all_features = False
        self.swipe_touch_features = False
        self.swipe_gravity_features = False

        if args.swipe_features == 0:
            self.swipe_all_features = True
        elif args.swipe_features == 1:
            self.swipe_touch_features = True
        elif args.swipe_features == 2:
            self.swipe_gravity_features = True
        else:
            assert 'False', 'swipe features have to be 0,1,2'
        self.skip_fing_area = args.skip_fing_area
        self.swipes_test_size = args.swipes_test_size

        self.swipes_num_users = args.swipes_num_users
  
        self.verbose = args.verbose

        self.eer_threshold = args.eer_threshold
        self.mean_threshold = args.mean_threshold
        self.median_threshold = args.median_threshold
        self.median_classifiers = args.median_classifiers
        
        self.digraph_attack = args.digraph_attack
        self.digraph_attack_type = args.digraph_attack_type
        self.prob_then_kmeans = args.prob_then_kmeans
        self.prob_kmeans_sample_size = args.prob_kmeans_sample_size
        self.gaussian_components = args.gaussian_components

        self.vgan = args.vgan
        self.wgan = args.wgan
        self.aegan = args.aegan
        self.prob_attack = args.prob_attack
        self.attack_clusters = args.attack_clusters
        self.attack_num_tries = args.attack_num_tries
        self.exclude_trained_samples = args.exclude_trained_samples

        self.manhattan = args.manhattan
        self.scaled_manhattan = args.scaled_manhattan
        self.svm = args.svm
        self.ae = args.ae
        self.var_ae = args.var_ae
        self.con_ae = args.con_ae
        self.random_forests = args.random_forests
        self.rf_trees = args.rf_trees
        self.pohmm = args.pohmm
        self.gaussian = args.gaussian
        self.knc = args.knc
        self.knc_neighbors = args.knc_n
        self.nearest_neighbors = args.nearest_neighbors
        self.gaussian_mixture = args.gaussian_mixture 
        self.ensemble = args.ensemble
        self.dbn = args.dbn
        self.deng = args.deng
        self.fc_net = args.fc_net

        self.shuffle_training_data = args.shuffle_training_data

        # self.impostor_samples = args.impostor_samples

        self.skip_inputs = args.skip_inputs

        self.skip_features = []  # Only for keystrokes
        self.add_features = False  
        
        if args.feature_norm == '0' or args.feature_norm == 'None':
            self.feature_norm = None
        elif args.feature_norm == 'stddev' or args.feature_norm == 'minmax':
            self.feature_norm = args.feature_norm
        else:
            print('invalid feature norm: ', args.feature_norm)
            assert False, 'invalid feature norm'

        if args.score_norm == '0' or args.score_norm == 'None':
            self.score_norm = None
        else:
            self.score_norm = args.score_norm

        self.seed = args.seed

        self.kmeans_attack = args.kmeans_attack
        self.gan_attack = args.gan_attack

        self.cracker = args.cracker

        self.kmeans_cluster_size = args.kmeans_cluster_size
        self.kmeans_impostors = args.kmeans_impostors
        self.kmeans_impostor_samples = args.kmeans_impostor_samples

        self.robustness_check = args.robustness_check

        self.sanity_genuine_results = []     # global arrays for stats
        self.sanity_impostor_results = []

        self.complete_check = args.complete_check

        self.attack_bad_users = args.attack_bad_users
        self.attack_ok_users = args.attack_ok_users
        self.attack_great_users = args.attack_great_users

        self.quick_test = args.quick_test
        self.quick_test_num = args.quick_test_num
        self.extract_attack_vectors = args.extract_attack_vectors

        self.num_features = 31
        self.skip_users = []
        
        # Will add all the string rep of classifiers after training to this
        # list so we can pass them around.
        self.classifiers_list = []

        self.induced_eer = args.induced_eer

