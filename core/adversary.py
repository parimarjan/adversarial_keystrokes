from anomaly import *
from classify import *
from cracker import *
from sklearn.cluster import KMeans
import random
import math
from collections import defaultdict
import imp
from scipy.stats import norm
from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture

try:
    imp.find_module('torch')
    torch_installed = True
except ImportError:
    torch_installed = False

if torch_installed:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.autograd import Variable

# This is for swipes data
EXPERT_LEVEL_ATTACK = False

def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
        return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)
    
def mean(l):
    return sum(l) / float(len(l))
    

def extract(v):
    return v.data.storage().tolist()

def stats(d):
    return [np.mean(d), np.std(d)]

def decorate_with_diffs(data, exponent):
    return data

# This should be replaced completely.
def get_distribution_sampler(mu, sigma):
    return lambda n: torch.Tensor(np.random.normal(mu, sigma, (1, n)))  # Gaussian

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

class KernelDensityAdversary():
    '''
    General framework for an adversary that trains GANs to model probability
    distributions of each hold time, and time interval between keys.
    ''' 
    # DO a stupid loop on all parameters?
    def __init__(self, dataset, bandwidth, clusters=1, verbose=True, dist=norm):
        '''
        '''
        self.verbose = verbose
        self.clusters = clusters
        self.dataset = dataset

        self.params = defaultdict(list) 
        self.data = defaultdict(list)
        self.kd = KernelDensity(bandwidth=bandwidth)
 
    def add_data(self, data):
        '''
        @data: dictionary / list or whatever format of the data.
        @dataset_type: 'cmu_style' will mean the data format we use right now
        for keystrokes, but later can have parsers for free text or other
        sources.

            gen_case: data['a'] = samples.
                      data['ac'] = samples
                for each possible case.

        TODO: Fix this so we can keep adding data multiple times
        '''
        for k, samples in data.iteritems():
            self.data[k] = np.array(samples)
    
    def train(self):
        '''
        Don't even want to add pickling support here because this stuff happens
        fast enough.
        '''
        for k, samples in self.data.iteritems():

            # FIXME: Need to set such thresholds for mturk datasets
            if 'cmu' in self.dataset:
                samples = np.array([s for s in samples if s < 0.5])

            elif 'mturk' in self.dataset:
                samples = np.array([s for s in samples if float(s)/1000 < 0.5])

            samples = samples.reshape(-1, 1)

            # TODO: Optimize for bandwidth estimation
            self.kd.fit(samples)
     
    def _train(self, samples, k):
        '''
        '''

    def generate_attack(self, password_keys, num_samples=1):
        '''
        Given a random password string, will look into its data dictionary,
        sample fron distribution for each hold key and digraph and generate an
        attack vector.

        Note: Same code works fine for swipes too.
        TODO: Change variable names to reflect that.
        '''
        for i in range(num_samples):
            timing_vector = []    
            for key in password_keys:
                sample = self.kd.sample(1)
                timing_vector.append(sample[0][0])
        
        return [timing_vector] 

# FIXME: This is basically the same as ProbabilisticAdversary but we try
# weirder things here.
class SwipesProbabilisticAdversary():
    '''
    General framework for an adversary that trains GANs to model probability
    distributions of each hold time, and time interval between keys.
    ''' 
    # DO a stupid loop on all parameters?
    def __init__(self, dataset, clusters=1, verbose=True, dist=norm):
        '''
        '''
        self.verbose = verbose
        self.clusters = clusters
        self.dataset = dataset

        self.params = defaultdict(list) 
        self.data = defaultdict(list)
        self.dist = dist
 
    def add_data(self, data):
        '''
        @data: dictionary / list or whatever format of the data.
        @dataset_type: 'cmu_style' will mean the data format we use right now
        for keystrokes, but later can have parsers for free text or other
        sources.

            gen_case: data['a'] = samples.
                      data['ac'] = samples
                for each possible case.

        TODO: Fix this so we can keep adding data multiple times
        '''
        for k, samples in data.iteritems():
            self.data[k] = np.array(samples)
    
    def train(self):
        '''
        Don't even want to add pickling support here because this stuff happens
        fast enough.
        '''
        for k, samples in self.data.iteritems():

            # FIXME: Need to set such thresholds for mturk datasets
            if 'cmu' in self.dataset:
                samples = np.array([s for s in samples if s < 0.5])

            elif 'mturk' in self.dataset:
                samples = np.array([s for s in samples if float(s)/1000 < 0.5])

            samples = samples.reshape(-1, 1)
            kmeans = KMeans(n_clusters=self.clusters, random_state=0).fit(samples) 
            unique_labels = set(kmeans.labels_) 
            cluster_samples = []
            cluster_samples.append(samples)
            for l in unique_labels:
                samples = [s for i, s in enumerate(samples) if kmeans.labels_[i] == l]
                cluster_samples.append(samples)
            # This way when constructing attacks, we always take the
            # samples from the same cluster group.
            cluster_samples.sort(key = lambda x: len(x), reverse=True)
            
            for samples in cluster_samples:
                # change this to general version
                params = self.dist.fit(samples)
                self.params[k].append(params)
 
    def generate_attack(self, password_keys, num_samples=1):
        '''
        Given a random password string, will look into its data dictionary,
        sample fron distribution for each hold key and digraph and generate an
        attack vector.

        Note: Same code works fine for swipes too.
        TODO: Change variable names to reflect that.
        '''        
        # TODO: Modify this to generate num_samples samples. Or just do it
        # outside this loop.

        timing_vector = []
        for gan_num in range(self.clusters):
            timing_vector.append([])
            for key in password_keys:
                if self.params[key][gan_num] is None:
                    # then just use the other gan.
                    params = self.params[key][0]
                else:
                    params = self.params[key][gan_num]

                sample = np.random.normal(loc=params[0], scale=params[1], size=1) 
                timing_vector[gan_num].append(sample[0])


        return timing_vector

class ProbabilisticAdversary():
    '''
    General framework for an adversary that trains GANs to model probability
    distributions of each hold time, and time interval between keys.
    ''' 
    def __init__(self, dataset, clusters=1, verbose=True,
            attack_alg='gaussian_mixture', components=2):
        '''
        TODO: Accept password_keys so we can optimize based on that.
        '''
        self.verbose = verbose
        self.clusters = clusters
        self.dataset = dataset

        self.data = defaultdict(list)
        self.attack_alg = attack_alg

        if attack_alg == 'gaussian':
            self.dist = norm
            self.params = {} 
        elif attack_alg == 'gaussian_mixture':
            print('attack alg of gaussian mixture type!')
            # don't need to do anything here.
            self.gms = {}
            self.gaussian_components = components
 
    def add_data(self, data):
        '''
        @data: dictionary / list or whatever format of the data.

            gen_case: data['a'] = samples.
                      data['ac'] = samples
                for each possible case.

        TODO: Fix this so we can keep adding data multiple times
        '''
        for k, samples in data.iteritems():
            if self.dataset == 'cmu.csv':
                # divide samples by 1000
                for i, s in enumerate(samples):
                    samples[i] = float(s)/1000;
            
            self.data[k] = np.array(samples)

    def train(self):
        '''
        Don't even want to add pickling support here because this stuff happens
        fast enough.

        I had tried a more general version with clustering before fitting
        distribution but that never got me better results so just got rid of
        it.
        '''
        for k, samples in self.data.iteritems(): 
            if len(samples) < 20:
                continue
            # Getting rid of outliers improves performance.
            if 'cmu' in self.dataset:
                samples = np.array([s for s in samples if s < 0.5])
            elif 'mturk' in self.dataset:
                samples = np.array([s for s in samples if float(s)/1000 < 0.5]) 
            
            if self.attack_alg == 'gaussian':
                params = self.dist.fit(samples)
                self.params[k] = params
            elif self.attack_alg == 'gaussian_mixture':
                samples = samples.reshape(-1, 1)
                self.gms[k]=GaussianMixture(n_components=self.gaussian_components)
                self.gms[k].fit(samples)
        
    def generate_attack(self, password_keys, num_samples=1):
        '''
        Given a random password string, will look into its data dictionary,
        sample fron distribution for each hold key and digraph and generate an
        attack vector.

        Note: Same code works fine for swipes too.
        TODO: Change variable names to reflect that.
        ''' 
        all_samples = []
        for i in range(num_samples):
            timing_vector = [] 
            for key in password_keys:
                if self.attack_alg == 'gaussian':
                    params = self.params[key]
                    sample = np.random.normal(loc=params[0], scale=params[1], size=1) 
                    timing_vector.append(sample[0])
                elif self.attack_alg == 'gaussian_mixture':
                    sample = self.gms[key].sample(n_samples=1)
                    timing_vector.append(sample[0][0][0])
            
            # print('timing vector: ', timing_vector)
            all_samples.append(timing_vector) 

        return all_samples

# TODO: Have an overlying class from which each of these classes inherit.
class AdversarialAE():
    '''
    General framework for an adversary that trains GANs to model probability
    distributions of each hold time, and time interval between keys.
    '''
    def __init__(self, verbose=True):
        '''
        '''
        self.verbose = verbose
        # key is the digraph pair, or the single key value - and the value will
        # be a trained network (just the parameter values of a network should
        # suffice)
        # This will map each char or digraph pair as keys to a list of values
        # for that particular guy
        self.data = defaultdict(list) 
        self.adv_samples = defaultdict(list)

        # all the zillion GAN parameters
        self.mb_size = 32
        self.X_dim = 1
        self.y_dim = 1
        self.z_dim = 10          # ? - how many inputs generator creates in one time.
        self.h_dim = 128         # hidden layer dim?
        # self.lam = 10            # ?
        # self.n_disc = 5          # ?
        self.lr = 1e-4
        self.c = 0
        self.num_epochs = 100000
 
    def add_data(self, data):
        '''
        @data: dictionary / list or whatever format of the data.
        @dataset_type: 'cmu_style' will mean the data format we use right now
        for keystrokes, but later can have parsers for free text or other
        sources.

            gen_case: data['a'] = samples.
                      data['ac'] = samples
                for each possible case.

        TODO: Fix this so we can keep adding data multiple times
        '''
        self.data = data 
        print('num of keys in data is ', len(data))  
        
    def train_gans(self):
        '''
        Will train a gan for each 'key' guy in the data dictionary using G, D. 
        The loop is essentially identical to the one used in other samples -
        and the dimensionality of the input is small, so its nice.

        Will save the parameter values (pickle)
        '''
        # TODO: Add pickling support here.
        file_name = self._gen_pickle_name_gans()
        print('file name is ', file_name) 
        if os.path.isfile(file_name):
            with open(file_name, 'rb') as handle:
                self.adv_samples = pickle.load(handle)
                if self.verbose:
                    print("loaded file ", file_name, "from disk") 
         
        num_keys = len(self.adv_samples)
        print('num keys already in file are: ', num_keys)

        for k in self.data:
            # k is the char, or digraph guy
            if k not in self.adv_samples:
                samples = self._train_gan(k)
                self.adv_samples[k] = samples

                # FUCK TENSORFLOW
                # file_name = self._gen_tf_name()
                # saver = tf.train.Saver()
                # with tf.Session() as sess:
                    # print('file name is ', file_name)
                    # saver.restore(sess, file_name)
                    # print('restored successfully')
                    # for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
                        # print(v)
                    # sess.run(tf.global_variables_initializer()) 
        
        # Update saved pickle if needed
        if len(self.adv_samples) > num_keys:
            with open(file_name, 'w+') as handle:
                pickle.dump(self.adv_samples, handle, protocol=pickle.HIGHEST_PROTOCOL)
                if self.verbose:
                    print("saved ", file_name, "on disk")
 
    def _train_gan(self, k):
        '''
        We get a shit ton of samples for this particular digraph etc and now we
        work on training the gan. Returns the GAN object (which is a class)

        @k: is the key in the main dict.
        @ret: is the trained GAN
        '''
        print('going to start training the gans!')
        
        
        """ Q(z|X) """
        X = tf.placeholder(tf.float32, shape=[None, self.X_dim])
        z = tf.placeholder(tf.float32, shape=[None, self.z_dim])

        Q_W1 = tf.Variable(xavier_init([self.X_dim, self.h_dim]))
        Q_b1 = tf.Variable(tf.zeros(shape=[self.h_dim]))

        Q_W2 = tf.Variable(xavier_init([self.h_dim, self.z_dim]))
        Q_b2 = tf.Variable(tf.zeros(shape=[self.z_dim]))

        theta_Q = [Q_W1, Q_W2, Q_b1, Q_b2]

        def Q(X):
            h = tf.nn.relu(tf.matmul(X, Q_W1) + Q_b1)
            z = tf.matmul(h, Q_W2) + Q_b2
            return z

        """ P(X|z) """
        P_W1 = tf.Variable(xavier_init([self.z_dim, self.h_dim]))
        P_b1 = tf.Variable(tf.zeros(shape=[self.h_dim]))

        P_W2 = tf.Variable(xavier_init([self.h_dim, self.X_dim]))
        P_b2 = tf.Variable(tf.zeros(shape=[self.X_dim]))

        theta_P = [P_W1, P_W2, P_b1, P_b2]

        def P(z):
            h = tf.nn.relu(tf.matmul(z, P_W1) + P_b1)
            logits = tf.matmul(h, P_W2) + P_b2
            prob = tf.nn.sigmoid(logits)
            return prob, logits


        """ D(z) """
        D_W1 = tf.Variable(xavier_init([self.z_dim, self.h_dim]))
        D_b1 = tf.Variable(tf.zeros(shape=[self.h_dim]))

        D_W2 = tf.Variable(xavier_init([self.h_dim, 1]))
        D_b2 = tf.Variable(tf.zeros(shape=[1]))

        theta_D = [D_W1, D_W2, D_b1, D_b2]

        def D(z):
            h = tf.nn.relu(tf.matmul(z, D_W1) + D_b1)
            logits = tf.matmul(h, D_W2) + D_b2
            prob = tf.nn.sigmoid(logits)
            return prob

        """ Training """
        z_sample = Q(X)
        _, logits = P(z_sample)

        # Sample from random z
        X_samples, _ = P(z)

        # E[log P(X|z)]
        recon_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
            targets=X))

        # Adversarial loss to approx. Q(z|X)
        D_real = D(z)
        D_fake = D(z_sample)

        D_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1. - D_fake))
        G_loss = -tf.reduce_mean(tf.log(D_fake))

        AE_solver = tf.train.AdamOptimizer().minimize(recon_loss, var_list=theta_P + theta_Q)
        D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
        G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_Q)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        if not os.path.exists('out/'):
            os.makedirs('out/')

        # i = 0
        for it in range(10000):
            # X_mb, _ = mnist.train.next_batch(mb_size)
            X_mb = random.sample(self.data[k], self.mb_size)
            X_mb = np.array(X_mb)
            X_mb = np.reshape(X_mb, (len(X_mb), 1))

            z_mb = np.random.randn(self.mb_size, self.z_dim)

            _, recon_loss_curr = sess.run([AE_solver, recon_loss], feed_dict={X: X_mb})
            _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: X_mb, z: z_mb})
            _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={X: X_mb})

            if it % 1000 == 0:
                print('key was: ', k)
                print('Iter: {}; D_loss: {:.4}; G_loss: {:.4}; Recon_loss: {:.4}'
                      .format(it, D_loss_curr, G_loss_curr, recon_loss_curr))

                samples = sess.run(X_samples, feed_dict={z: np.random.randn(16,
                    self.z_dim)})
                # print('samples: ', samples)
                print('for key ', k, 'mean of samples was')
                print(np.mean(samples))
                print('mean of X_mb was: ')
                print(np.mean(X_mb))

        # save samples for future use.
        samples = sess.run(X_samples, feed_dict={z: np.random.randn(16, self.z_dim)})
        # samples = sess.run(G_sample, feed_dict={z: sample_z(500, self.z_dim)})
        print('samples: ', samples)
        print('samples[0]: ', samples[0])
        print('for key ', k, 'mean of samples was')
        print(np.mean(samples))
        print('for key ', k, 'mean of orig data is')
        print(np.mean(self.data[k]))
        return samples

        # Time to save the session here using tf.save stuff
        # file_name = self._gen_tf_name()
        # oSaver.save(oSess, file_name)  #filename ends with .ckpt
        # print('saved tensorflow model!')
        # return None

    def _gen_tf_name(self):
        '''
        TODO: include hash of feature vectors here too.
        '''
        name = '/tmp/test.ckpt'
        return name

    def _gen_pickle_name_gans(self):
        '''
        TODO: include hash of feature vectors here too.
        '''
        # variables = [i for i in dir(self) if not callable(i)]
        class_data = []
        variables = self.__dict__.keys()
        print('variables are ', variables)
        for attr_name in variables:
            if hasattr(self, attr_name):
               class_data.append(getattr(self, attr_name))
        
        # keys = [k for k in self.gans]
        unique_id = hashlib.sha1(str(class_data)).hexdigest()[0:5]
        print('unique id is ', unique_id)
        name = 'pickle/' 'gan_samples' + unique_id + '.pickle'
        return name

    def generate_attack(self, password_keys):
        '''
        Given a random password string, will look into its data dictionary,
        sample fron distribution for each hold key and digraph and generate an
        attack vector.

        Note: Same code works fine for swipes too.
        TODO: Change variable names to reflect that.
        '''
        # gi_sampler = self.get_generator_input_sampler()        
        # gen_input = Variable(gi_sampler(1, self.g_input_size))
        timing_vector = []
        for key in password_keys:
            if len(self.adv_samples[key]) == 0:
                assert False, 'not trained on this key!'
            sample = random.sample(self.adv_samples[key], 1)
            # print('sample is ', sample)
            # print('sample[0][0] is ', sample[0][0])
            timing_vector.append(sample[0][0])
        
        return timing_vector

class WGANAdversary():
    '''
    General framework for an adversary that trains GANs to model probability
    distributions of each hold time, and time interval between keys.
    '''
    def __init__(self, verbose=True):
        '''
        '''
        self.verbose = verbose
        # key is the digraph pair, or the single key value - and the value will
        # be a trained network (just the parameter values of a network should
        # suffice)
        # This will map each char or digraph pair as keys to a list of values
        # for that particular guy
        self.data = defaultdict(list) 
        self.adv_samples = defaultdict(list)

        # all the zillion GAN parameters
        self.mb_size = 32
        self.X_dim = 1
        self.z_dim = 10          # ? - how many inputs generator creates in one time.
        self.h_dim = 128         # hidden layer dim?
        self.lam = 10            # ?
        self.n_disc = 5          # ?
        self.lr = 1e-4
        self.num_epochs = 100000
 
    def add_data(self, data):
        '''
        @data: dictionary / list or whatever format of the data.
        @dataset_type: 'cmu_style' will mean the data format we use right now
        for keystrokes, but later can have parsers for free text or other
        sources.

            gen_case: data['a'] = samples.
                      data['ac'] = samples
                for each possible case.

        TODO: Fix this so we can keep adding data multiple times
        '''
        self.data = data 
        print('num of keys in data is ', len(data))  
        
    def train_gans(self):
        '''
        Will train a gan for each 'key' guy in the data dictionary using G, D. 
        The loop is essentially identical to the one used in other samples -
        and the dimensionality of the input is small, so its nice.

        Will save the parameter values (pickle)
        '''
        # TODO: Add pickling support here.
        file_name = self._gen_pickle_name_gans()
        print('file name is ', file_name) 
        if os.path.isfile(file_name):
            with open(file_name, 'rb') as handle:
                self.adv_samples = pickle.load(handle)
                if self.verbose:
                    print("loaded file ", file_name, "from disk") 
         
        num_keys = len(self.adv_samples)
        print('num keys already in file are: ', num_keys)

        for k in self.data:
            # k is the char, or digraph guy
            if k not in self.adv_samples:
                samples = self._train_gan(k)
                self.adv_samples[k] = samples

                # FUCK TENSORFLOW
                # file_name = self._gen_tf_name()
                # saver = tf.train.Saver()
                # with tf.Session() as sess:
                    # print('file name is ', file_name)
                    # saver.restore(sess, file_name)
                    # print('restored successfully')
                    # for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
                        # print(v)
                    # sess.run(tf.global_variables_initializer()) 
        
        # Update saved pickle if needed
        if len(self.adv_samples) > num_keys:
            with open(file_name, 'w+') as handle:
                pickle.dump(self.adv_samples, handle, protocol=pickle.HIGHEST_PROTOCOL)
                if self.verbose:
                    print("saved ", file_name, "on disk")
 
    def _train_gan(self, k):
        '''
        We get a shit ton of samples for this particular digraph etc and now we
        work on training the gan. Returns the GAN object (which is a class)

        @k: is the key in the main dict.
        @ret: is the trained GAN
        '''
        print('going to start training the gans!')
        
        # Various training parameters...
        X = tf.placeholder(tf.float32, shape=[None, self.X_dim])

        D_W1 = tf.Variable(xavier_init([self.X_dim, self.h_dim]), name='dw1')
        D_b1 = tf.Variable(tf.zeros(shape=[self.h_dim]))

        D_W2 = tf.Variable(xavier_init([self.h_dim, 1]))
        D_b2 = tf.Variable(tf.zeros(shape=[1]))

        theta_D = [D_W1, D_W2, D_b1, D_b2]

        z = tf.placeholder(tf.float32, shape=[None, self.z_dim])

        G_W1 = tf.Variable(xavier_init([self.z_dim, self.h_dim]))
        G_b1 = tf.Variable(tf.zeros(shape=[self.h_dim]))

        G_W2 = tf.Variable(xavier_init([self.h_dim, self.X_dim]))
        G_b2 = tf.Variable(tf.zeros(shape=[self.X_dim]))
        
        theta_G = [G_W1, G_W2, G_b1, G_b2]
        
        # Change the np.random range to min / max of dataset? Or just leave it like
        # this for now.
        def sample_z(m, n):
            return np.random.uniform(-1., 1., size=[m, n])

        # This should probably all be the same.
        def G(z):
            G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
            G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
            G_prob = tf.nn.sigmoid(G_log_prob)
            return G_prob

        def D(X):
            D_h1 = tf.nn.relu(tf.matmul(X, D_W1) + D_b1)
            out = tf.matmul(D_h1, D_W2) + D_b2
            return out

        G_sample = G(z)
        D_real = D(X)
        D_fake = D(G_sample)

        eps = tf.random_uniform([self.mb_size, 1], minval=0., maxval=1.)
        X_inter = eps*X + (1. - eps)*G_sample
        grad = tf.gradients(D(X_inter), [X_inter])[0]
        grad_norm = tf.sqrt(tf.reduce_sum((grad)**2, axis=1))
        grad_pen = self.lam * tf.reduce_mean(grad_norm - 1.)**2

        D_loss = tf.reduce_mean(D_fake) - tf.reduce_mean(D_real) + grad_pen
        G_loss = -tf.reduce_mean(D_fake)

        D_solver = (tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.5)
                    .minimize(D_loss, var_list=theta_D))
        G_solver = (tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.5)
                    .minimize(G_loss, var_list=theta_G))

        # sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        
        oSaver = tf.train.Saver()
        oSess = sess

        # if not os.path.exists('out/'):
            # os.makedirs('out/')

        i = 0
        for it in range(self.num_epochs):
            for _ in range(self.n_disc):
                # X_mb, _ = mnist.train.next_batch(self.mb_size)
                # TODO: just use np.random.sample instead.
                X_mb = random.sample(self.data[k], self.mb_size)
                X_mb = np.array(X_mb)
                X_mb = np.reshape(X_mb, (len(X_mb), 1))

                _, D_loss_curr = sess.run(
                    [D_solver, D_loss],
                    feed_dict={X: X_mb, z: sample_z(self.mb_size, self.z_dim)}
                )

            _, G_loss_curr = sess.run(
                [G_solver, G_loss],
                feed_dict={z: sample_z(self.mb_size, self.z_dim)}
            )

            if it % 1000 == 0:
                print('key was: ', k)
                print('Iter: {}; D loss: {:.4}; G_loss: {:.4}'
                      .format(it, D_loss_curr, G_loss_curr))

                if it % 10000 == 0:
                    samples = sess.run(G_sample, feed_dict={z: sample_z(16,
                        self.z_dim)}) 
                    print('samples: ', samples)
                    print('for key ', k, 'mean of samples was')
                    print(np.mean(samples))

                    # WTF is being plotted here?
                    # fig = plot(samples)
                    i += 1
        
        samples = sess.run(G_sample, feed_dict={z: sample_z(500, self.z_dim)})

        print('samples: ', samples)
        print('samples[0]: ', samples[0])
        print('for key ', k, 'mean of samples was')
        print(np.mean(samples))
        print('for key ', k, 'mean of orig data is')
        print(np.mean(self.data[k]))
        return samples

        # Time to save the session here using tf.save stuff
        # file_name = self._gen_tf_name()
        # oSaver.save(oSess, file_name)  #filename ends with .ckpt
        # print('saved tensorflow model!')
        # return None

    def _gen_tf_name(self):
        '''
        TODO: include hash of feature vectors here too.
        '''
        name = '/tmp/test.ckpt'
        return name

    def _gen_pickle_name_gans(self):
        '''
        TODO: include hash of feature vectors here too.
        '''
        # variables = [i for i in dir(self) if not callable(i)]
        class_data = []
        variables = self.__dict__.keys()
        print('variables are ', variables)
        for attr_name in variables:
            if hasattr(self, attr_name):
               class_data.append(getattr(self, attr_name))
        
        # keys = [k for k in self.gans]
        unique_id = hashlib.sha1(str(class_data)).hexdigest()[0:5]
        print('unique id is ', unique_id)
        name = 'pickle/' 'gan_samples' + unique_id + '.pickle'
        return name

    def generate_attack(self, password_keys):
        '''
        Given a random password string, will look into its data dictionary,
        sample fron distribution for each hold key and digraph and generate an
        attack vector.

        Note: Same code works fine for swipes too.
        TODO: Change variable names to reflect that.
        '''
        # gi_sampler = self.get_generator_input_sampler()        
        # gen_input = Variable(gi_sampler(1, self.g_input_size))
        timing_vector = []
        for key in password_keys:
            if len(self.adv_samples[key]) == 0:
                assert False, 'not trained on this key!'
            sample = random.sample(self.adv_samples[key], 1)
            # print('sample is ', sample)
            # print('sample[0][0] is ', sample[0][0])
            timing_vector.append(sample[0][0])
        
        return timing_vector

class GANAdversary():
    '''
    General framework for an adversary that trains GANs to model probability
    distributions of each hold time, and time interval between keys.
    '''
    
    # DO a stupid loop on all parameters?
    def __init__(self, G, D, verbose=True, hidden_size=100, mb_size=100,
            d_lr=2e-4, g_lr=2e-4, o_betas1=0.9, o_betas2=0.999,
            num_epochs=40000, g_steps=1, d_steps=1, clusters=1):
        '''
        TODO: Set it up in a more general fashion like this:
            @G: Generator function.
            @D: Discriminator function.
            There is a lot of scope for experimentation with these two. 
            Will also need to pass in the args required for these.

        We can iteratively add newer 'datasets' to the gan dictionary.
        '''
        self.G = G
        self.D = D
         
        self.verbose = verbose
        # key is the digraph pair, or the single key value - and the value will
        # be a trained network (just the parameter values of a network should
        # suffice)
        self.gans = defaultdict(list) 
        # This will map each char or digraph pair as keys to a list of values
        # for that particular guy
        self.data = defaultdict(list)

        # all the zillion GAN parameters
        self.g_input_size = 1    
        self.g_hidden_size = hidden_size  
        self.g_output_size = 1    
        self.d_input_size = mb_size   # Minibatch size - cardinality of distributions
        self.d_hidden_size = hidden_size 
        self.d_output_size = 1    # Single dimension for 'real' vs. 'fake'
        self.minibatch_size = self.d_input_size

        # Temporary stuff:
        self.d_learning_rate = d_lr  # 2e-4
        self.g_learning_rate = g_lr

        # what's this for?
        self.optim_betas = (o_betas1, o_betas2) 
        self.num_epochs = num_epochs
        self.print_interval = 2000
        self.d_steps = d_steps  # 'k' steps in the original GAN paper. Can put the discriminator on higher training freq than generator
        self.g_steps = g_steps
        self.clusters = clusters
 
    def get_generator_input_sampler(self):
        return lambda m, n: torch.rand(m, n)  # Uniform-dist data into generator, _NOT_ Gaussian

    def add_data(self, data):
        '''
        @data: dictionary / list or whatever format of the data.
        @dataset_type: 'cmu_style' will mean the data format we use right now
        for keystrokes, but later can have parsers for free text or other
        sources.

            gen_case: data['a'] = samples.
                      data['ac'] = samples
                for each possible case.

        TODO: Fix this so we can keep adding data multiple times
        '''
        for k, samples in data.iteritems():
            self.data[k] = np.array(samples)
    
    def train_gans(self):
        '''
        Will train a gan for each 'key' guy in the data dictionary using G, D. 
        The loop is essentially identical to the one used in other samples -
        and the dimensionality of the input is small, so its nice.

        Will save the parameter values (pickle)
        '''
        # TODO: Add pickling support here.
        file_name = self._gen_pickle_name_gans() 
        if os.path.isfile(file_name):
            with open(file_name, 'rb') as handle:
                self.gans = pickle.load(handle)
                if self.verbose:
                    print("loaded file ", file_name, "from disk") 
         
        num_keys = len(self.gans)
        for k in self.data:
            # k is the char, or digraph guy
            if k not in self.gans:
                # now cluster them first and then send each cluster in.
                samples = self.data[k].reshape(-1, 1)
                kmeans = KMeans(n_clusters=self.clusters, random_state=0).fit(samples) 
                unique_labels = set(kmeans.labels_) 
                cluster_samples = []
                for l in unique_labels:
                    samples = [s for i, s in enumerate(self.data[k]) if kmeans.labels_[i] == l]
                    cluster_samples.append(samples)
                # This way when constructing attacks, we always take the
                # samples from the same cluster group.
                cluster_samples.sort(key = lambda x: len(x), reverse=True)
                
                for samples in cluster_samples:
                    if len(samples) > 1000:
                        self.gans[k].append(self._train_gan(samples, k))
                    else: 
                        self.gans[k].append(None)

        # Update saved pickle if needed
        if len(self.gans) > num_keys:
            with open(file_name, 'w+') as handle:
                pickle.dump(self.gans, handle, protocol=pickle.HIGHEST_PROTOCOL)
                if self.verbose:
                    print("saved ", file_name, "on disk")

        # gi_sampler = self.get_generator_input_sampler()        
        # gen_input = Variable(gi_sampler(self.minibatch_size, self.g_input_size))
        # g_fake_data = self.gans['2'](gen_input) 
        # g_fake_data = g_fake_data.data.numpy()
        
    def _gen_pickle_name_gans(self):
        '''
        TODO: include hash of feature vectors here too.
        '''
        # variables = [i for i in dir(self) if not callable(i)]
        class_data = []
        variables = self.__dict__.keys()
        print('variables are ', variables)
        for attr_name in variables:
            if hasattr(self, attr_name):
               class_data.append(getattr(self, attr_name))
        
        # keys = [k for k in self.gans]
        unique_id = hashlib.sha1(str(class_data)).hexdigest()[0:5]
        print('unique id is ', unique_id)
        name = 'pickle/' + unique_id + '.pickle'
        return name

    def _d_sampler(self, samples, input_size):
        ''' 
        '''
        # Turn this into a matrix
        x = np.array(random.sample(samples, input_size))
        x = np.reshape(x, (1, input_size))
        return torch.Tensor(x)

    def _train_gan(self, samples, k):
        '''
        We get a shit ton of samples for this particular digraph etc and now we
        work on training the gan. Returns the GAN object (which is a class)

        @k: is the key in the main dict.
        '''
        print('going to start training the gans!')
        
        # This is super weird
        # d_input_func = lambda x : x**2

        d_sampler = self._d_sampler
        #FIXME: This feels weird
        gi_sampler = self.get_generator_input_sampler()

        G = Generator(input_size=self.g_input_size,
                hidden_size=self.g_hidden_size, output_size=self.g_output_size)

        D = Discriminator(input_size=self.minibatch_size,
                hidden_size=self.d_hidden_size, output_size=self.d_output_size)

        # These will go into the train function.
        criterion = nn.BCELoss()  # Binary cross entropy: http://pytorch.org/docs/nn.html#bceloss
        d_optimizer = optim.Adam(D.parameters(), lr=self.d_learning_rate, betas=self.optim_betas)
        g_optimizer = optim.Adam(G.parameters(), lr=self.g_learning_rate, betas=self.optim_betas)
        preprocess = lambda data: decorate_with_diffs(data, 2.0) 
        for epoch in range(self.num_epochs):
            for d_index in range(self.d_steps):
                # 1. Train D on real+fake
                D.zero_grad()
                #  1A: Train D on real
                d_real_data = Variable(d_sampler(samples, self.d_input_size))
                d_real_decision = D(preprocess(d_real_data))
                d_real_error = criterion(d_real_decision, Variable(torch.ones(1)))  # ones = true
                d_real_error.backward() # compute/store gradients, but don't change params
                #  1B: Train D on fake
                d_gen_input = Variable(gi_sampler(self.minibatch_size, self.g_input_size))

                d_fake_data = G(d_gen_input).detach()  # detach to avoid training G on these labels
                d_fake_decision = D(preprocess(d_fake_data.t()))
                d_fake_error = criterion(d_fake_decision, Variable(torch.zeros(1)))  # zeros = fake
                d_fake_error.backward()
                d_optimizer.step()     # Only optimizes D's parameters; changes based on stored gradients from backward()

            for g_index in range(self.g_steps):
                # 2. Train G on D's response (but DO NOT train D on these labels)
                G.zero_grad()
                gen_input = Variable(gi_sampler(self.minibatch_size, self.g_input_size))
                g_fake_data = G(gen_input)
                dg_fake_decision = D(preprocess(g_fake_data.t()))
                g_error = criterion(dg_fake_decision, Variable(torch.ones(1)))  # we want to fool, so pretend it's all genuine

                g_error.backward()
                g_optimizer.step()  # Only optimizes G's parameters

            if epoch % self.print_interval == 0 and self.verbose:
                print("%s: D: %s/%s G: %s (Real: %s, Fake: %s) " % (epoch,
                                                                    extract(d_real_error)[0],
                                                                    extract(d_fake_error)[0],
                                                                    extract(g_error)[0],
                                                                    stats(extract(d_real_data)),
                                                                    stats(extract(d_fake_data))))
                
        # save samples for future use.
        # samples = sess.run(X_samples, feed_dict={z: np.random.randn(16, self.z_dim)})
        # samples = sess.run(G_sample, feed_dict={z: sample_z(500, self.z_dim)})

        # gen_input = Variable(gi_sampler(16, self.g_input_size))
        # samples = G(gen_input).numpy()
        # print('samples: ', samples)
        # print('samples[0]: ', samples[0])
        # print('for key ', k, 'mean of samples was')
        # print(np.mean(samples))
        # print('for key ', k, 'mean of orig data is')
        # print(np.mean(self.data[k]))

        return G

    def generate_attack(self, password_keys):
        '''
        Given a random password string, will look into its data dictionary,
        sample fron distribution for each hold key and digraph and generate an
        attack vector.

        Note: Same code works fine for swipes too.
        TODO: Change variable names to reflect that.
        '''
        gi_sampler = self.get_generator_input_sampler()        
        timing_vector = []
        gen_input = Variable(gi_sampler(1, self.g_input_size))
        
        for gan_num in range(self.clusters):
            timing_vector.append([])
            for key in password_keys:
                if self.gans[key][gan_num] is None:
                    # then just use the other gan.
                    gan = self.gans[key][0]
                else:
                    gan = self.gans[key][gan_num]

                sample = gan(gen_input) 
                sample = sample.data.numpy()[0]
                timing_vector[gan_num].append(sample[0])
        
        return timing_vector

# ##### MODELS: Generator model and discriminator model
if torch_installed:
    class Generator(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(Generator, self).__init__()
            self.map1 = nn.Linear(input_size, hidden_size)
            self.map2 = nn.Linear(hidden_size, hidden_size)
            self.map3 = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            x = F.elu(self.map1(x))
            x = F.sigmoid(self.map2(x))
            return self.map3(x)

    class Discriminator(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(Discriminator, self).__init__()
            self.map1 = nn.Linear(input_size, hidden_size)
            self.map2 = nn.Linear(hidden_size, hidden_size)
            self.map3 = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            x = F.elu(self.map1(x))
            x = F.elu(self.map2(x))
            return F.sigmoid(self.map3(x))


class Adversary(object):
    """
    Which user the attack is conducted against is passed in for the answer.  
    """

    def __init__(self, data, exp, expert_level=None):
        """
        @data: dictionary of users - and their features as values
        @exp: experiment class object - we use params and stats from there.

        @expert_level: just used in swipes - might as well get rid of it?
        """
        self.data = data
        self.params = exp.params
        self.exp = exp

        self.swipes_expert_level = expert_level
    
    def generate_random_attack(self, user):
        """
        Will just sample randomly from all possible things from data.
        Note: have already done essentially this in sanity check...
        """
        assert False, 'not implemented'
    
    def _construct_cluster(self, user,clusters=16, num_impostors=50,
            num_impostor_samples=400, given_users=None):
        """
            Just a wrapper function...
        """ 

        count_users = 0
        X = []

        for d in self.data:
            # skip current user obviously.
            if d == user:
                continue
            if count_users >= num_impostors:
                break
            
            # Support for giving an arbitrary list of users to use for
            # constructing the attack
            if given_users is not None and d not in given_users:
                continue
            
            count_samples = 0

            sel = random.sample(self.data[d], num_impostor_samples)
            for x in sel:
                X.append(x)
            
            count_users += 1 
        
        X = np.array(X)

        file_name = self.gen_pickle_name(X, user, clusters)

        if os.path.isfile(file_name) and self.params.pickle:
            # Then just read it in!
            with open(file_name, 'rb') as handle:
                clusters = pickle.load(handle)
                if self.params.verbose:
                    print("loaded file ", file_name, "from disk") 

        else:
            # Assuming default values for rest are good. Might want to experiment
            # with this as well.
            kmeans = KMeans(n_clusters=clusters, random_state=0).fit(X) 
            clusters = kmeans.cluster_centers_
            
            if self.params.pickle:
                with open(file_name, 'w+') as handle:
                    pickle.dump(clusters, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    if self.params.verbose:
                        print("saved ", file_name, "on disk")

        return clusters

    def kmeans_attack(self, user, clusters=16, num_impostors=50,
            num_impostor_samples=400, given_users=None):
        """
        Compute clusters based on all the non-user's samples.

        @num_impostors: Data from how many impostors do we use.

        FIXME: Right now, we just take the first num_impostors samples right
        out of 'data'...I guess we could randomize it, but I don't see why it
        makes a big deal.

        FIXME: Implement the given_users functionality.

        Note: We need to train each of these individually for every user
        because we are skipping adding their data to the mix.
        """
        # Usual case.
        if self.swipes_expert_level is None:
            return self._construct_cluster(user, clusters=clusters,
                    num_impostors=num_impostors,num_impostor_samples=num_impostor_samples,
                    given_users=given_users)

        # This part is just weird...I had been too inspired and wasted a
        # couple of hours on trying this, but doesn't really seem to work.
        else:
            if EXPERT_LEVEL_ATTACK:
                # We want to do something smarter here. Let's say there are 5
                # expert levels. So for each of them, we want to construct a
                # cluster of 8 options, and then attempt to break the swipe
                # defence...
                
                clusters = np.empty((0,11),dtype='float64')
                for i in range(1,6,1):
                    level = str(i)
                    given_users = []
                    for y in self.swipes_expert_level:
                        if self.swipes_expert_level[y] == level:
                            given_users.append(y)
                    
                    _clusters = self._construct_cluster(user, clusters=8,
                        num_impostors=num_impostors,num_impostor_samples=num_impostor_samples,
                        given_users=given_users)
                    
                    # FIXME: Stupid hack this should never be happening...
                    if _clusters.shape == (8,10):
                        break
                    clusters = np.vstack((clusters, _clusters))
                 
                # assert len(clusters) == 8*5
                return clusters

            else:
                return self._construct_cluster(user, clusters=clusters,
                        num_impostors=num_impostors,num_impostor_samples=num_impostor_samples,
                        given_users=given_users)

    
    def cracker_combo(self, users):
        """
        Wrapper function around calling vivek's cracker code.

        This will be called after training with a list of users that we want to
        attack. Then go over each user individually, and call crack with its
        User object as the classifier, and user string/int representation.
        
        @users: List of user objects

        Requirements for Cracker:
            user_X needs to be collections.defaultdict - Can modify our data to
            get this. Actually dict works too. 
            user_Z can be ignored as it is only being used in
            Serwadda's attack which we don't need to replicate here.
            max_tries = 100 seems good enough.
        """
        # First of all, let us take our data dict and convert it into a
        # collections.defaultdict as cracker expects that. 
        print('starting cracker combo!')
        classifier_names = [] 
        if self.params.ensemble:
            classifier_names.append('ensemble')
        else:
            for u in users:
                for cl_name in u.classifiers:
                    classifier_names.append(cl_name)
                break
         
        for cl_name in classifier_names:
            # A cracker object per classifier
            cracker = Cracker(dict(self.data), self.exp.unique_features_data,
                        self.exp.params.dataset)
            for user in users:
                cracker.crack(user, cl_name) 
            # Because we are only able to run one cracker per classifier so far
            self.exp.stats.cracker[cl_name] = cracker

    def gen_pickle_name(self, Xi, user, clusters):
        """
        """ 
        # Using this because inputs could be different to the classifier
        # depending on the settings we are using...especially in the 2-class
        # classifier case
        hashed_input = hashlib.sha1(str(Xi)).hexdigest()
        
        if clusters != 64:
            name = user + "_" + hashed_input + "_" + str(clusters) + "_kmeans"
        else:
            # Because we have already saved 64 clusters stuff in this format...
            name = user + "_" + hashed_input + "_kmeans"
 
        directory = "./pickle/"
        return directory + name + ".pickle"
    
    def limited_kmeans_attack(self, user, clusters=16, impostors=20,
            sample=100):
        """
        The idea here would be to attempt kmeans but with a randomly chosen
        much smaller subset of the impostor sample. Would it perform as well as
        the other attack?
        """
        assert False, 'not implemented yet'


    def simple_average_attack(self, user, num_avg=50):
        """
        For every feature, just get the mean of all other users, or a subset of
        all the users, and send that in as a result.

        @ret: A feature vector to try on the user.

        Not using this right now.
        """
        all_features = []
        features = []
        
        # Simple implementation looping over everything.
        # check is just used for initializing the lists the first time, could
        # have used a smarter way I guess.
        check = 0
        for d in self.data:
            if d == user:
                continue 
            # Get all the timing info for this particular user.
            examples = self.data[d] 
            if check == 0:
                for i, j in enumerate(examples[0]):
                    all_features.append([])

            for i, j in enumerate(examples[0]):
                l = [item[i] for item in examples] 
                all_features[i].append(mean(l))

            check += 1

        for f in all_features: 
            features.append(mean(f))
        
        # Now we go over every feature and average them:
        return np.array(features)

   
