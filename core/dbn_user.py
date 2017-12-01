import numpy as np
from nolearn.dbn import DBN
import pandas as pd
from sklearn.metrics import roc_curve
import os.path
import pickle
from sklearn.metrics import mean_squared_error

NUM_IMPOSTOR_TEST_SAMPLES = 4
PICKLE = False

class DBN_User(object):

    def __init__(self, user, X, normalize=False):
        """
        @X: np array.
        """
        self.user = user
        self.y = user
        
        # shouldn't be needed.
        self.normalize = normalize 
        
        #TODO: These can be set in the classifier object instead.
        # Major parameters for the DBN
        self.shape = [X.shape[1], 100, 100, 2]  # same as in deng.
        self.learn_rates = 0.1
        self.epochs = 10000
        self.learn_rates_pretrain = 0.1
        self.epochs_pretrain = 10000
        self.learn_rate_decays = 0.9
        self.verbose = 0
        self.minibatches_per_epoch = 5

        # Not sure what's the best thing here. If we set it None, then the
        # defaul one uses the num of mistakes as a loss function.
        self.loss_funct = None
        
        # Set up some constants for normalizing.
        # Shoudl we do do this normalization ONLY with trainX?
        if self.normalize:
            step = 3
            self.duration_mins = X[:, 0::step].mean() - X[:, 0::step].std()
            self.duration_maxs = X[:, 0::step].mean() + X[:, 0::step].std()
            self.latency_mins = X[:, 1::step].mean() - X[:, 1::step].std()
            self.latency_maxs = X[:, 1::step].mean() + X[:, 1::step].std()
            self.ud_mins = X[:, 2::step].mean() - X[:, 2::step].std()
            self.ud_maxs = X[:, 2::step].mean() + X[:, 2::step].std()

        # self.X = X
        #FIXME: Shuffle?
        np.random.shuffle(X)
        self.train_X, self.test_X = np.split(X, 2)

        # This is used to select test samples with newer impostors.
        self.cur_test_sample = 0
        
        # Scores for user/impostor data which we use to calculate eer.
        # For instance, in the Deng paper's case, we will add examples from
        # testing on every impostor here - along with their scores, and the
        # scores of the 200 remaining user scores - we need to see which dbn
        # (from all the dbns trained for diff impostors...) we use for this.
        self.results = []
        # This will be set after training all the dbns
        self.threshold = 0
        self.eers = {}

        # Let's divide this into train and test examples

        # Now we will have a dict of dbn classifiers - one for each of the
        # impostors, because we will not be using the impostor samples to train
        # this dbn.

        # we could potentially just use the self.classifiers in the User
        # object.
        self.dbns = {}
        self.classifiers = self.dbns
    
    def get_user_eer(self):
        """
        This will use the self.results variable which we have been adding to
        throughout the training phase.
        """
        # Depends on how many impostors we are choosing etc.
        # print(len(self.results))
        # assert len(self.results) == 450, "total results are wrong in eer"

        results = pd.DataFrame(self.results, columns=['label', 'genuine', 'score']) 
        # idx is the threshold where eer is what it is.
        eer, thresholds, idx = eer_from_scores(results)
        print("eer is ", eer)
        # FIXME: Need to change this for the multi classifiers setting
        self.eers['mixed'] = eer
        self.threshold = thresholds[idx]

        # FIXME: Set threshold for this user etc.
        return eer

    def _normalize(self, samples):
        """
        @samples: list of features that we are normalizing. 

        Because these are lists, the changes should be reflected without us
        returning the list.
        """
        for features in samples:
            for i, x in enumerate(features):
              if i % 3 == 0:
                features[i] = (x - self.duration_mins) / (self.duration_maxs - self.duration_mins)
              if i % 3 == 1:
                features[i] = (x - self.latency_mins) / (self.latency_maxs - self.latency_mins)
              if i % 3 == 2:
                features[i] = (x - self.ud_mins) / (self.ud_maxs - self.ud_mins) 
                
              if x < 0:
                features[i] = 0
              if x > 1:
                features[i] = 1


    def train(self, impostor, negative_samples):
        """
        impostor: just string rep of the impostor
        negative_samples: Set of impostor examples - which we label 0. This does not include
        examples from impostor.
        Training examples remain the same.
        
        At the end, will update self.dbns and add a new trained dbn for this
        impostor.
        """ 
        train_X = np.copy(self.train_X)
        samples = np.vstack((train_X, negative_samples))
        labels = []
        # Add labels: 1 - user, and 0 - impostor.
        for i in self.train_X:
            labels.append(1)
        for i in negative_samples:
            labels.append(0)

        labels = np.array(labels)
        
        # print("details of train_X AFTER NORM...")
        # print("len trainX is = ", len(samples))
        # print("train X[0] is ", samples[0])
        # print("train X is ", samples)

        if self.normalize:
            sample = samples[0][1]
            self._normalize(samples)
            assert samples[0][1] != sample, "normalization not done"
        
        # Nothing disastrous appears to have happened before/after...
        # print("details of train_X before...")
        # print("len trainX is = ", len(samples))
        # print("train X[0] is ", samples[0])
        # print("train X is ", samples)

        
        # FIXME: Adds some randomness into the training process - should we
        # avoid this?
        # samples, labels = unison_shuffled_copies(samples, labels)
        
        file_name = self.gen_pickle_name(impostor, str(len(negative_samples)))
        
        # Training is expensive so let us save and load it if possible.

        # FIXME Update this - just want to make sure we are training new guy
        # everytime for now...
        if os.path.isfile(file_name): 
        # if False:
            with open(file_name, 'rb') as handle:
                self.dbns[impostor] = pickle.load(handle)
                print("loaded file ", file_name, "from disk") 

        else:
            # let us train this guy and then save it.
            self.dbns[impostor] = DBN(self.shape,
                    learn_rates = self.learn_rates,
                    epochs = self.epochs,
                    learn_rates_pretrain = self.learn_rates_pretrain,
                    epochs_pretrain = self.epochs_pretrain,
                    learn_rate_decays = self.learn_rate_decays,
                    verbose= self.verbose,
                    minibatches_per_epoch=self.minibatches_per_epoch,
                    loss_funct = self.loss_funct)
            
            self.dbns[impostor].fit(samples, labels)

            with open(file_name, 'w+') as handle:
                pickle.dump(self.dbns[impostor], handle, protocol=pickle.HIGHEST_PROTOCOL)
                print("saved ", file_name, "on disk")
    
    def gen_pickle_name(self, impostor, num_negative):
        
        shape = ''
        for s in self.shape:
            shape += str(s)
        
        name = "./saved_dbns/"
        # FIXME Refine this later
        name += self.user + '_' + impostor + '_' + shape + \
        str(self.learn_rates) + str(self.epochs) + \
        str(self.learn_rates_pretrain) + str(self.epochs_pretrain) + \
        str(self.learn_rate_decays) + num_negative \
        + str(NUM_IMPOSTOR_TEST_SAMPLES)

        name += '.pickle'

        return name

    def set_user_thresholds(self):
        """
        Testing by adding all the genuine samples to the results list only once
        - here the problem is that we can only run it on one of these
          classifiers.
        """
        assert False
        for k in self.dbns:
            genuine_scores = self.dbns[k].get_final_activations(self.test_X, 1)

            score_types = np.array([True] * len(genuine_scores))
            scores = np.r_[genuine_scores]

            self.results.extend(zip([self.user]*len(scores), score_types, scores)) 
            
            # We only want to run it on one of the classifiers since we are
            # giving all 200 samples to it. 
            break

    def set_thresholds(self, impostor, impostor_samples, num_user_samples=4):
        """
        FIXME: Not sure what is the best name for this....
        @num_user_samples - just starts from 0 of self.test_X and keeps adding
        next num_user samples to get next set.
        @test_sampels: are examples from impostor.
        """
        # All the labels added so far are from the impostor
        impostor_labels = [0]*len(impostor_samples) 
        # Get the user's samples to test on.

        user_labels = []
        user_samples = []

        for i in range(num_user_samples):
            # % arithmetic incase we cross the total samples
            index = (self.cur_test_sample + i) % len(self.test_X)
            user_samples.append(self.test_X[index])
            user_labels.append(1)
        
        # Gotta create new guys so it doesn't fuck up the real data
        user_samples = np.copy(user_samples)
        impostor_samples = np.copy(impostor_samples)

        self.cur_test_sample += num_user_samples
        
        if self.normalize:
            sample = impostor_samples[0][1]
            self._normalize(impostor_samples)
            self._normalize(user_samples)
            assert impostor_samples[0][1] != sample, "normalization not done"

        # Note: I don't think we need to shuffle these - shouldn't affect the
        # scores at all.

        impostor_samples = np.array(impostor_samples)
        # test_labels = np.array(impostor_labels)
        user_samples = np.array(user_samples)
        # user_labels = np.array(user_labels)
        
        # Now, we don't want predict - but we want a function that will return
        # the last layer's neuron activations - where 0 corresponds to
        # impostor, and 1 corresponds to user.

        # Maybe we should be storing these in the user class? If we want to mess around with the
        # threshold values, it might be useful.

        if num_user_samples > 0:
            genuine_scores=self.dbns[impostor].get_final_activations(user_samples,1)

        impostor_scores=self.dbns[impostor].get_final_activations(impostor_samples,1)
        
        if num_user_samples > 0:
            score_types = np.array([True] * len(genuine_scores) + [False] * len(impostor_scores))
            scores = np.r_[genuine_scores, impostor_scores]
        
        else:
            # This is the case where we fit genuine samples later on just one
            # of the 50 impostor classifiers.
            score_types = np.array([False] * len(impostor_scores))
            scores = np.r_[impostor_scores]
    
        # print("scores are ", scores)

        # FIXME: Should we be normalizing the scores, or I guess they are
        # already sort of normalized?

        # Now we need to add all the results to the per user list - which will
        # be used to calculate the eer score later.

        # The first value doesn't matter - just the way I had eer function set
        # up last time.
        self.results.extend(zip([self.user]*len(scores), score_types, scores)) 
        
        # Just get some local results -- but there are too many of these.
        tmp_results = []
        tmp_results.extend(zip([self.user]*len(scores), score_types, scores)) 
            
        tmp_results = pd.DataFrame(tmp_results,columns=['label', 'genuine', 'score']) 
        # idx is the threshold where eer is what it is.
        eer, thresholds, idx = eer_from_scores(tmp_results)
        print("eer is ", eer)
        print("threshold is ", thresholds[idx])

    def test(self, samples, impostor='1'):
        """
        This will just take in a single feature vector and return +1 or -1.
        """
        assert len(samples) == 31, 'not 31'
        samples = [samples]
        samples = np.array(samples)
        scores = self.dbns[impostor].get_final_activations(samples, 1)
        
        ret_val = []
        for s in scores:
            if s < self.threshold:
               ret_val.append(-1) 
            else:
                ret_val.append(1)

        return ret_val

def unison_shuffled_copies(a, b):

    np.random.seed(1111)  # Only place we were using random
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def eer_from_scores(scores):
    """
    Compute the eer from a dataframe with genuine and score columns
    """
    far, tpr, thresholds = roc_curve(scores['genuine'], scores['score'])
    frr = (1 - tpr)
    idx = np.argmin(np.abs(far - frr))
    
    # 1st thing being returned is the eer.
    # 2nd thing - should be the threshold at that point.
    # FIXME: Need to confirm these two things.
    return np.mean([far[idx], frr[idx]]), thresholds, idx
