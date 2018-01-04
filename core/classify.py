import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from sklearn.model_selection import train_test_split

import hashlib
import pickle
import os
import random

def eer_from_scores(scores):
    """
    Compute the eer from a dataframe with genuine and score columns
    Method originally from https://github.com/vmonaco/kboc.
    """
    far, tpr, thresholds = roc_curve(scores['genuine'], scores['score'])
    frr = (1 - tpr)
    idx = np.argmin(np.abs(far - frr))
    return np.mean([far[idx], frr[idx]]), thresholds, idx, far

# We want to create one of these guys for each user.
class ClassifierFixedText(object):
    """
    class originally from https://github.com/vmonaco/kboc, but it has been completely modified since then.
    """

    def __init__(self, anomaly_detector, name, y,
            exp, two_class):
        '''
        @y: user for whom this classifier is trained.
        @name: classifier name
        @exp: the experiment class - being used to pass parameters/stats
        around. Might need to clean this up later.
        @two_class: two class vs one class classifier
        '''
        self.y = y
        self.params = exp.params
        self.stats = exp.stats
        self.two_class = two_class

        self.anomaly_detector = anomaly_detector()
        self.name = name

        # Specific only to fixed text.
        self.duration_mins = 0
        self.duration_maxs = 0
        self.latency_mins = 0
        self.latency_maxs = 0
        self.ud_mins = 0
        self.ud_maxs = 0

        # Score Normalization is used more generally.
        self.min_score = None
        self.max_score = None

    def fit(self, Xi, labels=None):
        """
            @Xi:
            Note: Normalization parameters are ONLY set in fit - so it only
            sees the train_X and NOT test_X.

            This is essentially a wrapper function around the classifier's fit
            - mostly here we just deal with normalization, and pickling the
              classifier.
        """
        ## Xi will be a array of arrays - with each array representing the
        # features of one password being typed in by the user.
        step = 3 # Step jumping up between features
        if self.params.feature_norm == 'stddev':
            assert self.params.keystrokes, 'normalization for keystrokes'

            self.duration_mins= Xi[:, 0::step].mean() - Xi[:, 0::step].std()
            self.duration_maxs= Xi[:, 0::step].mean() + Xi[:, 0::step].std()
            self.latency_mins= Xi[:, 1::step].mean() - Xi[:, 1::step].std()
            self.latency_maxs= Xi[:, 1::step].mean() + Xi[:, 1::step].std()
            self.ud_mins= Xi[:, 2::step].mean() - Xi[:, 2::step].std()
            self.ud_maxs= Xi[:, 2::step].mean() + Xi[:, 2::step].std()

            if self.duration_maxs == self.duration_mins:
                print('WTF??')
                print(Xi)

        elif self.params.feature_norm == 'minmax':
            assert self.params.keystrokes, 'normalization for keystrokes'
            self.duration_mins= Xi[:, 0::step].min()
            self.duration_maxs= Xi[:, 0::step].max()
            self.latency_mins= Xi[:, 1::step].min()
            self.latency_maxs= Xi[:, 1::step].max()
            self.ud_mins= Xi[:, 2::step].min()
            self.ud_maxs= Xi[:, 2::step].max()

        if self.params.feature_norm is not None:
            assert self.params.keystrokes, 'normalization for keystrokes'
            Xi = self._normalize_features(Xi)

        from tensorflow.python.framework import ops
        ops.reset_default_graph()

        file_name = self.gen_pickle_name(Xi)
        # Don't give an option to pickle or not to pickle. Pickling is always
        # good.

        # TODO: Clean up pickle code.
        if file_name is not None and os.path.isfile(file_name):
            with open(file_name, 'rb') as handle:
                self.anomaly_detector = pickle.load(handle)

        else:
            if labels is None:
                # One class classifier...
                self.anomaly_detector.fit(Xi)
            else:
                # Two class classifier...
                self.anomaly_detector.fit(Xi, labels)

            if file_name is None:
                return

            with open(file_name, 'w+') as handle:
                pickle.dump(self.anomaly_detector, handle, protocol=pickle.HIGHEST_PROTOCOL)
                if self.params.verbose:
                    print('dumped pickle ', file_name)

    def score(self, X, save_stats=False):
        """
        @X: np array of length N. X[0] are features, and so on.
        Each of these get a score based on our classification method.

        @ret: scores - array of scores of same length as X
        """
        scores = np.zeros(len(X))
        for i, Xi in enumerate(X):

            if self.params.feature_norm is not None:

                # Important to create a copy here - otherwise Xi will be
                # modified globally.

                # modify it to the form _normalize expects
                # Xi = np.copy(Xi[np.newaxis, :])
                Xi = Xi[np.newaxis, :]
                Xi = self._normalize_features(Xi)
                # Since this is just a single feature vector, we can remove the
                # unneccessary dimension and make it 1D - which
                # anomaly_dector.score expects
                Xi = Xi.squeeze()
                # row.append(Xi)

            # Now we are supposed to normalize these
            score = self.anomaly_detector.score(Xi)
            if self.max_score is not None:
                score = (score - self.min_score) / (self.max_score - self.min_score)

            scores[i] = score

            # in order to avoid saving the stats from the training phase.
            if save_stats:
                row.append(self.threshold)
                row.append(scores[i])
                self.stats.monaco_norm_effect[self.y].append(row)

        return scores

    def gen_pickle_name(self, Xi):
        """
        """
        if self.name is None or not self.params.pickle:
            return None

        # pickling doesn't seem to work well with tf models
        if "Autoencoder" in self.name:
            return None

        cl_name = self.name
        #TODO: Generalize this for different classifiers.
        if cl_name == 'RandomForests':
            # num of trees is an important parameter here.
            cl_name += str(self.params.rf_trees)
        if cl_name == 'KNC':
            cl_name += str(self.params.knc_neighbors)

        hashed_input = hashlib.sha1(str(Xi)).hexdigest()
        name = cl_name + "_" + hashed_input

        # FIXME: Use the correct os methods for this...
        directory = "./pickle/"
        return directory + name + ".pickle"

    def _normalize_features(self, X):
        """
        Method from https://github.com/vmonaco/kboc.
        @X is a np array of feature vectors. Normalize each of the vectors in
        X.
        """
        assert (self.duration_maxs - self.duration_mins) != 0, 'being divided by 0 in normalize '

        if (self.latency_maxs - self.latency_mins == 0):
            print 'latency_maxs - latency_mins is 0 for ', self.y

        if (self.ud_maxs - self.ud_mins == 0):
            print 'ud_maxs - ud_mins == 0 for ', self.y

        X[:, 0::3] = (X[:, 0::3] - self.duration_mins) / (self.duration_maxs - self.duration_mins)
        X[:, 1::3] = (X[:, 1::3] - self.latency_mins) / (self.latency_maxs - self.latency_mins)
        X[:, 2::3] = (X[:, 2::3] - self.ud_mins) / (self.ud_maxs - self.ud_mins)

        X[X < 0] = 0
        X[X > 1] = 1
        return X

def mix_samples(train_genuine, train_impostors):
    """
    Returns a single np.array with samples, and corresponding labels.
    """
    samples = np.vstack((train_genuine, train_impostors))

    labels = []
    # Add labels: 1 - user, and 0 - impostor.
    for i in train_genuine:
        labels.append(1)
    for i in train_impostors:
        labels.append(0)

    labels = np.array(labels)

    #FIXME: Do a unison shuffle on these? Might help with training?
    return samples, labels

def run_classifier(cl, X, impostor_samples, seed, score_normalization,
                impostor_train = None):
    """
        @X: numpy array - with each element being another array of features.
        Right now, we do very simple things with splitting X, impostor_samples
        etc, but if we want to be more complicated, then this will be a good
        place for that.

        @impostor_train: Only used in case of two class classifiers.
        @ensemble: FIXME: where is the best place to implement that?

        Does partitioning of training / test samples here.
    """
    np.random.seed(seed)
    results = []

    if cl.params.android:
        # in this case we do not use impostor_train parameter at all - so maybe
        # not send it in itself.
        # FIXME: Add random_state=42 or not? Not doing it right now so I can
        # keep using my pickled ones.
        assert len(X) == len(impostor_samples), 'should be same len'

        # By doing a randomized split, we can get better performance, but to be more consistent
        # with the keystrokes dynamics datasets, we do the split where first 50% samples are
        # training, and next 50% samples are test samples.

        # Randomized split.
        # train_samples, test_samples = train_test_split(X, test_size =
                # cl.params.swipes_test_size, random_state=cl.params.seed)

        # Ordered split.
        X_split = np.array_split(X,2)
        train_samples = X_split[0]
        test_samples = X_split[1]

        # For impostor training, and genuine impostor samples, we just do a randomized split.
        impostor_train, impostor_samples = train_test_split(impostor_samples,
                test_size = cl.params.swipes_test_size)

        assert len(train_samples) == len(test_samples) == len(impostor_train),\
                'valid test'

    # Splitting it into two equal halves - as done in the CMU paper etc.
    elif 'mturk' in cl.params.dataset:
        # BIG Question: how to split?
        # train_samples, test_samples = train_test_split(X,
                # test_size=cl.params.swipes_test_size,
                # random_state=cl.params.seed)
        X_split = np.array_split(X,2)
        train_samples = X_split[0]
        test_samples = X_split[1]
        assert abs(len(train_samples) - len(test_samples)) <= 1, 'should be split even'

    elif 'cmu' in cl.params.dataset:
        X_split = np.array_split(X,2)
        train_samples = X_split[0]
        test_samples = X_split[1]

    elif cl.params.mouse:
        # X_split = np.array_split(X,2)
        # train_samples = X_split[0]
        # test_samples = X_split[1]
        train_samples, test_samples = train_test_split(X, test_size = 0.5)
        print('len of test samples is : ', len(test_samples))
        print('len of impostor samples is : ', len(impostor_samples))
        if len(impostor_samples) > len(test_samples):
            impostor_samples = random.sample(impostor_samples,
                    len(test_samples))
        print('len of impostor samples is : ', len(impostor_samples))
    else:
        assert False, 'has to be keystrokes, android, or mouse'

    if not cl.two_class:
        cl.fit(train_samples)
    elif cl.two_class:
        samples, labels = mix_samples(train_samples, impostor_train)
        cl.fit(samples, labels)

    # How well it does on correct vs impostor samples?
    genuine_scores = cl.score(test_samples)
    impostor_scores = cl.score(impostor_samples)

    score_types = np.array([True]*len(genuine_scores) + [False]*len(impostor_scores))
    scores = np.r_[genuine_scores, impostor_scores]

    # normalize both the user + impostor scores together.
    scores = _normalize_scores(scores, score_normalization, cl)
    results.extend(zip([0] * len(scores), score_types, scores))
    results = pd.DataFrame(results, columns=['label', 'genuine', 'score'])

    # idx is the threshold where eer is what it is.
    eer, thresholds, idx, far = eer_from_scores(results)

    new_genuine_scores = scores[0:len(genuine_scores)]
    assert len(genuine_scores) == len(new_genuine_scores), 'genuine scores \
            should be of same length after normalization'

    # FIXME: Make this less ugly
    return eer, thresholds[idx], new_genuine_scores, thresholds, far

def _normalize_scores(scores, score_normalization, cl):
    """
    Similar to feature normalization - but fewer things to deal with.
    returns the updated scores array.

    FIXME: Dealing with it when all scores end up being the same...
    """
    if score_normalization is not None:
        if score_normalization == 'minmax':
            # Normalize scores between min and max for each claimed user (each model)
            min_score, max_score = scores.min(), scores.max()
        elif score_normalization == 'stddev':
            # Normalize scores between +/- 2 std devs of the scores of each model
            min_score = scores.mean() - 2 *scores.std()
            max_score = scores.mean() + 2 * scores.std()
        else:
            raise Exception('Unrecognized score normalization:', score_normalization)

        cl.max_score = max_score
        cl.min_score = min_score

        scores = (scores - min_score) / (max_score - min_score)

        # Clipping step
        scores[scores < 0] = 0
        scores[scores > 1] = 1

    #FIXME: Wasn't actually required to return anything
    return scores
