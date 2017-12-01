from anomaly import *
from classify import *
from cracker import *
from sklearn.cluster import KMeans
import random
import math
from collections import defaultdict
import copy

EXPERT_LEVEL_ATTACK = False

def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
        return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

class User(object):
    
    def __init__(self, y, X, impostor_X, impostor_train, exp):
       """
       @y: user.       
       @X: All samples of this user.
       @impostor_X: Selected samples of the impostors
       @impostor_train: Selected samples of impostors used for training - 2
       class classifiers.
       @exp: experiment object.
       
       FIXME: Ideally, should get impostor_X, impostor_train in train or so.
       impostor_train is used only for two class clasifiers.
       """  
       # Key is the cl_name
       self.classifiers = {}

       self.y = y # Name of user

       # all the positive samples.
       self.X = X
       self.impostor_X = impostor_X
       self.impostor_train = impostor_train

       self.params = exp.params
       self.exp = exp
       
       self.exp.stats.monaco_norm_effect[self.y] = []
        
       self.eers = {}   # cl.name will be keys

       # These aren't being always used - just in case we want to run
       # robustness checks.
       self.test_genuine_samples = []
 
    def train(self, classifier, two_class=False, name=None):
        """
        We want the user to be able to add multiple classifiers to train it.
        """
        cl = ClassifierFixedText(classifier, name, self.y, self.exp, two_class)
        self.classifiers[cl.name] = cl
        
        # Note: It is crucial to pass in np.copy - as the classifiers mess with
        # these arrays and then the second classifier will start failing.
        if two_class:
            eer, eer_thresh, genuine_scores, thresholds, fars = run_classifier(cl,
                    np.copy(self.X), np.copy(self.impostor_X),
                    self.params.seed,
                    self.params.score_norm, 
                    impostor_train=np.copy(self.impostor_train))
        else:
            eer, eer_thresh, genuine_scores, thresholds, fars = run_classifier(cl,
                        np.copy(self.X), np.copy(self.impostor_X),
                        self.params.seed,
                        self.params.score_norm)

        self.eers[cl.name] = eer
        # self.genuine_scores = genuine_scores

        # Now we can set an arbitrary threshold for deciding whether its the
        # print('eer thresh = ', eer_thresh) 
        # print('eer = ', eer)
        # print('mean thresh = ', np.mean(genuine_scores))
        # print('median thresh = ', np.median(genuine_scores))
        # print('25th percentile = ', np.percentile(genuine_scores, 25))
        # print('50th percentile = ', np.percentile(genuine_scores, 50))
        # print('75th percentile = ', np.percentile(genuine_scores, 75))
        
        if self.params.eer_threshold:
            cl.threshold = eer_thresh
        elif self.params.mean_threshold:
            cl.threshold = np.mean(genuine_scores)
        elif self.params.median_threshold:
            cl.threshold = np.median(genuine_scores)
         
        median_vals = []
        for med in range(0,100,5):
            median_vals.append(med)

        if self.params.median_classifiers:
            for i, val in enumerate(median_vals):
                thresh = np.percentile(genuine_scores, val)
                # name = cl.name + str(i) + '_median_' + str(val) + '_thresh_'+str(thresh)
                name = cl.name + str(i) + '_median_' + str(val)
                cl2 = copy.deepcopy(cl)
                cl2.threshold = thresh
                self.classifiers[name] = cl2 

    def test(self, features, cl_name):
        '''
        @features: feature vector to be tested - only one feature vector tested
        at a time - so features is an np array of num_feature elements 

        tests across all classifiers, or just the classifier at the given
        index. 
        
        # FIXME: turn this into a dict using cl.name?
        @ret: list of scores - for each classifier --> might want to just
        change this to return only one score for a specified classifier?
        '''
        assert len(features) == self.exp.loader.num_features, 'len offeatures in user.test does not match' 

        if 'cmu' in self.exp.params.dataset or 'mturk' in self.exp.params.dataset:
            self.exp._sanity_check_cmu_features([features])

        scores = []
        if cl_name == 'all':
            for cl in self.classifiers:
                scores.append(self._cl_score(self.classifiers[cl], features))
        elif cl_name =='ensemble':
            all_scores = []
            for cl in self.classifiers:
                all_scores.append(self._cl_score(self.classifiers[cl], features))

            # If any of the scores were 0, then return 0, else return 1.
            for score in all_scores:
                if score == 0:
                    scores.append(0)
                    break
            if len(scores) == 0:
                scores.append(1)

        else:
            cl = self.classifiers[cl_name]
            scores.append(self._cl_score(cl, features))

        return scores

    def _cl_score(self, cl, features):
        '''
        @scores a single classifier
        ''' 
        feature_list = [features]
        new_features = np.array(feature_list)
        
        scores = cl.score(new_features,
                save_stats=self.params.extract_attack_vectors)
        # Only one element in returned array
        assert len(scores) == 1, 'array in cl_score should just return 1 el'
        score = scores[0]
        
        # Don't need to clip it to 0 or 1 here as we did in training because
        # this data isn't used for training.
            
        if score < cl.threshold:
            return 0
        else:
            return 1

# TODO: Combine this with the normal user, or use inheritance crap.
class MouseUser(object):
    
    def __init__(self, y, mouse_features, impostor_samples, exp):
       """
       TODO: Support for two class classifiers?
       @y: user.       
       @mouse_features: MouseFeatures object for this user.
       @impostor_samples: Task objects representing a single completed task.
       @exp: experiment object. 
       """  
       # Key is the cl_name
       self.classifiers = {}

       self.y = y # Name of user

       self.mouse_features = mouse_features
       self.impostor_samples = impostor_samples

       self.params = exp.params
       self.exp = exp  
       self.eers = {}   # cl.name will be keys
 
    def train(self, classifier, two_class=False, name=None):
        """
        We want the user to be able to add multiple classifiers to train it.
        """
        cl = ClassifierFixedText(classifier, name, self.y, self.exp, two_class)
        self.classifiers[cl.name] = cl
        
        # Now need to generate X, impostor samples - which will be
        # feature-distance vectors. 
        X = self.mouse_features.get_all_distance_vectors(self.mouse_features.tasks)
        impostor_samples=self.mouse_features.get_all_distance_vectors(self.impostor_samples) 
        # TODO: Get rid of np.copy here.
        if two_class:
            eer, eer_thresh, genuine_scores, thresholds, far  = run_classifier(cl,
                    np.copy(self.X), np.copy(self.impostor_X),
                    self.params.seed,
                    self.params.score_norm, 
                    impostor_train=np.copy(self.impostor_train))
        else:
            eer, eer_thresh, genuine_scores, thresholds, far = run_classifier(cl,
                        np.copy(X), np.copy(impostor_samples),
                        self.params.seed,
                        self.params.score_norm)

        self.eers[cl.name] = eer 

        print('eer = for user {} is {}'.format(self.y, eer))
        if self.params.eer_threshold:
            cl.threshold = eer_thresh
        elif self.params.mean_threshold:
            cl.threshold = np.mean(genuine_scores)
        elif self.params.median_threshold:
            cl.threshold = np.median(genuine_scores)
        
    
    def test(self, features, cl_name):
        '''
        @features: feature vector to be tested - only one feature vector tested
        at a time - so features is an np array of num_feature elements 

        tests across all classifiers, or just the classifier at the given
        index. 
        
        # FIXME: turn this into a dict using cl.name?
        @ret: list of scores - for each classifier --> might want to just
        change this to return only one score for a specified classifier?
        '''
        assert len(features) == self.exp.loader.num_features, 'len offeatures in user.test does not match' 
        
        if 'cmu' in self.exp.params.dataset or 'mturk' in self.exp.params.dataset:
            self.exp._sanity_check_cmu_features([features])

        scores = []
        if cl_name == 'all':
            for cl in self.classifiers:
                scores.append(self._cl_score(self.classifiers[cl], features))
        elif cl_name =='ensemble':
            all_scores = []
            for cl in self.classifiers:
                all_scores.append(self._cl_score(self.classifiers[cl], features))

            # If any of the scores were 0, then return 0, else return 1.
            for score in all_scores:
                if score == 0:
                    scores.append(0)
                    break
            if len(scores) == 0:
                scores.append(1)

        else:
            cl = self.classifiers[cl_name]
            scores.append(self._cl_score(cl, features))

        return scores

    def _cl_score(self, cl, features):
        '''
        @scores a single classifier
        '''
        
        feature_list = [features]
        new_features = np.array(feature_list)
        
        scores = cl.score(new_features, save_stats=self.params.extract_attack_vectors)
        # Only one element in returned array
        assert len(scores) == 1, 'array in cl_score should just return 1 el'
        score = scores[0]
        
        # Don't need to clip it to 0 or 1 here as we did in training because
        # this data isn't used for training.
            
        if score < cl.threshold:
            return 0
        else:
            return 1


def mean(l):
    return sum(l) / float(len(l))
    
