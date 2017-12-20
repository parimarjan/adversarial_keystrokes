import os
import sys
import random
import time
import pickle
import csv
import hashlib
import numpy as np
import json
np.set_printoptions(precision=5, suppress=True)

from parameters import Parameters
from loader import Loader
from adversary import *
from stats import Stats
from user import User, MouseUser
from anomaly import *
from pohmm import Pohmm
from collections import defaultdict
from util import *

from scipy.stats import norm, gamma, beta
from sklearn.cluster import KMeans
# from gap import gap
from collections import defaultdict

from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split

try:
    import seaborn as sns
except ImportError:
    pass

def plot_distribution(samples, file_name):
    print('num of samples are ', len(samples)) 
    main_fig = plt.figure()
    ax1 = main_fig.add_subplot(131)
    fig = sns.distplot(samples, kde=False, ax=ax1)

    ax1.set_xlim((min(samples)-0.1,max(samples)+0.1))
    # sns.plt.show()
    fig.get_figure().savefig(file_name)
    print('saved file ', file_name)
    sns.plt.close() 

def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
        return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)
 
class Experiment():
    '''
    The overarching class that keeps all information about the current
    experiment being run - and then can use this to display/save/compare stats
    later etc.

    FIXME: Define __repr__ or __str__ method for this which prints all the
    parameters etc.
    '''
    def __init__(self):
        '''
        Basically parses command line flags, and sets other 'global' experiment
        variables.
        '''
        # FIXME: Need to ensure the equivalent of init_globals in Parameters
        self.params = Parameters()
        self.loader = Loader(self)
        self.stats = Stats(self.params)
        self.data = self.loader.load_data()
        
        # list of hashes of each of the used samples
        self.used_samples = []
        self.attack_data = defaultdict(list)
        
        # used for serwadda attack.
        # TODO: Turn this into a loop and check all arguments.
        if 'cmu' in self.params.dataset or 'mturk' in self.params.dataset:
            self.unique_features_data = self._get_unique_data() 
            a = self.data['0'][0][0]
            b = self.data['0'][0][1]
            c = self.data['0'][0][2]
            assert isclose(a+c, b), 'test 1'
        else:
            # Run some other asserts here
            self.unique_features_data = None

        # Now that we have the data object, we can proceed to the actual
        # experiment
    
    def _gen_distribution_file_name(self, samples, key_num=None):
        '''
        Creates file name based on current dataset.
        TODO: Add hash of samples if required.
        @i: is used to determine if this was for hold or interval time.
        '''
        name = './distribution_plots/' + self.params.dataset + '_'
        if self.params.keystrokes:
            if (key_num % 3 == 1):
                # We are usually just avoiding this one.
                key_type = 'DD'
            elif key_num % 3 == 0:
                key_type = 'HOLD'
            elif key_num % 3 == 0: 
                key_type = 'UD'
            else:
                key_type = ''
        else:
            key_type = ''
 
        name += key_type + '_' + str(key_num) + '.png' 
        return name
           
    def probability_analysis(self):
        '''
        fits distributions? Didn't work too well.
        Make graphs of distributions - like digraph distributions etc.
        '''
        data = self._get_gan_data() 
        for k, samples in data.iteritems():
            file_name = self._gen_distribution_file_name(samples, int(k))
            # Basically got too many outlier max values
            m1 = min(samples)
            m2 = max(samples)
            m3 = np.mean(np.array(samples))
            print('min : ', m1)
            print('max : ', m2)
            print('mean: ', m3)
            
            # if 'cmu' in self.params.dataset:
                # samples = [s for s in samples if s < 0.5]

            # samples = random.sample(samples, 2000)
            fig = plot_distribution(samples, file_name) 
            # n = norm.fit(test_data)
            # print('normal = ', n)
            loc, scale = norm.fit(samples)
            print('fitting normal to {}, we get loc: {}, scale: {}'.format(k, loc, scale))
            print('let us try to generate samples from this distribution: ')
            gen_samples = np.random.normal(loc=loc, scale=scale, size=10) 
            print(gen_samples) 

    def cluster_analysis(self):
        # from gap import gap 
        gap_stat = False
        tsne = False
        cluster_analysis = True
        cohesion = False
        silhouette = False

        X = []
        accepted = 0
        removed = 0
        for d, features in self.data.iteritems():
            for f in features:
                if np.max(np.array(f)) < 1000.0:
                    accepted += 1
                    X.append(f)
                else:
                    removed += 1
        print "accepted: ", accepted
        print "removed: ", removed

        if gap_stat:
            X = np.array(X) 
            print(X.shape) 
            gaps, s_k, K = gap.gap_statistic(X, refs=None, B=10, K=range(3,15), N_init = 10) 
            print('gaps = ', gaps)
            print('sK = ', s_k)
            print('K = ', K)
            bestKValue = gap.find_optimal_k(gaps, s_k, K)
            print('bestKValue: ', bestKValue)

        if cluster_analysis:
            # X = np.array(X) 
            # kmeans = KMeans(n_clusters=16).fit(np.random.rand(X.shape[0], X.shape[1])) 
            kmeans = KMeans(n_clusters=16).fit(X)
            print(len(kmeans.labels_))
            
            new_clusters = defaultdict(int)

            final_clusters = defaultdict(int)
            good_clusters = 0
            bad_clusters = 0
            entropies = []
            for i, c in enumerate(kmeans.labels_):
                # i is the sample num
                # c is the cluster assigned to it.
                final_clusters[c] += 1
                if (i+1) % 400 == 0:
                    e_val = 0.0
                    # new_clusters = {0: 400}
                    for k in new_clusters:
                        fraction_points = new_clusters[k]/400.0
                        e_val += -fraction_points * math.log(fraction_points)

                    entropies.append(e_val)

                if i % 400 == 0 and i != 0:
                    # print(new_clusters)
                    good = False
                    for c, num in new_clusters.iteritems():
                        if num > 250:
                            good_clusters += 1
                            good = True
                    if not good:
                        bad_clusters += 1

                    new_clusters = defaultdict(int)
                
                new_clusters[c] += 1

            
            entropies = np.array(entropies)/math.log(16)
            print('good clusters = ', good_clusters)
            print('bad clusters = ', bad_clusters)
            print('len of new clusters = ', len(final_clusters))
            print "entropies:\n", entropies
            print "Average entropy: ", np.sum(entropies)/entropies.shape[0]
            print(final_clusters)
            print "Beginning K-means analysis\n"
            distances = []
            for i in range(100):
                kmeans = KMeans(n_clusters=i+1, max_iter=2000, random_state=0).fit(X)
                print "For i: ", i+1, "       distance: ", kmeans.inertia_
                distances.append(kmeans.inertia_)
            print "Distances: ", distances
            exit(0)

        if cohesion:
            # cohesion analysis:
            clusters = defaultdict(list)
            kmeans = KMeans(n_clusters=16).fit(X) 
            for i, c in enumerate(kmeans.labels_):
                clusters[c].append(X[i])
            
            print(len(clusters))
            for k, c in clusters.iteritems():
                print('k = ', k)
                print('len of elements in cluster = ', len(c))
                clusters[k] = np.array(clusters[k])
                row_mean = np.sum(clusters[k], axis=0)
                # print(row_sum[0])
                row_mean /= float(len(clusters[k]))
                # print(len(clusters[k]))
                # print(row_sum[0]) 
                sum = 0
                for feature in c:
                    sum += np.sum(np.square(row_mean - feature))
                print(sum)
        
        if silhouette:
            # for n_clusters in [2,4,6,8,10,12,14,16]:
            n_clusters = 16
            X = np.array(X)
            # X = np.random.rand(X.shape[0], X.shape[1])
            cluster_labels = KMeans(n_clusters=n_clusters).fit_predict(X) 
            silhouette_avg = silhouette_score(X, cluster_labels)
            print("For n_clusters =", n_clusters,
                  "The average silhouette_score is :", silhouette_avg)
        

        print('succesfully finished cluster analysis!')

    def _get_data(self, feature_num):
        '''
        @feature_num is an array.
        Assuming cmu format of the keystrokes data, feature_num = 0 is hold
        (0,3,6 etc) guys in each feature vector.
        '''
        extracted_features = []
        for _, feature_vectors in self.data.iteritems(): 
            for feature_vector in feature_vectors:
                for i, feature in enumerate(feature_vector):
                    if i % 3 in feature_num:
                        extracted_features.append(feature)
        
        print(len(extracted_features))
        # assert len(extracted_features) == 51 * 400 * 11, 'len does not match'
        return extracted_features

    def _get_gan_data(self, exclude=[], num=None):
        '''
        Get the hold data in the format needed for the gan adversary.
        
        TODO: Exclude a particular user from this data collection process etc.

        @ret: dict.
            i    :  [t_1, t_2 ...] (where the numbers show the feature
            numbers in the original feature vector.
        '''
        gan_data = defaultdict(list)
        random.seed(1234)
        if len(self.attack_data) > 0:
            data = self.attack_data
        else:
            print('in get gan data, attack data was not valid!')
            data = self.data

        for user, feature_vectors in data.iteritems():
            if user in exclude:
                continue

            for num_user_samples, feature_vector in enumerate(feature_vectors):
                for i, feature in enumerate(feature_vector):
                    if not (i % 3 == 1 and self.params.keystrokes):
                        gan_data[str(i)].append(feature)
                    elif not self.params.keystrokes:
                        # This is because for swipes etc there is no unique
                        # thing.
                        gan_data[str(i)].append(feature)
        
        if num is not None:
            # Reduce the number of samples per data point.
            smaller_gan_data = defaultdict(list)
            for k, samples in gan_data.iteritems():
                if len(gan_data[k]) == 0:
                    continue
                if num > len(gan_data[k]):
                    num = len(gan_data[k])
                smaller_gan_data[k] = random.sample(gan_data[k], num)
        else: 
            smaller_gan_data = gan_data 

        return smaller_gan_data
    
    def _get_unique_data(self):
        '''
        '''
        unique_features_user = defaultdict(list)

        for user, feature_vectors in self.data.iteritems():
            for feature_vector in feature_vectors:
                unique_features = [] # for each feature vector
                for i, feature in enumerate(feature_vector):
                    if not i % 3 == 1:
                        unique_features.append(feature)
                unique_features_user[user].append(unique_features)
            unique_features_user[user] = np.array(unique_features_user[user]) 
        return unique_features_user

    def _get_digraph_data(self,feature_num=[0,1], num_users=51):
        '''
        '''
        unique_features_user = []
        labels = [] 
        cur_user = 0
        for user, feature_vectors in self.data.iteritems():
            cur_user += 1
            if cur_user >= num_users:
                continue

            for feature_vector in feature_vectors:
                unique_features = [] # for each feature vector
                for i, feature in enumerate(feature_vector):
                    if not i % 3 in feature_num:
                        unique_features.append(feature)
                unique_features_user.append(unique_features)
                labels.append(user)

            # unique_features_user[user] = np.array(unique_features_user[user]) 
        
        print('unique feature users len: ', len(unique_features_user))
        # print(len(unique_features_user[0]))
        return np.array(unique_features_user), np.array(labels)
    
    def _exclude_samples(self):
        '''
        '''
        new_data = defaultdict(list)
        for u, samples in self.data.iteritems():
            for sample in samples:
                if hashlib.sha1(str(sample)).hexdigest() in self.used_samples:
                    continue
                new_data[u].append(sample)
        
        # TODO: Do this nicer.
        for k in new_data:
            new_data[k] = np.array(new_data[k])

        return new_data

    def start(self):
        '''
        What used to be the main starting function.
        ''' 
        # run statistics on the data. 
        users = self._train_users(self.data)  
        # get rid of doubles
        if not self.params.mouse:
            self.used_samples = set(self.used_samples)

        self.stats.mean_eer = self._get_mean_eer(users) 
        # print('len of samples to exclude are: ', len(self.used_samples))
        self.attack_data = self._exclude_samples()

        print('eer for classifiers are ', self.stats.mean_eer) 
        # add flag.
        # self._great_ok_bad(users)

        if self.params.robustness_check:
            assert False, 'Need to fix robustness check'
            self._test_classifier(users, data)
        
        # Time for attacks
        attacker = Adversary(self.attack_data, self)

        if self.params.cracker: 
            # Should have this in main.py itself?
            attacker.cracker_combo(users)
        
        if self.params.kmeans_attack: 
            # Note: This seems a bit weird because we want to pass the
            # classifier num to kmeans_attack
            for c in range(len(users[0].classifiers)):
                self._kmeans_attack(users, attacker, classifier=c)
        
        if False:
            for band in [0.0001, 0.001, 0.002, 0.003, 0.004, 0.005]:
                print('bandwidth is ', band)
                prob_adv=KernelDensityAdversary(self.params.dataset, band)
                gan_data = self._get_gan_data() 
                prob_adv.add_data(gan_data)
                prob_adv.train() 
                self._gan_attack(users, prob_adv)  

        if self.params.digraph_attack:
 
            num_hacked = 0
            # Need to only train this once because there is no overlap with the
            # user data!
            with open(self.params.digraph_attack_file) as data_file:
                attack_prob_data = json.load(data_file)

            attacker =ProbabilisticAdversary(self.params.dataset,
                    clusters=self.params.attack_clusters,
                    attack_alg=self.params.digraph_attack_type,
                    components=self.params.gaussian_components)
            attacker.add_data(attack_prob_data)
            attacker.train() 

            if self.params.password_keys is None:
                assert False, 'should not be doing digraph attack'

            if self.params.prob_then_kmeans:
                samples = attacker.generate_attack(self.params.password_keys,
                        num_samples=self.params.prob_kmeans_sample_size)

                # Update samples to be correct format
                for i, sample in enumerate(samples):
                    sample = np.array(sample)
                    if self.params.keystrokes:
                        samples[i] = self._add_dependent_features(sample)
                
                # Want to do this for every classifier.
                
                for cl_name in self.params.classifiers_list:
                    print('cl_name: ', cl_name) 
                    # single cracker is enough to run on all the users one by one.
                    tmp_data = {}
                    tmp_data['attacker'] = np.array(samples)  
                    cracker = Cracker(dict(tmp_data), self.unique_features_data,
                                self.params.dataset, num_users=len(self.data))
                    for user in users:
                        num_hacked += self._prob_kmeans_attack(user, attacker,
                                    cracker, cl_name) 

                    print('num hacked users = ', num_hacked)
                    print('num secure users = ', len(users) - num_hacked)
                    print('percentage of hacked users =',float(num_hacked)/len(users))
                    self.stats.cracker['prob_kmeans_attack' + cl_name] = cracker
            else:
                self._gan_attack(users, attacker, password_keys=self.params.password_keys)
                
        if self.params.prob_attack:

            if self.params.android:
                prob_adv=SwipesProbabilisticAdversary(self.params.dataset, clusters=self.params.attack_clusters)
                gan_data = self._get_gan_data() 
                prob_adv.add_data(gan_data)
                prob_adv.train() 
                self._gan_attack(users, prob_adv)
            
            if self.params.keystrokes:
                num_hacked = 0
                exclude = []
                for i in range(0):
                    exclude.append(str(i))

                for user in users:
                    prob_adv=ProbabilisticAdversary(self.params.dataset,
                            clusters=self.params.attack_clusters,
                            components=self.params.gaussian_components)
                    gan_data =self._get_gan_data(exclude=[user.y]+exclude,num=None) 
                    prob_adv.add_data(gan_data)
                    prob_adv.train() 
                    # num_hacked += self._prob_kmeans_attack(user, prob_adv)

                    # actually returns num hacked users - but can treat it as +ve
                    # or -ve here
                    num_hacked += self._gan_attack([user], prob_adv)
     
                print('num hacked users = ', num_hacked)
                print('num secure users = ', len(users) - num_hacked)
                print('percentage of hacked users = ',float(num_hacked)/len(users))

        if self.params.gan_attack:
            if self.params.vgan:
                gan_adv = GANAdversary('test', 'test', clusters=2,
                        num_epochs=5000)
                gan_data = self._get_gan_data() 
                gan_adv.add_data(gan_data)
                gan_adv.train_gans() 
                self._gan_attack(users, gan_adv)
            if self.params.wgan:
                gan_adv = WGANAdversary()
                gan_data = self._get_gan_data(num=None) 
                gan_adv.add_data(gan_data)
                gan_adv.train_gans() 
                self._gan_attack(users, gan_adv)
            if self.params.aegan:
                gan_adv = AdversarialAE()
                gan_data = self._get_gan_data(num=None) 
                gan_adv.add_data(gan_data)
                gan_adv.train_gans() 
                self._gan_attack(users, gan_adv)
           
    def get_classifiers_list(self):
        '''
        Useful for stats.export_cracker etc
        '''
        pass
    
    def _prob_kmeans_attack(self, user, attacker, cracker, cl_name=None):
        '''
        @cracker: pass in a cracker if you are willing to use them on all the
        users (for instance if the attack data comes from a completely
        different sample set). If it is None, then will retrain it based on
        samples generated by attacker.
        '''  
        # yi = user.y 
        if cl_name is None:
            print('cl name was none!!')
            exit(0)

        broke = cracker.crack(user, cl_name, n_samples=100000) 
        
        if broke:
            return 1
        else:
            return 0

    def _gan_attack(self, users, gan_attacker, password_keys=None):
        '''
        TODO: Need to fix this so we print stats about % users broken after n
        tries.
        '''
        classifier_names = [] 
        if self.params.ensemble:
            classifier_names.append('ensemble')
        else:
            for u in users:
                for cl_name in u.classifiers:
                    classifier_names.append(cl_name)
                break
        
        if password_keys is None:
            password_keys = []
            for i in range(len(self.data['0'][0])):
                if not (i % 3 == 1) and self.params.keystrokes:
                    password_keys.append(str(i))
                elif self.params.android:
                    # swipe data
                    password_keys.append(str(i))

        hacked_users = 0 
        secure_users = 0

        num_hack_tries = []
        for cl_name in classifier_names:
            for user in users:
                yi = user.y
                # Each centroid can essentially be viewed as a feature vector.
                broken = False
                i = 0
                while not broken:
                    if i >= self.params.attack_num_tries:
                        break
                    # TODO: return the correct format from all.
                    samples = gan_attacker.generate_attack(password_keys)
                    for sample in samples:
                        sample = np.array(sample)
                        # print(sample)
                        # Note: result is still not in the correct format for cmu
                        # data
                        if self.params.keystrokes:
                            sample = self._add_dependent_features(sample)

                        result = user.test(sample, cl_name)
                        if result[0] > 0:
                            broken = True
                        i += 1        

                if broken:
                    # print('broke user {} in tries {}'.format(yi, i))
                    hacked_users += 1
                    num_hack_tries.append(i)
                else:
                    # print('could not break user {}'.format(yi))
                    secure_users += 1
        
        if self.params.verbose:
            print('num hacked users are ', hacked_users)
            print('num secure users are ', secure_users)

        # print(num_hack_tries)
        if len(num_hack_tries) != 0:
            avg_hack_tries = sum(num_hack_tries) / len(num_hack_tries)
            print('average tries to hack users ', avg_hack_tries)

        percentage_hacked_users = float(hacked_users) / len(users)
        print('percentage of hacked users are: ', percentage_hacked_users)

        return hacked_users

    def _add_dependent_features(self, result):
        '''
        @results: Hold-Key Interval Times - Hold - ... etc format
        Will return a feature vector where the second feature is a sum of Hold
        + key interval times.
        '''
        total_len = len(result) / 2 + len(result)
        features = [None]*total_len
        for i, r in enumerate(result):
            keystroke = i / 2 
            feature = i % 2     # 0 or 1
            # index in features vector: keystroke*3 + feature
            index = keystroke * 3 + feature
            if feature == 1:  
                features[index] = result[i] + result[i-1]
                features[index+1] = r
            else: 
                features[index] = r
        
        self._sanity_check_cmu_features([features])
        return features

    def _sanity_check_cmu_features(self, feature_vectors):
        '''
        Basic test to see Hold + KIT = 2nd feature as in the cmu dataset
        '''
        for features in feature_vectors:
            # note: last key does not have all 3 keystroke features.
            for i in range(0, len(features)-1, 3):
                assert isclose(features[i] + features[i+2], features[i+1])

    def _kmeans_attack(self, users, attacker, impostor_list=None, classifier=-1,
            expert_level=None):
        '''
        FIXME: Need to adapt this to start using cl_names instead of indices!!
        But we aren't using this anyway for now because of cracker so don't
        have to hurry with this.

        Wrapper method for kmeans attack. We take in a list of users we want to
        attack, a list of users whose data we want to use - defaulting to the whole
        dictionary, data, 
        expert_level only used for swipes data...
        '''
        if len(users) == 0:
            return
        
        start = time.time()
        kmeans_attack = {}
        
        total_fails = 0
        
        for user in users:
            yi = user.y
            # Each centroid can essentially be viewed as a feature vector.
            centroids = attacker.kmeans_attack(yi,clusters=self.params.kmeans_cluster_size,
                                        num_impostors=self.params.kmeans_impostors,
                                        num_impostor_samples=self.params.kmeans_impostor_samples)
        
            kmeans_attack[yi] = []

            failed_to_break = True

            # TODO: Should loop over centroids randomly and then exit the loop
            # when we find a break. So we can calculate average tries to break it
            # etc.

            for c in centroids:

                assert(len(c) == self.loader.num_features)

                result = user.test(c, classifier)
                for i, score in enumerate(result):
                    # Initialize the value if needed.
                    if len(kmeans_attack[yi]) == i:
                        kmeans_attack[yi].append(0)
                    if score > 0:
                        kmeans_attack[yi][i] += 1
                        total_fails += 1
                        failed_to_break = False

            if self.params.verbose:
                if failed_to_break:
                    print 'failed to break user ', yi
                else:
                    print 'break user ', yi
        
        if self.params.verbose:
            total_tries = self.params.kmeans_cluster_size*len(users)
            print 'fails / tries = ', float(total_fails) / total_tries

        # Time for stats!
        best_users = []
        for user in users:
            hacked = True
            yi = user.y
            if self.params.verbose:
                print('user ', yi)
            for score in kmeans_attack[yi]:
                if score == 0 and hacked:
                    # Wow this was a damn good user.
                    best_users.append(user)
                    hacked = False
                if self.params.verbose:
                    print score
        
        end = time.time()
        if self.params.verbose:
            print('number of users which we failed to break were ',len(best_users))
            print('kmeans attack took ', end-start)

    def _get_mean_eer(self, users):
        '''
        ret: dict, key:classifier, val: mean eer
        ''' 
        if (len(users) == 0):
            return 0

        ret = defaultdict(float)

        for u in users:
            for name, cl in u.classifiers.iteritems():
                # Edge case in far_classifiers...
                if 'median' in name:
                    continue
                ret[name] += float(u.eers[name]) 

        for name in ret:
            ret[name] = float(ret[name]) / len(users)

        return ret

    def _great_ok_bad(self, users):
        '''
        Stuff I was doing in 229 - doesn't seem required for now.
        '''
        bad_eer = 0
        great_eer = 0
        ok_eer = 0

        great_users = []
        ok_users = []
        bad_users = []

        great_threshold = 0.03      # Not entirely clear what the best value is
        for user in users:
            eer = user.eers['Manhattan']

            if eer > 0.10:
                bad_eer += 1
                bad_users.append(user)
            
            # Anything less than 1 percent is perfect!
            if (eer) < great_threshold:
                great_eer += 1
                great_users.append(user)
            
            # Between 1 and 10 %, life's meh.
            if eer < 0.10 and eer > great_threshold:
                ok_eer += 1
                ok_users.append(user)
         
        # Now we can run our run on the mill attacks on only a subset of the users.
        # Question to decide - for the adversary, do we just throw all possible
        # data at it, or just from bad buckets - could be interesting to see if bad
        # users data also results in successful breakins.
        
        # print 'mean of great users is ', get_mean_eer(great_users)
        # print 'mean of ok users is ', get_mean_eer(ok_users)
        # print 'mean of bad users is ', get_mean_eer(bad_users)
        
        print('great users ---------------')
        print('num = ', len(great_users))
        self._analyze_features(great_users)
        print('ok users next --------------')
        print('num = ', len(ok_users))
        self._analyze_features(ok_users)
        print('bad users time -----------------')
        print('num = ', len(bad_users))
        self._analyze_features(bad_users)

    def _analyze_features(self, users):

        # features = [user.X for user in users]
        features = []
        for user in users:
            for x in user.X:
                features.append(x)
            break

        features = np.array(features)
        print('shape of features = ', features.shape)
        mean_features = np.mean(features, axis=0)
        std_features = np.std(features, axis=0)
        print('shape of mean features = ', mean_features.shape)
        print(mean_features)
        print('std features are: ')
        print(std_features)

    def stats(self):
        '''
        Need to decide what stats to present here.
        '''
        pass

    def print_experiment_details(self):
        '''
        '''
        self.stats.print_params()
    
    def _get_mouse_impostors(self, data, yi, num_per_user=5):
        '''
        TODO: Decide what format to use here.
        Each element is a task object.
        '''
        impostors = []
        for user, mouse_features in data.iteritems():
            if user == yi:
                continue
            for i, task in enumerate(mouse_features.tasks):   
                if i >= num_per_user:
                    break
                impostors.append(task)

        return impostors

    def _train_users(self, data):
        '''
        @ret: users list - with user object for each trained user.
        '''        
        data_hash1 = hashlib.md5(pickle.dumps(data)).hexdigest()
        users = []
        start_time = time.time()
       
        np.random.seed(self.params.seed)
        
        if self.params.verbose:
            print('total num users are ', len(data))

        for i, yi in enumerate(data):

            if i >= self.params.quick_test_num and self.params.quick_test:
                break

            # All reps of this user. Each element in Xi represents the features of
            # the typed password. 
            if yi in self.params.skip_users:
                print 'skipped user ', yi
                continue
            
            # So we can separate out the mouse training process.
            if self.params.keystrokes or self.params.android:
                '''
                TODO: Create new function.
                Basically deals with the point till we create a user, including
                a lot of ugly stuff.
                '''
                train_samples = data[yi]          
                if len(train_samples) < 50:
                    print('skipping user {} because too few samples'.format(yi))
                    continue

                self._sanity_check(train_samples)
                
                if self.params.android: 
                    # impostors = self._get_first_impostors(data, yi)
                    impostors = self._get_impostors(data, yi,
                            num=len(train_samples),
                            samples_per_user=2)
                    train_impostors = []

                elif 'mturk' in self.params.dataset:
                    # Not using train impostors so far...
                    train_impostors = self._get_impostors(data,
                            yi,num=100,samples_per_user=5,seed=2468)
                    
                    # Also, I'm not using get_first_impostors here, because then we
                    # will at least have 600 (or more) impostor samples, as opposed to
                    # just 50 genuine samples...
                    impostors = self._get_impostors(data, yi,
                            num=len(train_samples)/2,
                            samples_per_user=1)

                else:
                    # FIXME: decide how to select train_impostors...
                    train_impostors = self._get_impostors(data,
                            yi,num=len(train_samples)/2, samples_per_user=5)
                    impostors = self._get_first_impostors(data, yi)

                    if self.params.induced_eer:
                        impostors = self._get_kmeans_impostors(data, yi,
                                num=len(train_samples)/2)
                
                self._sanity_check(impostors)

                user = User(yi, np.copy(train_samples), np.copy(impostors),
                        np.copy(train_impostors), self)
            
            elif self.params.mouse:
                if len(data[yi].tasks) < 10:
                    continue
                impostors = self._get_mouse_impostors(data, yi, num_per_user=25)
                user = MouseUser(yi, data[yi], impostors, self) 

            else:
                assert False, 'not supported experiment kind'

            if self.params.svm:
                user.train(lambda: OneClassSVM(), name = 'SVM1')

            if self.params.ae:
                user.train(lambda: Autoencoder([5, 4, 3]), name='Autoencoder')
            
            if self.params.var_ae:
                user.train(lambda: VariationalAutoencoder(dict(n_hidden_recog_1=5,
                                                    n_hidden_recog_2=5,  
                                                    n_hidden_gener_1=5,  
                                                    n_hidden_gener_2=5, 
                                                    n_z=3), 
                                               batch_size=2),
                                               name='VariationalAutoencoder')
            
            if self.params.con_ae:
                start_ac = time.time()
                user.train(lambda: ContractiveAutoencoder(400, lam=1.5),
                name='ContractiveAutoencoder')
                end_ac = time.time()
            
            if self.params.manhattan: 
                user.train(lambda: Manhattan(), name='Manhattan')

            if self.params.random_forests:
                user.train(lambda: RandomForests(n_estimators=self.params.rf_trees),
                        two_class=True, name='RandomForests')

            if self.params.knc:
                user.train(lambda: KNC(n_neighbors=self.params.knc_neighbors), 
                        two_class = True, 
                        name='KNC')

            if self.params.fc_net:
                user.train(lambda: FullyConnectedNetwork(),
                        two_class=True, name='FC_Net')

            if self.params.gaussian:
                user.train(lambda: Gaussian(), name='Gaussian')
            
            if self.params.nearest_neighbors:
                user.train(lambda: NN(), name='NearestNeighbors')
            
            #TODO: this still doesn't work because of the way fit works right
            # now.
            if self.params.pohmm:
                user.train(lambda: Pohmm(n_hidden_states=2,
                                                init_spread=2,
                                                emissions=['lognormal', 'lognormal'],
                                                smoothing='freq',
                                                init_method='obs',
                                                thresh=1e-2))
            if self.params.gaussian_mixture:
                user.train(lambda: GM(), name='Gaussian Mixture')
            
            users.append(user)             
            # This doesn't seem particularly useful. Maybe get stats later, but
            # people did generally well...can just use EER stats instead.
            # Full on test on every random guy we got:
            if self.params.complete_check:
                self._complete_check(data, yi, user)
        
        data_hash2 = hashlib.md5(pickle.dumps(data)).hexdigest()
        assert data_hash1 == data_hash2, 'data hashes diff!'
        end_time = time.time()

        # print 'total time for _train_users function ',(end_time-start_time)

        # Just update the list of used classifiers - not the ideal place to do
        # it but whatever.
        for user in users:
            for cl in user.classifiers:
                self.params.classifiers_list.append(cl)
            break
        return users

    def _sanity_check(self, samples):
        '''
        FIXME
        '''
        # assert len(samples) >= 50, 'user samples fewer than 50'

        for f in samples:
            assert len(f) == self.loader.num_features, 'len of features doesn not match'

    def _get_swipes_impostors(self, data):
        '''
        Selecting impostors based on the Antal et al. paper.
        '''
        pass
    
    def _get_kmeans_impostors(self, data, user, num=200):
        '''
        '''
        # get all samples into array
        X = []
        for d in data:
            # skip current user obviously.
            if d == user:
                continue 
            for x in data[d]:
                X.append(x) 
        X = np.array(X)
        kmeans = KMeans(n_clusters=num).fit(X) 
        clusters = kmeans.cluster_centers_
        return clusters

    def _get_impostors(self, data, user, num=None,
            seed=None, samples_per_user=100):
        '''
        Will find a random number (based on seed) of impostors picked from the
        other users.
        @data: dict with everyone's data.
        @user: just the string value like 0, 4 etc.
        @seed: Seed we are using. (used to be '12')

        FIXME: Simplify and combine this method with get_first_impostors maybe.
        '''
        if seed is None:
            seed = self.params.seed
        
        # if 'greyc' in self.params.dataset:
            # samples_per_user = num / 10
        
        # This should ensure consistent results.
        random.seed(seed)

        # Will be more efficient to make it np array immediately, but whatever.
        impostors = []
        total = 0
        # So we don't repeat users.
        used_users = [user]
        total_users = len(data)
        it = 0
        while True:
            # Break condition - if we have run through all the 'impostor'
            # users, then just break out.
            if (total >= num) or it >= total_users-1:
                break
            
            # FIXME: This won't work in cases where the users aren't labeled 0....N
            user = str(random.randrange(total_users))
            if user in used_users:
                continue 

            X = data[user]
            # Pick a random collection of (samples_per_user) examples from it.
            # Note: We don't want to pick the first sample because they might be
            # bad samples anyway...
            selected_x = random.sample(X, samples_per_user)

            for x in selected_x:
                impostors.append(x)
            
            used_users.append(user)
            total += samples_per_user
            it += 1 

        impostors = np.array(impostors)
        
        if self.params.exclude_trained_samples:
            for impostor in impostors:
                self.used_samples.append(hashlib.sha1(str(impostor)).hexdigest())
        
        return impostors

    # FIXME: include these here?
    def _get_first_impostors(self, data, user, num_per_impostor=5, skip_first_features=0):
        '''
        First n samples from all other users.
        
        user = current user. string.

        Note: Can basically replicate results from maxion et al etc with this - but
        with the more random get_impostors method, the results were slightly worse.
        '''
        ret_list = []
        
        # FIXME: Get rid of special case conditions....just call this function
        # with different parameters...
        if 'greyc' in self.params.dataset:
            num_per_impostor = 4

        elif 'android' in self.params.dataset:
            num_per_impostor = 2

        elif 'mturk' in self.params.dataset:
            num_per_impostor = 1
            skip_first_features = 2

        for cur_user in data:

            if(cur_user == user):
                continue
            
            cur_features = data[cur_user]
            
            # Select first 4 users.
            for i in range(num_per_impostor):
                if i >= len(cur_features):
                    break

                ret_list.append(cur_features[i])

        if self.params.exclude_trained_samples:
            for impostor in ret_list:
                self.used_samples.append(hashlib.sha1(str(impostor)).hexdigest())

        return ret_list

    def _complete_check(self, data, yi, user):
        
        success_impostor = 0
        fail_impostor = 0

        for k in data:
            if k == yi:
                continue
            impostors_new = data[k]
            for x in impostors_new:
                scores = user.test(x, -1)
                for score in scores:
                    if score > 0:
                        success_impostor += 1
                    else:
                        fail_impostor += 1

        impostor_percentage = float(success_impostor) / (fail_impostor+success_impostor)
        print 'complete impostor success percentage is ', impostor_percentage

# Figure out what to do with this later - being called from one of the no
# longer used attack methods. 
def robustness_check(Xi_new, impostors_new, user, classifier=-1):
    '''
    
    @Xi and impostors are just features of these guys.
    @user: user object.
    @classifier: index of the classifier, -1 for all.

    Things we want:
        1.  % of failures (both types)  
        2.  want to separate it by class of users. 
        I guess (1) should be comparable to eer from our training phase.
    '''
    # next two loops are just a sanity check.

    success_impostor = 0
    fail_impostor = 0

    success_genuine = 0
    fail_genuine = 0
    
    for x in Xi_new:
        scores = user.test(x, classifier)
        for score in scores:
            if score > 0:
                success_genuine += 1
            else:
                fail_genuine += 1
    
    for x in impostors_new:
        scores = user.test(x, classifier)
        for score in scores:
            if score > 0:
                success_impostor += 1
            else:
                fail_impostor += 1
    
    genuine_percentage = float(success_genuine)/(fail_genuine+success_genuine)

    impostor_percentage = float(success_impostor) / (fail_impostor+success_impostor)
    
    # return the error rate.
    total = fail_genuine + success_genuine + fail_impostor + success_impostor
    error_rate = float(fail_genuine + success_impostor) / total
    return error_rate
