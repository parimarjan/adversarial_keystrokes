import csv
import numpy as np
import os
import json
from mouse_features import MouseFeatures
from util import *
import hashlib

from math import sqrt
from joblib import Parallel, delayed

def _gen_mouse_pickle_name(user, hashed_dict):
    ''' 
    TODO: Fix this!
    '''
    hashed_input = hashlib.sha1(str(hashed_dict)).hexdigest() 
    name = user + hashed_input[0:5] + '.pickle' 
    return './pickle/' + name

def process_dict(user, hashed_dict):
    
    data = {}
    print('in process dict!')
    new_dict = {}
    print 'user is: ', user
    print('len of hashed dict is: ', len(hashed_dict))
    if not isinstance(hashed_dict, dict):
        print('bad dict')
        return {}

    for k, v in hashed_dict.iteritems():
        for i, lst in enumerate(v):
            # assert not new_key in new_dict, 'new key must not bethr'
            new_dict[k + str(i)] = lst
    
    print('len of new dict is: ', len(new_dict))
    pickle_name = _gen_mouse_pickle_name(user, new_dict)
    mouse_features = do_pickle(True, pickle_name, 1, MouseFeatures,
            new_dict, num_single_clicks=0, ref_task=True)        
    
    # Just testing features without creating ref task:
    # mouse_features = MouseFeatures(new_dict, num_single_clicks=0,
            # ref_task=False)

    print('num tasks done were: ')
    print(len(mouse_features.tasks)) 
    data[user] = mouse_features
    return data

class Loader():
    """
    loads in different datasets etc.
    """

    def __init__(self, exp):
        """
        @dataset: name of the file
        """
        # FIXME: These should be set from the params or somethign
        self.num_features = 0
        self.skip_inputs = 0
        self.skip_features = []

        # files and stuff 
        self.root_dir = os.path.dirname(os.path.realpath(__file__))
        self.root_dir = os.path.join(self.root_dir, './..')
        self.data_dir = os.path.join(self.root_dir, 'datasets')
        self.dataset = os.path.join(self.data_dir, exp.params.dataset)
        self.params = exp.params

    def load_data(self):
        """
        """
        dataset = self.dataset

        if "cmu" in dataset:
            data = self._read_cmu_data(dataset)  
        elif "mturk" in dataset:
            data = self._read_cmu_data(dataset)
        elif "greyc" in dataset:
            assert False, 'untested after restructuring'
            data = self._read_greyc_data()
        # FIXME: Deal with expert_level later
        elif "android" in dataset: 
            data, expert_level = self._read_swipe_data(dataset)
        elif "mouse" in dataset:
            data = self._read_mouse_data(dataset)
        else:
            assert False, "Please specify a dataset to read"

        return data
    
    # TODO:
    
    def _read_mouse_data_mp(self, dataset):
        '''
        Just read the json in, and pass it ahead for now.
        Each value is a dictionary again - but the MouseFeatures class knows
        how to deal with it.
        '''
        print('multiprocessing!')
        data = {}
        results = []
        with open(dataset) as data_file:    
            d = json.load(data_file)
            print('num users are ', len(d))
            with Parallel(20) as parallel:
                results = parallel(delayed(process_dict)(user,hashed_dict) for user, hashed_dict in d.iteritems())

        print('done with multiproc!')
        for r in results:
            data.update(r)

        print(data.keys())
        return data
        

    def _read_mouse_data(self, dataset):
        '''
        Just read the json in, and pass it ahead for now.
        Each value is a dictionary again - but the MouseFeatures class knows
        how to deal with it.
        '''
        data = {}
        with open(dataset) as data_file:    
            d = json.load(data_file)
            print('num users are ', len(d))
            for user, hashed_dict in d.iteritems():
                new_key = 0
                new_dict = {}
                print 'user is: ', user
                print('len of hashed dict is: ', len(hashed_dict))
                if not isinstance(hashed_dict, dict):
                    print('bad dict')
                    continue

                for k, v in hashed_dict.iteritems():
                    for lst in v:
                        assert not new_key in new_dict, 'new key must not bethr'
                        new_dict[str(new_key)] = lst
                        new_key += 1
                
                print('len of new dict is: ', len(new_dict))
                pickle_name = self._gen_mouse_pickle_name(user, new_dict)
                mouse_features = do_pickle(True, pickle_name, 1, MouseFeatures,
                        new_dict, num_single_clicks=0)        
                print(mouse_features.ref_feature_task)
                
                # Just testing features without creating ref task:
                # mouse_features = MouseFeatures(new_dict, num_single_clicks=0,
                        # ref_task=False)

                print('num tasks done were: ')
                print(len(mouse_features.tasks)) 
                data[user] = mouse_features
        
        # with open('data/old_mouse.json') as data_file:
            # d = json.load(data_file)
            # print('num users are ', len(d))
            # for user, hashed_dict in d.iteritems():
                # print('user is ', user)
                # if 'rish' == user or 'Rish' == user or 'ABC' == user:
                    # continue

                # pickle_name = self._gen_mouse_pickle_name(user, hashed_dict)
                # mouse_features = do_pickle(True, pickle_name, 1, MouseFeatures,
                                # hashed_dict, num_single_clicks=0)        
                # print(mouse_features.ref_feature_task)
                # print('num tasks done were: ')
                # print(len(mouse_features.tasks)) 
                # print('adding user to dict: ', user)
                # data[user] = mouse_features

        return data

    def _gen_mouse_pickle_name(self, user, hashed_dict):
        ''' 
        TODO: Fix this!
        '''

        hashed_input = hashlib.sha1(str(hashed_dict)).hexdigest() 
        name = user + hashed_input[0:5] + '.pickle' 
        return './pickle/' + name

    def _read_swipe_data(self, filename, skip_input=None):
        """
        Returns a dictionary with user names as key, and a list of vectors
        (features) as the value.
        """
        
        # Maps from user : expert level. Useful for constructing attacks...
        expert_level = {}

        # FIXME: Generalize this to be able to skip any feature etc.
        FINGER_AREA = 11

        # Start_col is where the features start.
        # should just ignore android_user.csv maybe - as the features are just
        # a subset of android_swipes.csv?
        assert 'android_swipes.csv' in filename, 'Just support full csv'

        '''
        different subsets of features selected in Antal et al. 
            - All 11 features 
            - 8 touch features
            - 3 gravity features
        ''' 
        # START_COL to LAST_COL inclusive will be extracted.
        if self.params.swipe_all_features:
            FEATURE_LENGTH = 11
            if self.params.skip_fing_area:
                FEATURE_LENGTH -= 1
            
            START_COL = 4
            LAST_COL = 14
            USER_COL = 15

        elif self.params.swipe_touch_features:
            FEATURE_LENGTH = 8
            if self.params.skip_fing_area:
                FEATURE_LENGTH -= 1
            
            START_COL = 4
            LAST_COL = 11
            USER_COL = 15

        elif self.params.swipe_gravity_features:
            FEATURE_LENGTH = 3
            START_COL = 12
            LAST_COL = 14
            USER_COL = 15

        else:
            assert False, 'specify swipe features'

        # Will map each user_name to its new index, which is just increasing
        # sequence from 0....98 etc.
        data = {}    
        users = {}
        user_index = 0
        import csv

        f = open(filename)
        csv_f = csv.reader(f)
        
        # What values we initialize them to shouldnt really make a difference as
        # they will be rest in the loop in the first iteration itself.
        prev_user = '100'
        new_user = True

        for i, row in enumerate(csv_f):
            # skip header.
            if (i == 0):
                continue 
            
            # This value can be changed between 40 or 98.
            
            # This will be s00.. codes used in the cmu dataset.
            user = row[USER_COL]
            
            # Doing all these acrobatics so we can rename each user 0....51 instead
            # of the real names (not required, but whatever, was convenient at some
            # places)

            if user not in users:
                users[user] = str(user_index)
                # Add an entry in data so we can start adding stuff there.
                data[str(user_index)] = []
                # Update to get the next key for the next user.
                user_index += 1
            
            # I'm assuming the first 40 users in the dataset were the ones used by
            # Antal et all when doing their study.
            if user_index > self.params.swipes_num_users:
                continue

            # This will give the update value of user.
            user = users[user]
             
            features = []
            
            # these are all the features.

            for i in range(START_COL, LAST_COL+1, 1): 
                
                if self.params.skip_fing_area:
                    if i == FINGER_AREA:
                        continue
                features.append(float(row[i]))
            
            data[user].append(features)
            
            # It doesn't matter if we update this once or many times.
            expert_level[user] = row[1]
        
        # Turn it into np array so don't have to mess around with later parts.
        # Also, it's efficient!

        for i in data:
            data[i] = np.array(data[i])

        self.num_features = len(data['0'][0])

        assert self.num_features == FEATURE_LENGTH, 'swipes feature length \
        does not match'

        return data, expert_level


    def _read_cmu_data(self, filename):
        """
        @filename: 
        @ret: dict, with y : X's.
        """
        data = {}    
        # Will map each user_name to its new index, which is just increasing
        # sequence from 0....50.
        users = {}
        user_index = 0

        f = open(filename)
        csv_f = csv.reader(f)
        
        # What values we initialize them to shouldnt really make a difference as
        # they will be rest in the loop in the first iteration itself.
        prev_user = '100'
        new_user = True
        skipped_rows = 0

        for i, row in enumerate(csv_f):
            # skip header.
            if (i == 0):
                continue 
            
            user = row[0]
            
            # Doing all these acrobatics so we can rename each user 0....51 instead
            # of the random serial nums in cmu or mturk

            if user not in users:
                users[user] = str(user_index)
                # Add an entry in data so we can start adding stuff there.
                data[str(user_index)] = []
                # Update to get the next key for the next user.
                user_index += 1
            
            # This will give the update value of user.
            user = users[user]
            
            if user != prev_user:
                skipped_rows = 0
                prev_user = user
            
            if (skipped_rows < self.skip_inputs):
                skipped_rows += 1
                continue

            # Now we need to add all the values from column 3...n as features.
            # Also, convert them to floats.
            features = []
            
            # these are all the features.
            num_zeros = 0
            for i, j in enumerate(range(3,len(row))): 
                # in vs not in.
                if ((i/3) in self.skip_features):
                    continue 
                if row[j] == 0.00:
                    num_zeros += 1

                features.append(float(row[j]))
            
            if np.isnan(np.array(features).all()) or \
                    np.isinf(np.array(features).all()):
                print('skipped input because nan!')
                continue
            if num_zeros > 1:
                print('skipped row because zeros')
                continue 
            data[user].append(features)
        
        for i in data:
            data[i] = np.array(data[i])

        # FIXME: We are assuming all feature vectors are of the same size here.        
        self.num_features = len(data['0'][0])
        if self.params.verbose:
            print("features = ", self.num_features)

        return data


