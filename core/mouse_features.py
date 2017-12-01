from collections import defaultdict
import math
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import time

'''
Things to do:
    1. Eigenspace transformation / PCA - do we even need this?
    2. Need to look more into the normalization step.
'''
class MouseFeatures():

    def __init__(self, raw_tasks, num_single_clicks=0, num_double_clicks=8,
            ref_task=True):
        '''
        Takes in all the tasks, raw_vectors outputed by our mouse movements
        task, done by one user and computes on those.

        @num_single_clicks: single clicks in one task
        @num_double_clicks: double clicks in one task
        '''
        self.do_norm = False
        self.num_single_clicks = num_single_clicks
        self.num_double_clicks = num_double_clicks
        self.mus = None
        self.stds = None
        
        # self._sanity_check(raw_tasks) 
        self.raw_tasks = raw_tasks
        # this will be a bunch of task objects.
        self.tasks = self._get_tasks()
        # print('number of tasks were: ', len(self.tasks)) 

        if ref_task:
            self._create_ref_feature_vector()

    def _get_tasks(self):
        '''
        Creates a list of tasks from the raw_tasks.         
        In our task, we sent data at the granularity of one task, so this is
        straightforward.
        '''
        start = time.time()
        tasks = []
        for _, task in self.raw_tasks.iteritems():
            try:
                mouse_task = MouseTask(task)
            except Exception as e:
                print('got exception in mouse task: ', e)
                continue

            tasks.append(mouse_task)
        
        print('_get tasks took: ', time.time() - start)
        return tasks

    def _create_ref_feature_vector(self):
        '''
        Each user will basically have 'her' own ref_feature_vector - and all future
        vectors will be compared to this vector to get the new distance_vector.
        
        TODO: This is basically of the type returned by
        MouseTask.get_feature_vector.
        
        Note the difference between distance vectors and feature vectors. The
        procedural features of the feature vector can be of different lengths.

        Feature Vectors can be compared together to generate distance vectors.

        Distance vectors are what will be fed to the classifier - and this
        won't have the weird issues like different lengths etc.
        '''
        start = time.time()
        print('in create ref feature vector!')
        all_dist_vectors = []
        # For each pair of tasks.
        best_distance = float('inf')
        best_task = None
        
        # Can also update the self.mus and self.stds here
        pre_computed = defaultdict(list)

        for i, task1 in enumerate(self.tasks):
            # print('i = ', i)
            total_distance = 0
            for j, task2 in enumerate(self.tasks):
                if i == j:
                    continue
                # print('j = ', j)
                # compute distance between these and append to total distance.
                if pre_computed[str(i) + '+' + str(j)] != []:
                    dist_vector = pre_computed[str(i)+str(j)]
                    # print('it is: ', dist_vector)
                else:
                    dist_vector = self._get_distance_vector(task1, task2)
                    pre_computed[str(j) + '+' + str(i)] = dist_vector

                if i < j:
                    all_dist_vectors.append(dist_vector)

                total_distance += np.linalg.norm(np.array(dist_vector))

            if total_distance < best_distance:
                best_distance = total_distance
                best_task = task1
        
        self.ref_feature_task = best_task

        # compute mu / std for normalization - which we can use for normalized
        # calculation of _get_distance_vector in the future.
        all_dist_vectors = np.array(all_dist_vectors)
        
        if self.do_norm:
            if len(all_dist_vectors) > 1:   # otherwise stds will be 0
                self.mus = np.mean(all_dist_vectors, axis=0)
                self.stds = np.std(all_dist_vectors, axis=0)
                assert len(self.mus) == len(all_dist_vectors[0])
        
        # print('find ref feature vector took ', time.time() - start)

    def _sanity_check(self, raw_tasks):
        '''
        ''' 
        for h, task in raw_tasks.iteritems():
            # each task done by the user. Will generate a feature vector for each
            # task.
            single_clicks = 0
            double_clicks = 0
            
            # compute_holistic_features(task)
            for step in task: 
                if step[3] == 1:
                    single_clicks += 1
                elif step[3] == 2:
                    double_clicks += 1
            
            # assert single_clicks == self.num_single_clicks, 'should be same!'
            print('double clicks = ', double_clicks)
            assert double_clicks >= self.num_double_clicks, 'should be same!'

    def _get_distance_vector(self, a, b):
        '''
        Two MouseTasks to distance vector.

        @ret: distance_vector. Its length should be same as the feature vector
        length.
        '''
        feature1 = a.get_feature_vector()
        feature2 = b.get_feature_vector()
        distance_vector = []
        # For same length features, we will use manhattan distance
        # Procedural features are different length so will use dtw

        for i, f1 in enumerate(feature1):
            f2 = feature2[i]
            if not isinstance(f2, list): 
                # calculate manhattan distance between two values.
                # FIXME: Is this correct?
                distance_vector.append(abs(f2-f1))
            else:
                # Assume these are the procedural features, and calculate DTW
                # distance between them
                x = np.array(f1)
                y = np.array(f2)
                distance, path = fastdtw(x, y, dist=euclidean)
                distance_vector.append(distance)

        assert len(feature1) == len(distance_vector), 'test'  
        # TODO: Normalization: I think let us just do it ONLY for test
        # samples, and use the mu and std vectors based on just this user's
        # training data?
        if self.mus is not None:
            # can normalize this!
            # FIXME: Should we ensure this is positive?
            distance_vector = np.divide((distance_vector - self.mus),
                    self.stds)
            # both are equivalent methods:
            # distance_vector2 = (distance_vector - self.mus) / self.stds 
            # print(np.mean(distance_vector2))

        return distance_vector

    def get_all_distance_vectors(self, all_tasks):
        '''
        Gives us a whole bunch of mouse tasks, and we compare it with the
        ref task - and generate distance vectors for each of these.
        '''
        # FIXME: while training svm should we avoid if task is the reference
        # task? Shouldn't be needed.
        distance_vectors = []
        for task in all_tasks:
            d = self._get_distance_vector(task, self.ref_feature_task)
            distance_vectors.append(d)
        
        return np.array(distance_vectors)

class MouseTask():

    def __init__(self, task, num_actions=8):
        '''
        @task: single mouse task, as defined in our task. Will be made up of a
        series of n-Actions. n = 16 in shen et al.
        
        Can create things from this like: holistic features, procedural features
        etc
        '''
        self.num_actions = num_actions
        # holistic features combine all the actions.
        self.holistic_features = defaultdict(list) 
        self.procedural_features = defaultdict(list)     
        
        self.actions = self._get_actions(task)
        assert len(self.actions) == num_actions, 'num actions should \
            be consistent'
        
        # feature vector for this task
        self.feature_vector = self._compute_feature_vector()

    def get_feature_vector(self):
        '''
        Convert the task object into a feature vector - in case of Shen et al.,
        this would be the 74 feature vector with:
        
        FIXME: Right now I just do two intervals in the double clicks, but
        check what three intervals shen was talking about.

        Basically we will act on the vector of actions.

            Holistic Features:
                10: click related features -- loop over the actions and get
                these.
                    - mean, std of single clicks (2)
                    - mean, std of double clicks (2)
                    - mean, std of 3 interval times of double clicks (3*2 = 6)
                
                The rest are all one per mouse action:

                16: Time related feautures (Time elapsed)
                16: Distance related features (Movement Offset)

            Procedural Features, 1 for each task:
                The big difference here is that each of these 16 features can
                have a variable length. Will use DTW on these when computing
                distances.

                16: speed related features
                16: acceleration related features
        '''
        return self.feature_vector

    def _compute_feature_vector(self):
        '''
        '''
        # print('in compute feature vector!')
        feature_vector = []
        # Time elapsed features
        for action in self.actions:
            feature_vector.append(action.time_elapsed)
        # Distance features  
        for action in self.actions:
            feature_vector.append(action.movement_offset)

        # click features, 1: single clicks. Each action may or may not have
        # single click - but we will average these times over the 16 actions in
        # the task. Take mean and std as features.
        single_click_times = []
        for action in self.actions:
            if action.double_click:
                continue
            for click in action.click_times:
                single_click_times.append(click)
        
        if len(single_click_times) > 0:
            single_click_times = np.array(single_click_times)
            feature_vector.append(np.mean(single_click_times))
            feature_vector.append(np.std(single_click_times))
        
        # click features, 2: double clicks, interval 1
        double_click_times = []
        for action in self.actions:
            if action.single_click:
                continue
            double_click_times.append(action.double_click_times[0])

        double_click_times = np.array(double_click_times)
        feature_vector.append(np.mean(double_click_times))
        feature_vector.append(np.std(double_click_times))

        # click features, 2: double click interval 2
        double_click_times = []
        for action in self.actions:
            if action.single_click:
                continue
            double_click_times.append(action.double_click_times[1])

        double_click_times = np.array(double_click_times)
        feature_vector.append(np.mean(double_click_times))
        feature_vector.append(np.std(double_click_times))

        # click features, 2: double click interval, complete interval
        double_click_times = []
        for action in self.actions:
            if action.single_click:
                continue
            double_click_times.append(action.double_click_times[2])

        double_click_times = np.array(double_click_times)
        feature_vector.append(np.mean(double_click_times))
        feature_vector.append(np.std(double_click_times))

        # procedural features, speed - NOTE: Here each feature is an array, and
        # can be of different length.
        for action in self.actions:
            feature_vector.append(action.speed_curve)

        #procedural features, acceleration:
        for action in self.actions:
            feature_vector.append(action.accelerations)

        return feature_vector

    def _get_actions(self, task):
        '''
        @ret: a bunch of Mouse action objects
        ''' 
        # print('in get actions!')
        actions = []
        sequence = []
        for v in task:
            sequence.append(v)
            if v[3] == 1 or v[3] == 2:
                # dumb way to avoid weird edge case issues
                try:
                    action = MouseAction(sequence)
                except Exception as e:
                    print('got exception: ', e)
                    continue

                actions.append(action)        
                sequence = []
                # print('---------------------')
        
        return actions

class MouseAction():

    def __init__(self, sequence):
        '''
        Each action ends with a single click or a double click. So it will be a
        vector of [mx, my, time, click], where click is 0, 1 or 2 for no click,
        single click and double click.

        sequence_i is: (x,y,time,mouse)
        mouse = 0: not clicked
                1: single click (completed)
                2: double click (completed)
                3: mouse down
                4: mouse up
        '''  
        start = time.time()
        # which direction, and how much distance did this action cover?
        self.action_summary = None
        self.single_click = False
        self.double_click = False

        if sequence[-1][3] == 1:
            self.single_click = True
        elif sequence[-1][3] == 2:
            self.double_click = True
        else:
            for v in sequence:
                print(v)
            assert False, 'last value wasnt click?'

        # Holistic features - time elapsed,movement offset, click mean, click
        # std, double_click mean, std; double_click extra.
        # for v in sequence:
            # print(v)
        self.time_elapsed = sequence[-1][2] - sequence[0][2]
        # print('time elapsed is: ', self.time_elapsed)

        # Offset calculation (ideal distance, and what is actually travelled)
        x1, y1 = sequence[0][0], sequence[0][1]
        x2, y2 = sequence[-1][0], sequence[-1][1]
        ideal_dist = math.hypot(x2-x1, y2-y1)
        # print('ideal dist = ', ideal_dist)
        practical_dist = 0
        for i in range(1, len(sequence)):
            # add distance from prev point to current point
            x1, y1 = sequence[i-1][0], sequence[i-1][1]
            x2, y2 = sequence[i][0], sequence[i][1]
            practical_dist += math.hypot(x2-x1, y2-y1)
        # print('practical dist is ', practical_dist)
        self.movement_offset = practical_dist - ideal_dist

        #TODO: Should we deal with this special case scenario?
        if self.movement_offset > 500: 
            print('movement offset was greater than 500!')
        
        # click features.
        # Basically just take every click_down - click_up pair, and compute
        # time between them.
        time_downs = [x[2] for x in sequence if x[3] == 3]
        time_ups = [x[2] for x in sequence if x[3] == 4]
        if self.single_click:
            # print(time_downs)
            # assert len(time_downs) == 1, 'single click'
            self.click_times = [time_ups[0] - time_downs[0]]

        elif self.double_click:
            # print(time_downs)
            # assert len(time_downs) == 2, 'double click'
            # just assuming that if the guy pressed multiple times here, then
            # we only consider the first couple of tries
            self.double_click_times = []
            # All these 3 things were being used by Shen.
            if len(time_ups) < 2:
                raise Exception('mouse up/down times not recorded')

            self.double_click_times.append(time_ups[0] - time_downs[0])
            self.double_click_times.append(time_ups[1] - time_downs[1])
            self.double_click_times.append(time_ups[1] - time_downs[0])

        # Procedural Features: speed, acceleration. We can get a vector of
        # distances per 100 ms or so?
        '''
        Might want to do a more principled way to calculate speeds etc rather
        than the naive, and possibly, noisy thing I do by considering every
        time position is changed...Can also be useful for the ideal distance vs
        practical distance feature above.
        https://dsp.stackexchange.com/questions/9498/have-position-want-to-calculate-velocity-and-acceleration
        '''
        # speed: 1. we'll compute it per 5 values - so each value in the speed
        # array will be per 100 m/s
        # Option 2: Just compute it every time a mouse movement occurs - so
        # will get a bunch of speed: pixels/ms points.
        
        self.speed_curve = []
        timings = []    # useful for computing acceleration

        x1, y1, t1 = sequence[0][0], sequence[0][1], sequence[0][2]
        self.speed_curve.append(0)
        timings.append(t1)

        for i in range(1, len(sequence)):
            # add distance from prev point to current point
            x2, y2, t2 = sequence[i][0], sequence[i][1], sequence[i][2]
            dist = math.hypot(x2-x1, y2-y1)
            if dist > 0:
                # only need to calculate speed here.
                self.speed_curve.append(float(dist) / (t2-t1))
                timings.append(t2)
                # update x1, y1, t1
                x1, y1, t1 = x2,y2,t2
        
        # acceleration: same setup as above (just look at the speeds) and see how
        # they were changing...
        self.accelerations = []

        for i in range(1, len(self.speed_curve)):
            # add distance from prev point to current point
            # x1, y1 = self.speed_curve[i-1][0], self.speed_curve[i-1][1]
            # x2, y2 = self.speed_curve[i][0], self.speed_curve[i][1]
            dx = self.speed_curve[i] - self.speed_curve[i-1]
            tx = timings[i] - timings[i-1]
            assert tx > 0, 'time should be increasing'
            self.accelerations.append(float(dx) / tx)
        
        # print('mouse action took ', time.time() - start)
