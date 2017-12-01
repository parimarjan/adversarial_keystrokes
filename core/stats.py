import hashlib
import csv
import os

from collections import OrderedDict

class Stats():
    '''
    Will store all the stats about a running experiment, and provide functions
    to print them / export them to csv/latex?!! or make graphs etc.
    '''

    def __init__(self, params):
        '''
        '''
        self.params = params        
        # key will be user - and value will be a list of tuples about each
        # generated attack vector - attempt, monaco_norm, result 
        self.monaco_norm_effect = {}
        
        # FIXME: Base this on self.params.loader.root_dir or sth
        self.dir = './stats/'

        # Dict of all classifiers and the eer scores
        self.mean_eer = {}
        self.cracker = {}

        # self.cracker_num_tries = [0, 9, 49, 99, 999]
        self.cracker_num_tries = []
        for i in range(100):
            self.cracker_num_tries.append(i)
        
        name = params.dataset + '_cracker.csv'
        if params.digraph_attack:
            name = 'digraph_' + name

        self.cracker_file_name = os.path.join(self.dir, name)

    def print_params(self):
        '''
        Print all the params - to stdout, or a file, or whatever.
        '''
        # TODO:
        # for k, v in self.params.__dict__.iteritems():
            # if v:
                # print('{} : {}'.format(k,v))
        
        print('digraph attack: {}, dataset: {}, prob_then_kmeans: {}, \
        password_keys: {}'.format(self.params.digraph_attack,
            self.params.dataset, self.params.prob_then_kmeans,
            self.params.password_keys))
    
    def _get_monaco_norm_filename(self):
        '''
        '''
        hashed_params = hashlib.sha1(str(self.params)).hexdigest()
        
        attack_type = ''
        if self.params.cracker:
            attack_type += 'cracker_'
        if self.params.kmeans_attack:
            attack_type += 'kmeans_'

        file_name = self.dir + 'monaco_detailed_' + attack_type + hashed_params + '.csv'

        return file_name

    def export_eer(self):
        '''
        Prints out the eer of all the classifiers.
        '''
        pass

    def export_cracker(self, verbose=True):
        '''
        Just print out the stuff.
        '''        
        # Assuming only one classifier - because cracker works well just in
        # that case.
        reports = {}
        for cl_name in self.cracker:
            reports[cl_name] = self.cracker[cl_name].report()

        rows = [] 
        for cl_name in reports:
            # one row for each report!
            report = reports[cl_name]
            results = OrderedDict()
            # FIXME: 
            if cl_name == 'ensemble':
                cl_name += str(self.params.classifiers_list)

            results['Classifier Name'] = cl_name

            percentages = []
            for alg, percentage_cracked in report.iteritems():
                results[alg] = []
                for i in self.cracker_num_tries:
                    if i >= len(percentage_cracked):
                        break
                    # Each element should be a different row
                    # key = alg + str(i)
                    results[alg].append(round(percentage_cracked[i], 2))
            
            if verbose:
                print 'results are ', results
                print '*********************************************'
            
            # TODO: Write header            
            # with open(self.cracker_file_name, 'a') as csvfile:
                # wr = csv.writer(csvfile, delimiter=' ',
                                        # quotechar='|', quoting=csv.QUOTE_MINIMAL)

                # for alg, percentage_cracked in results.iteritems():
                    # for i, percentage in enumerate(percentage_cracked):
                        # print('alg: {}, i : {}, cracked: {}'.format(alg, i, percentage))
                        # row = []
                        # row.append(alg)
                        # row.append(str(i) + ': ' + str(percentage))
                        # wr.writerow(row)

            # with open(self.cracker_file_name, 'a') as f:  
                # w = csv.DictWriter(f, results.keys())
                # w.writeheader()
                # w.writerow(results)
        
        return results


    def export_monaco_norm(self):
        '''
        Use the dictionary to write out details to a csv file
        '''
        file_name = self._get_monaco_norm_filename()

        header = ['user', 'attack_vector', 'normalized_attack_vector',
                    'threshold', 'score']

        f = open(file_name, 'w+')
        results = csv.writer(f)
        results.writerow(header)
        
        for user in self.monaco_norm_effect:
            # print("user is ", user)
            # print("len of vals is ", len(self.monaco_norm_effect[user]))
            for row in self.monaco_norm_effect[user]:
                row.insert(0, user) 
                # write it out to the file
                results.writerow(row)
