from core.experiment import *

datasets = ['mturk_letmein.csv', 'mturk_password.csv', 'mturk_123456789.csv',
            'mturk_mustang.csv', 'mturk_abc123.csv']
COMBINE_MTURK = False
def main():

    if COMBINE_MTURK:
        # temporary stuff to combine all MTurk datasets
        results = []
        for d in datasets:
            experiment = Experiment()  
            # run this stuff for every dataset.
            experiment.params.dataset = d
            # need to load the new dataset
            experiment.loader = Loader(experiment)
            experiment.data = experiment.loader.load_data()
            experiment.print_experiment_details()
            experiment.start()    
            
            if experiment.params.cracker or experiment.params.prob_then_kmeans:
                results.append(experiment.stats.export_cracker(verbose=True))

        # Combine everything at this stage, and then print the info in the same
        # form as before.
        final_result = {}
        for k in results[0]:
            # average the values of k into final_result
            if isinstance(results[0][k], list):
                # need to average this shit up in final_result.
                final_result[k] = [0]*len(results[0][k])
                for r in results:
                    for i, el in enumerate(r[k]):
                        final_result[k][i] += float(el) / len(results) 

            else:
                # must be a string, name of classifier
                final_result[k] = results[0][k]

        print(final_result)
    else:    
        # Let's set up the experiment based on the default values, or provided
        # arguments.
        experiment = Experiment() 
        # experiment.probability_analysis()
        # experiment.cluster_analysis()
        experiment.print_experiment_details()
        experiment.start()    
        if experiment.params.cracker or experiment.params.prob_then_kmeans:
            experiment.stats.export_cracker()

    # print some stats - need to implement experiment. functions for that
        
if __name__ == '__main__':
    
    main()
  
