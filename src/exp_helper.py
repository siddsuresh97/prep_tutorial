import logging
from statistics import linear_regression
import time
import os

from src.preprocess import *
from src.evaluate_performance import *
from src.data_to_features import *

def run_exp(exp_name, features_dir, subsampling_levels, conditions, 
            intermediate_layer_names, results_dir, result_fname, performance_measure):
    logging.info("Running experiement %s" % exp_name)
    if exp_name == '1c':
        stim_type = 'test_stim'
        analysis_fn = logistic_regression
        analysis_type = 'logistic_regression'
    elif exp_name in ['1a', '1b']:
        if exp_name == '1a':
            stim_type = 'ind'
        elif exp_name == '1b':
            stim_type = 'test_stim'
        analysis_fn = linear_regression
        analysis_type = 'linear_regression'
    else:
        logging.debug("NOT IMPLEMENTED FOR OTHER EXPERIMENTS")
    if "1" in exp_name:
        set_sizes = conditions
        for set_size in set_sizes:
            logging.info('########  SET SIZE %d #########'%set_size)
            intermediate_layer_feats_dir = os.path.join(features_dir, 'set_size_{}/'.format(set_size))
            start_time = time.time()
            number_of_random_features = subsampling_levels
            results_test = {}
            # results_number_of_samples = {}
            for layer in intermediate_layer_names:
                logging.info('####################       %s          ##########################'%layer)
                network_results_test = {}
                # network_results_number_of_samples = {}
                # test_stim contains circles of multiple set sizes and ind stim only contains a single circle
                data, labels = get_data_and_labels(stim_type = stim_type,
                                                layer = layer,
                                                intermediate_layer_feats_dir = intermediate_layer_feats_dir,
                                                analysis_type = analysis_type, 
                                                condition = set_size)
                for random_features_number in number_of_random_features:
                    network_results_test[random_features_number] = {}
                    # network_results_number_of_samples[random_features_number] = {}
                    if data.shape[1]<random_features_number:
                        network_results_test.update({random_features_number:float('nan')})
                        continue
                    # TO DO :- only sparse distribution now, maybe implement dim reduction too
                    subsample_data = subsample_img_features(data, random_features_number, 100)
                    if analysis_type == 'logistic_regression':
                        for circle_size in range(labels.shape[1]):
                            evalutaion_measure = analysis_fn(subsample_data, labels[:,circle_size], performance_measure, exp_name)
                            network_results_test[random_features_number].update({circle_size:evalutaion_measure})
                            # network_results_number_of_samples[random_features_number].update({circle_size:samples})
                    elif analysis_type == 'linear_regression':
                        evalutaion_measure = analysis_fn(subsample_data, labels, performance_measure, exp_name)
                        network_results_test[random_features_number] = evalutaion_measure
                    else:
                        logging.error('Analysis type not implemented')
                results_test.update({layer:network_results_test})
                # results_number_of_samples.update({layer:network_results_number_of_samples})
            dir_path = os.path.join(results_dir, exp_name, 'set_size_{}'.format(set_size))
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            with open(os.path.join(results_dir,exp_name, 'set_size_{}'.format(set_size), result_fname), 'wb') as f:
                pickle.dump(results_test,f)
            # with open(os.path.join(results_dir,exp_name, 'set_size_{}'.format(set_size), result_fname), 'wb') as f:
            #   pickle.dump(results_number_of_samples,f)
            logging.info('Total time %d'%(time.time()-start_time))
    elif "2" in exp_name:
        if exp_name == "2a":
            analysis_type = 'logistic_regression'
            analysis_fn = logistic_regression
        elif exp_name == "2b":
            analysis_type = 'logistic_regression'
            analysis_fn = logistic_regression 
        elif exp_name == "2c":
            analysis_type = 'logistic_regression'
            analysis_fn = logistic_regression 
        else:
            logging.error('Other experiments not implemented')
        results_test = {}
        number_of_random_features = subsampling_levels
        start_time = time.time()
        for layer in intermediate_layer_names:
            network_results_test = {}
            logging.info('####################       {}          ##########################'.format(layer))
            if exp_name == "2a":
                low_div_data, low_div_labels = get_data_and_labels(stim_type = 'low_diversity', layer = layer, 
                                                                intermediate_layer_feats_dir = features_dir, 
                                                                analysis_type='logistic_regression', 
                                                                condition='low_diversity')
                high_div_data, high_div_labels = get_data_and_labels(stim_type = 'high_diversity', layer = layer, 
                                                                intermediate_layer_feats_dir = features_dir, 
                                                                analysis_type='logistic_regression', 
                                                                condition='high_diversity')
            elif exp_name == "2b":
                low_div_data, low_div_labels = get_data_and_labels(stim_type = 'low_diversity', layer = layer, 
                                                                intermediate_layer_feats_dir = features_dir, 
                                                                analysis_type='logistic_regression', 
                                                                condition='ind_color')
                high_div_data, high_div_labels = get_data_and_labels(stim_type = 'high_diversity', layer = layer, 
                                                                intermediate_layer_feats_dir = features_dir, 
                                                                analysis_type='logistic_regression', 
                                                                condition='ind_color')
            elif exp_name == "2c":
                low_div_data, low_div_labels = get_data_and_labels(stim_type = 'low_diversity', layer = layer, 
                                                                intermediate_layer_feats_dir = features_dir, 
                                                                analysis_type='logistic_regression', 
                                                                condition='ind_letter')
                high_div_data, high_div_labels = get_data_and_labels(stim_type = 'high_diversity', layer = layer, 
                                                                intermediate_layer_feats_dir = features_dir, 
                                                                analysis_type='logistic_regression', 
                                                                condition='ind_letter')
            else:
                logging.error('only implemented 2abc in exp 2')                    
            data = np.concatenate((low_div_data, high_div_data))
            labels = np.concatenate((low_div_labels, high_div_labels))
            for random_features_number in number_of_random_features:
                network_results_test[random_features_number] = {}
                if data.shape[1]<random_features_number:
                    network_results_test.update({random_features_number:float('nan')})
                    continue
                subsample_data = subsample_img_features(data, random_features_number, 100)
                if exp_name == '2a':
                    evalutaion_measure = analysis_fn(subsample_data, labels, performance_measure, exp_name)
                    network_results_test.update({random_features_number:evalutaion_measure})
                elif exp_name == '2b':
                    for color in range(labels.shape[1]):
                        evalutaion_measure = analysis_fn(subsample_data, labels[:,color], performance_measure, exp_name)
                        network_results_test[random_features_number].update({color:evalutaion_measure})
                elif exp_name == '2c':
                    temp = []
                    for letter in range(labels.shape[1]):
                        evalutaion_measure = analysis_fn(subsample_data, labels[:,letter], performance_measure, exp_name)
                        temp.append(evalutaion_measure)
                        network_results_test[random_features_number].update({letter:evalutaion_measure})
                else:
                    logging.error('Only logistic regression implemented for experiment 2a and 2b')
            results_test.update({layer:network_results_test})
            dir_path = os.path.join(results_dir, exp_name)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            with open(os.path.join(results_dir,exp_name, result_fname), 'wb') as f:
                pickle.dump(results_test, f)
            logging.info('Total time %d'%(time.time()-start_time))
        



def extract_intermediate_layer_representations(model_name, intermediate_layer_names, dataset_base_dir, batch_size, stim_type, features_base_dir, conditions, exp_name):
    """
    This function extracts and stores intermediate layer representations given a model
    and a dataset   
    """
    if exp_name == '1a':
        stim_name = 'random_stim'
    elif exp_name == '1b':
        stim_name = 'test_stim' 
    else:
        logging.error('Only Exp 1a, 1b implemented')

    # condition is set size if exp is 1a or 1b, otherwise it is color diveristy
    for condition in conditions:
        if exp_name in ['1a' or '1b']:
            dataset_dir = os.path.join(dataset_base_dir, '{}_generated_stimuli'.format(condition), stim_name)
            features_dir = os.path.join(features_base_dir, 'set_size_{}'.format(condition))
        elif exp_name in ['2a']:
            stim_type = condition
            dataset_dir = os.path.join(dataset_base_dir, condition)
            features_dir = os.path.join(features_base_dir)
        else:
            logging.error('Only 1a, 1b, 2a activation extraction implemented')
        store_dataset_fnames(intermediate_layer_names, dataset_dir, batch_size, stim_type, features_dir)
        start = time.time()
        store_intermediate_layer_features(model_name, intermediate_layer_names, dataset_dir, batch_size, stim_type, features_dir)
        logging.info('Total time to extract intermediate layer reprsentations (in seconds): {}'.format(time.time()-start))
















