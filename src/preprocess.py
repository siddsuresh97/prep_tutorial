import ast 
import random
import pickle
import numpy as np
import logging
import os
import deepdish as dd
from sklearn.preprocessing import MultiLabelBinarizer

def get_data_and_labels(stim_type, layer, intermediate_layer_feats_dir, analysis_type, condition):
    """
    Returns intermediate layer features and corresponding
    labels. 
    Labels are one hot encoded for logistic regression and 
    float (average size otherwise)

    condition is set size for avg.size experiments and color diversity otherwise
    """
    part_1_fname = 'resnet_50_features_{}_{}_part_1.h5'.format(stim_type, layer)
    part_2_fname = 'resnet_50_features_{}_{}_part_2.h5'.format(stim_type, layer)
    try:
        data_1 = np.concatenate(np.asarray(dd.io.load(os.path.join(intermediate_layer_feats_dir, part_1_fname))['features'],dtype = 'float16'))
        if layer != 'avg_pool':
            data_1 = np.squeeze(np.apply_over_axes(np.mean, data_1, [1, 2]))
    except ValueError:
        # occurs because part_1 file is empty because batch_size < dataset_size
        logging.info('Value error while loading data 1')
        logging.info("%r %r %r %r %r", stim_type, layer, intermediate_layer_feats_dir, analysis_type, condition)
    logging.info('dict 1 loaded')
    data_2 = np.concatenate(np.asarray(dd.io.load(os.path.join(intermediate_layer_feats_dir, part_2_fname))['features']))
    if layer != 'avg_pool':
        data_2 = np.squeeze(np.apply_over_axes(np.mean, data_2, [1, 2]))
    if stim_type in ['low_diversity', 'high_diversity']: # because of value error, data_1 is empty
        data = data_2
    else:
        data = np.concatenate((data_1, data_2))
    data.astype('float16')
    logging.info('dict 2 loaded')
    fnames = pickle.load(open(os.path.join(intermediate_layer_feats_dir , "filenames_{}.p".format(stim_type)), "rb" ))
    labels = []
    if analysis_type == 'logistic_regression':
        if condition in [4, 8, 10, 16]:
            for fname in fnames['fnames']:
                temp = fname.split('_')[-1].split('.png')
                radii = ast.literal_eval(temp[0])
                # pre processing code for logistic regression dataset with no confounds
                # temp = fname.split('radii')[-1].split('.png')[0]
                # temp = temp[1:-1]
                # temp = temp.split(' ')
                # temp = [i[:-1] if '_' in i else i for i in temp]
                # radii = [float(i) for i in temp if i != '']
                assert(len(radii) == condition)
                labels.append(radii)
            one_hot = MultiLabelBinarizer()
            one_hot_encoded = one_hot.fit_transform(labels)
            assert(np.asarray(data).shape[0]==np.asarray(labels).shape[0])
            return np.asarray(data), np.asarray(one_hot_encoded)
        elif condition in ['low_diversity', 'high_diversity']:
            for fname in fnames['fnames']:
                if stim_type == 'low_diversity':
                    labels.append(0)
                elif stim_type == 'high_diversity':
                    labels.append(1)
                else:
                    logging.error('Only high and low diversity stimuli allowed')
            return np.asarray(data), np.asarray(labels)
        elif condition in ['ind_color']:
            for fname in fnames['fnames']:
                temp = fname.split('test_array_')[-1].split('_[')
                colors_rgb = ast.literal_eval(temp[0])
                labels.append(colors_rgb)
            assert(np.asarray(data).shape[0]==np.asarray(labels).shape[0])
            one_hot = MultiLabelBinarizer()
            one_hot_encoded_colors = one_hot.fit_transform(labels)
            return np.asarray(data), np.asarray(one_hot_encoded_colors)
        elif condition in ['ind_letter']:
            for fname in fnames['fnames']:
                letters = [i for i in fname.split(']_[')[-1].split('.png')[0] if i.isupper()]
                labels.append(letters)
            assert(np.asarray(data).shape[0]==np.asarray(labels).shape[0])
            one_hot = MultiLabelBinarizer()
            one_hot_encoded_letters = one_hot.fit_transform(labels)
            return np.asarray(data), np.asarray(one_hot_encoded_letters)
        else:
            logging.error('Only color diversity and average size experiments implemented')

    elif analysis_type == 'linear_regression':
        if condition in [4, 8, 10, 16]:
            if stim_type in ['test_stim', 'ind']:
                for fname in fnames['fnames']:
                    temp = fname.split('_')[-1].split('.png')
                    radii = ast.literal_eval(temp[0])
                    # pre processing code for logistic regression dataset with no confounds
                    # temp = fname.split('radii')[-1].split('.png')[0]
                    # temp = temp[1:-1]
                    # temp = temp.split(' ')
                    # temp = [i[:-1] if '_' in i else i for i in temp]
                    # radii = [float(i) for i in temp if i != '']
                    if stim_type == 'ind':
                        # Sanity check stimuli have single circles so there should be only 1 radius in the list
                        # assert(type(radii[0])== float)
                        assert(type(radii)== float)
                        # assert(len(radii) == 1)
                    else:
                        assert(len(radii) == condition)
                    labels.append(np.mean(radii))
                assert(np.asarray(data).shape[0]==np.asarray(labels).shape[0])
                return np.asarray(data), np.asarray(labels)
            else:
                logging.error("Only Ind and Test Stim implemented")
        else:
            print('Only average size implemented for linear regression')
    else:
        logging.error("Invalid type of analysis. Only linear regression and logistic regression implemented")

    

def subsample_img_features(data, n_features, seed):
    """
    Subsamples data(img_features) 
    TO DO - implement tsne
    """
    random.seed(seed)
    indices = random.sample(range(0, data.shape[1]), n_features)
    data_subset = []
    for instance in data:
        data_subset.append([instance[index] for index in indices])
    return np.asarray(data_subset)