import ipdb
from datetime import datetime
import gc
import os
import deepdish as dd
import numpy as np
import logging
import pickle
import time

import tensorflow as tf
from tensorflow.keras.backend import clear_session
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_input_resnet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import ResNet50


def store_dataset_fnames(intermediate_layer_names, dataset_dir, batch_size, stim_type, features_dir):
    # Stores filenames of stimuli in a pickle file
    # Fnames of stimuli contain metadata about labels
    # Only one intermediate layer is used because the fnames are same regardless of the intermediate layer
    datagen = ImageDataGenerator(preprocessing_function=preprocess_input_resnet50)
    start = time.time()
    if not os.path.exists(features_dir):
        os.makedirs(features_dir)
    for layer in intermediate_layer_names[0]:
        generator = datagen.flow_from_directory(dataset_dir, shuffle = False, batch_size = batch_size)
        filenames = generator.filenames
        fname_dict = {'fnames':filenames}
        pickle.dump( fname_dict, open(os.path.join(features_dir,"filenames_{}.p".format(stim_type)), "wb" ))  
    logging.info(datetime.now().strftime("%H:%M:%S"))
    


def store_intermediate_layer_features(model_name, intermediate_layer_names, dataset_dir, batch_size, stim_type, features_dir):
    for layer in intermediate_layer_names:
        logging.info('------------------------------- {} ----------------------------'.format(layer))
        if model_name == 'resnet50':
            datagen = ImageDataGenerator(preprocessing_function=preprocess_input_resnet50)
        else:
            logging.error('Models apart from resnet50 not implemented')
        generator = datagen.flow_from_directory(dataset_dir, shuffle = False, batch_size = batch_size, target_size = (224, 224))
        len = generator.n
        batches = np.ceil(len/batch_size)
        extract_and_store(model_name, 1, layer, generator, features_dir, stim_type, batches)
        extract_and_store(model_name, 2, layer, generator, features_dir, stim_type, batches)
    return


def extract_and_store(model_name, part, layer, generator, features_dir, stim_type, batches):
    '''
    Extracts intermediate layer features and stores them in two h5 files
    '''
    if model_name == 'resnet50':
        model = ResNet50(weights='imagenet', include_top=True)
    else:
        logging.error('Models apart from resnet not implementde')
    extractor = tf.keras.Model(inputs=model.inputs,
                                outputs=[model.get_layer(layer).output])
    features_dict = {'fnames':[],'features':[]}
    if part == 1:
        min_range = 0
        max_range = int(batches)//2
    elif part == 2:
        min_range = int(batches)//2
        max_range = int(batches)
    for batch in range(min_range, max_range):
        time_for_generator_operation = time.time()
        x,y = generator.next()
        # logging.info('Time for generator %f' % time.time()-time_for_generator_operation)
        time_for_prediction = time.time()
        generator_features = extractor.predict(x)
        features_dict['features'].append(generator_features)
        # logging.info('Prediction Time = %f'%time.time()-time_for_prediction)
        time_for_deletion = time.time()
        del generator_features
        # logging.info('Time_for_deletion = ', time.time()-time_for_deletion)
        idx = (generator.batch_index - 1) * generator.batch_size
        import ipdb;ipdb.set_trace()
        features_dict['fnames'].append(generator.filenames[idx : idx + generator.batch_size])
    del extractor
    del model
    if not os.path.exists(features_dir):
        os.makedirs(features_dir)
    dd.io.save(os.path.join(features_dir, 'resnet_50_features_{}_{}_part_{}.h5'.format(stim_type, layer, part)), features_dict)
    del features_dict
    gc.collect()
    clear_session()
    logging.info("Saved {} part {}".format(layer, part))
    return 