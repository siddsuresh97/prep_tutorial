import logging
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix

def logistic_regression(data, labels, performance_measure, exp_name):
    if exp_name == '1c':
        label_0_idx = np.where(labels==0)[0]
        label_1_idx = np.where(labels==1)[0]
        num_stimuli = 430
        label_0_data = data[label_0_idx[:num_stimuli]]
        label_1_data = data[label_1_idx[:num_stimuli]]
        '''
        if label_1_idx.shape[0]<label_0_idx.shape[0]:
        num_stimuli = label_1_idx.shape[0]
        label_0_data = data[label_0_idx[:num_stimuli]]
        label_1_data = data[label_1_idx]
        else:
        num_stimuli = label_0_idx.shape[0]
        label_1_data = data[label_1_idx[:num_stimuli]]
        '''
        data = np.concatenate((label_0_data, label_1_data))
        labels = np.concatenate((np.zeros(num_stimuli), np.ones(num_stimuli))).reshape(-1)
        logging.info('data shape : %r, labels shape : %r, label_0 : %r, label:1,  %r'%(data.shape, labels.shape, label_0_idx.shape, label_1_idx.shape))
    
    elif exp_name == '2a':
        # this is because we want to use all the data and labels for exp 2a
        pass
    elif exp_name in ['2b', '2c']:
        label_0_idx = np.where(labels==0)[0]
        label_1_idx = np.where(labels==1)[0]
        if exp_name == '2b':
            num_stimuli = 329
        elif exp_name == '2c':
           num_stimuli = 773
        label_0_data = data[label_0_idx[:num_stimuli]]
        label_1_data = data[label_1_idx[:num_stimuli]]
        # if label_1_idx.shape[0]<label_0_idx.shape[0]:
        #     num_stimuli = label_1_idx.shape[0]
        #     label_0_data = data[label_0_idx[:num_stimuli]]
        #     label_1_data = data[label_1_idx]
        # else:
        #     num_stimuli = label_0_idx.shape[0]
        #     label_1_data = data[label_1_idx[:num_stimuli]]
        data = np.concatenate((label_0_data, label_1_data))
        labels = np.concatenate((np.zeros(num_stimuli), np.ones(num_stimuli))).reshape(-1)
        logging.info('data shape : %r, labels shape : %r, label_0 : %r, label:1,  %r'%(data.shape, labels.shape, label_0_idx.shape, label_1_idx.shape))
    # x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.1, random_state=42)
    # baseline_acc_test = max(np.count_nonzero(y_test==0)/y_test.shape[0], np.count_nonzero(y_test==1)/y_test.shape[0])
    cv = KFold(n_splits=10, random_state=1, shuffle=True)
    model = LogisticRegression(random_state=1, max_iter=1000, solver='liblinear')
    scores = cross_val_score(model, data, labels, scoring='accuracy', cv=cv, n_jobs=-1)
    DEBUG = True
    if DEBUG:
        y_pred = cross_val_predict(model, data, labels, cv=10)
        conf_mat = confusion_matrix(labels, y_pred)
        logging.info('Confusion matrix : %r'%conf_mat)
    # model.fit(x_train, y_train)
    logging.info('test data label 0 :%r,  label 1:%r'%(np.where(data==0)[0].shape, np.where(labels==1)[0].shape))
    if performance_measure == 'accuracy':
        # return model.score(x_test, y_test)
        return np.mean(scores)
    else:
        logging.error("Performance measure not implemented")
    # return num_stimuli

def linear_regression(data, labels, performance_measure):
    # x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.25, random_state=42)
    kf = KFold(n_splits=10, random_state=1, shuffle=True)
    percentage_abs_error_list = []
    rmspe_list = []
    rmse_list = []
    for train_index, test_index in kf.split(data):
        x_train, x_test = data[train_index], data[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        model = LinearRegression() 
        model.fit(x_train,y_train)
        y_pred = model.predict(x_test)
        percentage_abs_error = np.average(np.abs(y_pred-y_test)/y_test) * 100
        percentage_abs_error_list.append(percentage_abs_error) 
        rmspe = (np.sqrt(np.mean(np.square((y_test - y_pred) / y_test)))) * 100
        rmspe_list.append(rmspe)
        rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
        rmse_list.append(rmse)
    # rmse_train = np.sqrt(metrics.mean_squared_error(y_train, model.predict(x_train)))
    if performance_measure == 'percentage_abs_error':
        return np.mean(percentage_abs_error_list)
    elif performance_measure == 'rmspe':
        return np.mean(rmspe_list)
    elif performance_measure == 'rmse':
        return np.mean(rmse_list)
    else:
        logging.error("Performance measure not implemented")