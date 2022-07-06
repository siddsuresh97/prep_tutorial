
import os
from turtle import title
import ipdb
import pickle
import numpy as np
import pandas as pd
from statsmodels.graphics.factorplots import interaction_plot
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.facecolor'] = 'white'

# regression_dirs = [regression_results_set_size_4, regression_results_set_size_8, regression_results_set_size_10, regression_results_set_size_16]
def plot_graphs(regression_dirs, fname, plot_title, x_label, y_label, 
                        subsampling_levels, mode, figsize, ylim, legend_title, analysis_type, exp_name):
    """
    Takes in the directories of the saved pickled files
    of each set size and plots accuracy~layer_name*sampling_size
    for multiple set sizes

    fname - name of the pickle file which you want to plot
    """
    fig, ax = plt.subplots(1, len(regression_dirs), figsize=figsize)
    for subplot, regression_dir in enumerate(regression_dirs):
        infile = open(os.path.join(regression_dir, fname), 'rb')
        var_fit_results = pickle.load(infile)
        infile.close()
        # column_names = [i.split('.')[0].split('_features')[0] for i in var_fit_results.keys()]
        data = {}
        if "1" in exp_name:
            for key in var_fit_results.keys():
                row_names = var_fit_results[key].keys()
                if analysis_type == 'logistic_regression':
                    accuracy = []
                    for row in row_names:
                        ind_accuracy = []
                        if type(var_fit_results[key][row]) is dict:
                            for circle_size in var_fit_results[key][row].keys():
                                ind_accuracy += [var_fit_results[key][row][circle_size]]
                                #accuracy += [sum(ind_accuracy)/len(ind_accuracy)]
                            accuracy += [np.mean(ind_accuracy)]
                        else:
                            accuracy += [float('nan')]
                    data.update({key.split('.')[0].split('_features')[0]:accuracy})
                else:
                    performance_measure = [(var_fit_results[key][i]) for i in row_names]
                    data.update({key.split('.')[0].split('_features')[0]:performance_measure})
        elif "2" in exp_name:
            for key in var_fit_results.keys():
                row_names = var_fit_results[key].keys()
                if exp_name == '2a':
                    performance_measure = [(var_fit_results[key][i]) for i in row_names]
                    data.update({key.split('.')[0].split('_features')[0]:performance_measure}) 
                elif exp_name in ['2b', '2c']:
                    accuracy = []
                    for row in row_names:
                        ind_accuracy = []
                        if type(var_fit_results[key][row]) is dict:
                            for color in var_fit_results[key][row].keys():
                                ind_accuracy += [var_fit_results[key][row][color]]
                                #accuracy += [sum(ind_accuracy)/len(ind_accuracy)]
                            accuracy += [np.mean(ind_accuracy)]
                        else:
                            accuracy += [float('nan')]
                    data.update({key.split('.')[0].split('_features')[0]:accuracy})
                else:
                    print('Only implemented plotting for 2a, 2b, 2c in exp 2')   
        else:
            print('Only exp 1 and exp 2 implemented')
        num_of_subsampling_levels = list(set([len(data[key]) for key in data.keys()]))[0]
        assert(num_of_subsampling_levels in [10,20,50,100,200,500])
        df = pd.DataFrame(data)
        layer_names = df.columns
        df = df.T
        df.columns = row_names
        average_fit = []
        for i in reversed(df.values):
            average_fit += list(i)
        layer = []
        for layer_name in reversed(layer_names):
            if layer_name == 'avg_pool':
                layer += ['pre_final']*num_of_subsampling_levels
            elif layer_name == 'pool1_pool':
                layer += ['conv1(pool1_pool)']*num_of_subsampling_levels
            else:
                layer += [layer_name]*num_of_subsampling_levels
        if num_of_subsampling_levels == 10:
            units = [10,20,50,100,200,500, 800, 1000, 1500, 2000]*layer_names.shape[0]
        elif num_of_subsampling_levels == 6:
            units = [10,20,50,100,200,500]*layer_names.shape[0]
        else:
            print('Invalid number of subsampling levels')
        
        reg_df = pd.DataFrame({"performace_measure":average_fit, "layer":layer, "units":units})
    
        reg_df = reg_df[reg_df.units.isin(subsampling_levels)]
        
        if len(regression_dirs) == 1:
            subplot = ax
        else:
            subplot = ax[subplot]
        if len(subsampling_levels) == 1:
            subplot.plot(reg_df['layer'],reg_df['performace_measure'])
        else: 
            # TO DO :- Add functionality to support plotting of any gives subsampling levels
            # RIght now, only one or all is supported
            fig = interaction_plot(np.asarray(layer), units, average_fit, ax = subplot, 
                                xlabel = "     ", ylabel = "      ")
            # fig = interaction_plot(list(reg_df['layer']), list(reg_df['units']), 
            #                     np.array(reg_df['accuracy']), ax = subplot)  
        if mode == 'avg_size':
            subplot.set_title('Set Size {}'.format(regression_dir.split('_')[-1][:-1]), fontsize = 25)
        elif mode == 'color_diversity':
            pass
        else:
            print('Unspecified mode')
        subplot.set_ylabel('', fontsize = 25)
        subplot.set_ylim(ylim)
        subplot.tick_params(axis='y', which='major', labelsize=25)
        subplot.tick_params(axis='x', which='major', labelsize=15)

        subplot.legend().set_visible(False)
    if mode == 'avg_size':
        if len(subsampling_levels) != 1:
            lines_labels = ax[0].get_legend_handles_labels()
            lines, labels = lines_labels[0], lines_labels[1]
            fig.legend(lines, labels, title = legend_title)
    else:
        if len(subsampling_levels) != 1:
            lines_labels = subplot.get_legend_handles_labels()
            lines, labels = lines_labels[0], lines_labels[1]
            fig.legend(lines, labels, title = legend_title)
          
    # fig.legend(handles, labels, loc='upper center')
    fig.supylabel(y_label, fontsize = 25)
    fig.supxlabel(x_label, fontsize = 25)
    fig.suptitle(plot_title, fontsize=30)
    plt.close()
    return fig

