import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.metrics import precision_recall_curve, roc_curve
from scipy.stats import gaussian_kde


def train_plot(df,
               logx = False,
               error_bars= False,
               n_folds = None,
               f_size = (6,6)):
    '''
    plotting function to give a  quick visualistion of results over hyperparameter
    grid when performing cross validation.

    For one hyperparameter, produces a simple x-y plot of hp1 vs result
    For two hyperparameters, produces an x-y plot of hp2 vs result for each value of hp1
    For three hyperparameters, produces a grid of x-y plots. Each facet is for a distinct hp1 value,
    displaying hp3 vs result for each value of hp2

    For >3 hyperparameters, the user should create their own plots.

    inputs:
    metric_results- pandas dataframe with up to three columns for hyperparamters, and the final
    two columns must be the mean metric  and the standard deviation for that
    choice of hyperparameters


    logx: should the x axis be log transfomred? binary
    error_bars- boolean. should error bars (st err) be plotted?
    n_folds: needed if error_bars is true. st err = std / sqrt(n_folds)
    f_size: tuple of (x size, y size) to be passed to subplot

    '''
    n_hyperparams = df.shape[1] - 2
    if error_bars:
        if not n_folds:
            raise ValueError('Need n_folds to calculate standard error')

    if n_hyperparams == 1:

        hp1 = df.columns.values[0]
        metric = df.columns.values[1]
        std = df.columns.values[2]
        if error_bars:
            serr = [x/n_folds for x in df[std]]

        f, axarr = plt.subplots(1, 1, figsize = f_size, dpi=80)

        axarr.spines['right'].set_visible(False)
        axarr.spines['top'].set_visible(False)
        axarr.tick_params(axis=u'both', which=u'both',length=5)

        if error_bars:
            axarr.errorbar(df[hp1],df[metric],yerr=serr, fmt='-x', capsize = 4)
        else:
            axarr.plot(df[hp1], df[metric], '-x')

        axarr.set_ylabel(metric)
        axarr.set_xlabel(hp1)
        if logx:
            axarr.set_xscale('log')

    elif n_hyperparams == 2:

        hp1 = df.columns.values[0]
        hp2 = df.columns.values[1]
        metric = df.columns.values[2]
        std = df.columns.values[3]

        f, axarr = plt.subplots(1, 1, figsize = f_size, dpi=80)

        axarr.spines['right'].set_visible(False)
        axarr.spines['top'].set_visible(False)
        axarr.tick_params(axis=u'both', which=u'both',length=5)

        for _1 in df[hp1].drop_duplicates():

            hp2_tmp = df.loc[(df[hp1] == _1), [hp2]].values
            hp2_tmp = [x[0] for x in hp2_tmp]
            metric_tmp = df.loc[(df[hp1] == _1), [metric]].values
            metric_tmp = [x[0] for x in metric_tmp]
            if error_bars:
                std_tmp = df.loc[(df[hp1] == _1), [std]].values
                serr_tmp = [(x[0]/n_folds) for x in std_tmp]

            if error_bars:
                axarr.errorbar(hp2_tmp, metric_tmp,yerr=serr_tmp, label = _1, fmt='-x', capsize = 2)
            else:
                axarr.plot(hp2_tmp, metric_tmp, '-x', label = _1)


        axarr.set_ylabel(metric)
        axarr.set_xlabel(hp2)
        axarr.legend(loc = 'best', title = hp1 )
        if logx:
            axarr.set_xscale('log')

    elif n_hyperparams == 3:
        # urg hacky
        hp1 = df.columns.values[0]
        hp2 = df.columns.values[1]
        hp3 = df.columns.values[2]
        metric = df.columns.values[3]
        std = df.columns.values[4]
        
        # initialise grid
        n_hp1 = df[hp1].drop_duplicates().shape[0]
        n_row = math.ceil(n_hp1/ 3)
        n_col = 3
        
        f, axarr = plt.subplots(n_row, n_hp1, figsize = f_size, dpi=80) # WAS A BUG HERE

        ymin = min(df[metric].values) * 0.99
        ymax = max(df[metric].values) * 1.01

        i = 0
        j = 0
        plt_count = 0

        for _1 in df[hp1].drop_duplicates():
            title_flag = False
            for _2 in df[hp2].drop_duplicates():
                hp3_tmp = df.loc[(df[hp1] == _1), :].loc[(df[hp2] == _2), [hp3]].values
                hp3_tmp = [x[0] for x in hp3_tmp]
                metric_tmp = df.loc[(df[hp1] == _1), :].loc[(df[hp2] == _2), [metric]].values
                metric_tmp = [x[0] for x in metric_tmp]
                if error_bars:
                    std_tmp = df.loc[(df[hp1] == _1), :].loc[(df[hp2] == _2), [std]].values
                    serr_tmp = [(x[0] / n_folds) for x in std_tmp]

                if n_row > 1:
                    axarr[i][j].set_ylim([ymin, ymax])
                    axarr[i][j].spines['right'].set_visible(False)
                    axarr[i][j].spines['top'].set_visible(False)
                    axarr[i][j].tick_params(axis=u'both', which=u'both',length=5)

                    if error_bars:
                        axarr[i][j].errorbar(hp3_tmp, metric_tmp,yerr=serr_tmp, label = _2, fmt='-x', capsize = 4)
                    else:
                        axarr[i][j].plot(hp3_tmp, metric_tmp, '-x', label = _2)

                    axarr[i][j].set_title('{} = {}'.format(hp1, str(_1)))
                    axarr[i][j].set_ylabel(metric)
                    axarr[i][j].set_xlabel(hp3)

                    if logx:
                        axarr[i][j].set_xscale('log')

                    if (i == 0) & (j == 2):
                        axarr[i][j].legend(loc = 'best', title = hp2)

                else:
                    axarr[j].set_ylim([ymin, ymax])
                    axarr[j].spines['right'].set_visible(False)
                    axarr[j].spines['top'].set_visible(False)
                    axarr[j].tick_params(axis=u'both', which=u'both',length=5)

                    # standard error or standard deviation???
                    if error_bars:
                        axarr[j].errorbar(hp3_tmp, metric_tmp,yerr=serr_tmp, label = _2, fmt ='-x', capsize = 4)
                    else:
                        axarr[j].plot(hp3_tmp, metric_tmp, '-x', label = _2)


                    if not title_flag:
                        axarr[j].set_title('{} = {}'.format(hp1, str(_1)))
                        title_flag= True

                    axarr[j].set_ylabel(metric)
                    axarr[j].set_xlabel(hp3)

                    if logx:
                        axarr[j].set_xscale('log') # was a bug here

                    if j == n_hp1 -1:
                        axarr[j].legend(loc = 'best', title = hp2) # was a bug here

                    if j > 0:
                        axarr[j].set_ylabel('')
                        axarr[j].set_yticklabels([])
                        axarr[j].set_yticks([])

            plt_count += 1
            j += 1
            if j == 3:
                i += 1
                j = 0

        # kill unused axis
        if n_row > 1:
            if n_row * 3 > len(df[hp2].drop_duplicates()):
                for k in range(j, 3):
                    axarr[n_row-1][k].axis('off')

        f.subplots_adjust(hspace = 0.3)

    elif n_hyperparams > 3:
        raise ValueError('Too many hyperparameters for a simple plot. Go manual!')
    return f


class LiftChart(object):
    '''
    Class to calculate neccesary information for a lift chart,
    and a helper method to plot

    The attribute uplift contains a dict with the neccesary information
    to plot
    '''
    def __init__(self, outcome):
        '''
        inputs:
        outcome: np array of true outcome. must be binary
        '''
        self.outcome_arr = outcome
        self.n_samples = outcome.shape[0]
        self.n_pos = np.where(outcome == 1)[0].shape[0]
        self.n_neg = self.n_samples - self.n_pos
        self.pct_pos = self.n_pos / self.n_samples * 100
        self._uplift = dict()

    def calc_uplift(self, pred_probs, name):
        '''
        inputs:
        pred_probs: np array of predicted probabilities of outcome
        name: string, corresponding to the model name that made the predictions
        '''
        tmp = np.column_stack((self.outcome_arr, pred_probs))
        tmp = np.flipud(tmp[np.argsort(tmp[:, 1])])

        ix = np.arange(1, self.n_samples + 1)
        pct_samp = ix / self.n_samples * 100
        count_found = np.cumsum(tmp[:, 0])
        pct_found = count_found / self.n_pos * 100

        uplift_res = np.column_stack((ix, pct_samp, count_found, pct_found))
        self.uplift = (name, uplift_res)

    @property
    def uplift(self):
        return self._uplift

    @uplift.setter
    def uplift(self, val_tuple):
        self._uplift[val_tuple[0]] = val_tuple[1]

    def plot(self, thresh_pct = None,  figsize = (10,6)):
        '''
        method to plot lift chart using uplift attribute
        inputs:
        thresh_pct: float between 0 and 100. If not None, will highlight where
        this threshold is obtained on the chart
        figsize: tuple of (x size, y size) to be passed to subplot
        '''
        fig, ax = plt.subplots(figsize=figsize, facecolor='white', edgecolor='white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        for k, v in self.uplift.items():
            plt.plot(v[:, 1], v[:, 3], label = k)

        if thresh_pct:
            model_thresh = dict()
            val_max = 0
            for k, v in sorted(self.uplift.items()):
                val_ix = np.argmax(v[:, 3] > thresh_pct)
                val = v[val_ix, 1]
                model_thresh[k] = val
                plt.plot([val, val],[0, thresh_pct], 'k--', alpha = 0.4)
                if val > val_max: val_max = val

            plt.plot([0, val_max],[thresh_pct, thresh_pct], 'k--', alpha = 0.4)

        plt.legend(loc=4, fontsize=12)
        plt.xlabel('Percentage tested', fontsize=12)
        plt.ylabel('Percentage found', fontsize=12)
        plt.ylim([0, 105])
        plt.xlim([0, 105])
        plt.fill([0,100, self.pct_pos], [0, 100, 100], 'grey', alpha = 0.1)

        # TODO: perhaps obtain thresh from underlying np.array
        # this will allow user to extract wherever they want without making
        # multiple plots
        if thresh_pct:
            return (ax, model_thresh)
        else:
            return ax


# TODO: perhaps extract closest topleft cutoff and AUC? store as attribute?
class ROCPlot(object):
    '''
    Class to calculate neccesary information for a ROC plot,
    and a helper method to plot

    The attribute roc contains a dict with the neccesary information
    to plot
    '''
    def __init__(self, outcome):
        '''
        inputs:
        outcome: np array of true outcome. must be binary
        '''
        self.outcome_arr = outcome
        self._roc = dict()

    def calc_roc(self, pred_probs, name):
        '''
        inputs:
        pred_probs: np array of predicted probabilities of outcome
        name: string, corresponding to the model name that made the predictions
        '''
        fpr, tpr, _ = roc_curve(self.outcome_arr, pred_probs)
        self.roc = (name, np.column_stack((fpr, tpr)))

    @property
    def roc(self):
        return self._roc

    @roc.setter
    def roc(self, val_tuple):
        self._roc[val_tuple[0]] = val_tuple[1]

    def plot(self,  figsize = (10,6)):
        '''
        method to plot lift chart using uplift attribute
        inputs:
        thresh_pct: float between 0 and 100. If not None, will highlight where
        this threshold is obtained on the chart
        figsize: tuple of (x size, y size) to be passed to subplot
        '''
        fig, ax = plt.subplots(figsize=figsize, facecolor='white', edgecolor='white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        for k, v in self.roc.items():
            plt.plot(v[:, 0], v[:, 1], label = k)

        plt.legend(loc=4, fontsize=12)
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.ylim([0, 1])
        plt.xlim([0, 1])
        plt.fill([0,0, 1], [0, 1, 1], 'grey', alpha = 0.1)

        return ax


# def hist_plot(df, n_col = 3, outcome_col = None,
#               n_bins = 20, plot_legend = False,
#               norm = True, f_size = (15,15)):
#     '''
#     Function to generate histogram plots, optionally grouping by outcome_col

#     Inputs:
#     df: pandas dataframe of features to plot
#     n_col: number of columns
#     outcome_col: string specifying the outcome column. If None, will not group
#     n_bins: number of bins
#     plot_legend: Boolean. should a legend be included in the plot?
#     f_size: tuple of (x size, y size) to specify size of plot
#     '''

#     if outcome_col is None:
#         n_features = df.shape[1]
#         feature_names = df.columns.values
#         df_features = df
#     else:
#         n_features = df.shape[1] -1
#         feature_names = [x for x in df.columns.values if x != outcome_col]
#         df_features = df.loc[:, feature_names]
#         outcome_values = sorted([x for x in df.loc[:, outcome_col].unique()])

#     if n_features % n_col == 0:
#         n_row = n_features // n_col
#     else:
#         n_row = n_features // n_col + 1

#     f, axarr = plt.subplots(n_row, n_col, figsize = f_size, dpi=80)

#     v = 0
#     for i in np.arange(n_row):
#         for j in np.arange(n_col):

#         # turn off axis if empty grids
#             if v >= n_features:
#                 if n_row == 1:
#                     axarr[j].axis('off')
#                     continue
#                 elif n_col == 1:
#                     axarr[i].axis('off')
#                     continue
#                 else:
#                     axarr[i][j].axis('off')
#                     continue

#             xmin = df_features.loc[:, feature_names[v]].min()
#             xmax = df_features.loc[:, feature_names[v]].max()

#             if (n_row == 1) and (n_col == 1):
#                 axarr.spines['right'].set_visible(False)
#                 axarr.spines['top'].set_visible(False)
#                 axarr.tick_params(axis=u'both', which=u'both',length=5)
#             elif n_row == 1:
#                 axarr[j].spines['right'].set_visible(False)
#                 axarr[j].spines['top'].set_visible(False)
#                 axarr[j].tick_params(axis=u'both', which=u'both',length=5)
#             elif n_col == 1:
#                 axarr[i].spines['right'].set_visible(False)
#                 axarr[i].spines['top'].set_visible(False)
#                 axarr[i].tick_params(axis=u'both', which=u'both',length=5)
#             else:
#                 axarr[i][j].spines['right'].set_visible(False)
#                 axarr[i][j].spines['top'].set_visible(False)
#                 axarr[i][j].tick_params(axis=u'both', which=u'both',length=5)

#             if outcome_col is not None:
#                 for c in outcome_values:
#                     lab = None if plot_legend is False else c
#                     plt_data = df_features.loc[df[outcome_col] == c,
#                                       feature_names[v]]


#                     if (n_row == 1) and (n_col ==1):
#                         axarr.hist(plt_data,
#                                    bins=n_bins,
#                                    label = lab,
#                                    normed = norm,
#                                    alpha = 0.6)
#                         axarr.set_title(feature_names[v])
#                         if plot_legend:
#                             if (j == 0) & (i == 0):
#                                 axarr.legend()
#                     elif n_row == 1:
#                         axarr[j].hist(plt_data,
#                                       bins=n_bins,
#                                       label = lab,
#                                       normed = norm,
#                                       alpha = 0.6)
#                         axarr[j].set_title(feature_names[v])
#                         if plot_legend:
#                             if (j == 0) & (i == 0):
#                                 axarr[j].legend()
#                     elif n_col == 1:
#                         axarr[i].hist(plt_data, range = bin_range, label = lab,
#                                       normed = norm, alpha = 0.6)
#                         axarr[i].set_title(feature_names[v])
#                         if plot_legend:
#                             if (j == 0) & (i == 0):
#                                 axarr[j].legend()
#                     else:
#                         axarr[i][j].hist(plt_data,
#                                          bins=n_bins,
#                                          label = lab,
#                                          normed = norm,
#                                          alpha = 0.6)
#                         axarr[i][j].set_title(feature_names[v])
#                         if plot_legend:
#                             if (j == 0) & (i == 0):
#                                 axarr[i,j].legend()
#                 v += 1
#             else:
#                 plt_data = df_features.loc[:, feature_names[v]].values
#                 if (n_row == 1) and (n_col == 1):
#                     axarr.hist(plt_data,
#                                 bins=n_bins,
#                                 label = lab,
#                                 normed = norm,
#                                 alpha = 0.6)
#                     axarr.set_title(feature_names[v])
#                     if plot_legend:
#                         if (j == 0) & (i == 0):
#                             axarr.legend()
#                 elif n_row == 1:
#                     axarr[j].hist(plt_data,
#                                   bins=n_bins,
#                                   label = lab,
#                                   normed = norm,
#                                   alpha = 0.6)
#                     axarr[j].set_title(feature_names[v])
#                     if plot_legend:
#                         if (j == 0) & (i == 0):
#                             axarr[j].legend()
#                 elif n_col == 1:
#                     axarr[i].hist(plt_data,
#                                   bins=n_bins,
#                                   label = lab,
#                                   normed = norm,
#                                   alpha = 0.6)
#                     axarr[i].set_title(feature_names[v])
#                     if plot_legend:
#                         if (j == 0) & (i == 0):
#                             axarr[j].legend()
#                 else:
#                     axarr[i][j].hist(plt_data,
#                                      bins=n_bins,#np.arange(bin_range[0], bin_range[1] + bin_width, bin_width),
#                                      label = lab,
#                                      normed = norm,
#                                      alpha = 0.6)
#                     axarr[i][j].set_title(feature_names[v])
#                     if plot_legend:
#                         if (j == 0) & (i == 0):
#                             axarr[i][j].legend()
#                 v += 1


#     if n_col == 1:
#         f.subplots_adjust(hspace = 1)
#     else:
#         f.subplots_adjust(hspace = 0.5)
#     #return f

def hist_plot(df, n_col = 3, outcome_col = None,
              n_bins = 20, plot_legend = False,
              norm = True, f_size = (15,15)):
    '''
    Function to generate histogram plots, optionally grouping by outcome_col
    Inputs:
    df: pandas dataframe of features to plot
    n_col: number of columns
    outcome_col: string specifying the outcome column. If None, will not group
    n_bins: number of bins
    plot_legend: Boolean. should a legend be included in the plot?
    f_size: tuple of (x size, y size) to specify size of plot
    
    TODO: consistent bin sizes. define bin_range lower, upper and a bin width
          CURRENTLY BINS AREW FIXED BY FIRST OUTCOME HAVING 20 BINS
    TODO: if yaxis very small, switch to scientific notation OR increase y axis spacing
    '''

    if outcome_col is None:
        n_features = df.shape[1]
        feature_names = df.columns.values
        df_features = df
    else:
        n_features = df.shape[1] -1
        feature_names = [x for x in df.columns.values if x != outcome_col]
        df_features = df.loc[:, feature_names]
        outcome_values = sorted([x for x in df.loc[:, outcome_col].unique()])

    if n_features % n_col == 0:
        n_row = n_features // n_col
    else:
        n_row = n_features // n_col + 1

    f, axarr = plt.subplots(n_row, n_col, figsize = f_size, dpi=80)

    v = 0
    for i in np.arange(n_row):
        for j in np.arange(n_col):
            
            # bins will be calculated by first outcome
            # if multiple outcomes (so they allign)
            curr_bins = None

        # turn off axis if empty grids
            if v >= n_features:
                if n_row == 1:
                    axarr[j].axis('off')
                    continue
                elif n_col == 1:
                    axarr[i].axis('off')
                    continue
                else:
                    axarr[i][j].axis('off')
                    continue

            xmin = df_features.loc[:, feature_names[v]].min()
            xmax = df_features.loc[:, feature_names[v]].max()

            if (n_row == 1) and (n_col == 1):
                axarr.spines['right'].set_visible(False)
                axarr.spines['top'].set_visible(False)
                axarr.tick_params(axis=u'both', which=u'both',length=5)
            elif n_row == 1:
                axarr[j].spines['right'].set_visible(False)
                axarr[j].spines['top'].set_visible(False)
                axarr[j].tick_params(axis=u'both', which=u'both',length=5)
            elif n_col == 1:
                axarr[i].spines['right'].set_visible(False)
                axarr[i].spines['top'].set_visible(False)
                axarr[i].tick_params(axis=u'both', which=u'both',length=5)
            else:
                axarr[i][j].spines['right'].set_visible(False)
                axarr[i][j].spines['top'].set_visible(False)
                axarr[i][j].tick_params(axis=u'both', which=u'both',length=5)
                
            if outcome_col is not None:
                for c in outcome_values:
                    
                    if curr_bins is not None:
                        hist_bins = curr_bins
                    else:
                        hist_bins = n_bins
                                                
                    lab = None if plot_legend is False else c
                    plt_data = df_features.loc[df[outcome_col] == c,
                                      feature_names[v]]


                    if (n_row == 1) and (n_col ==1):
                        (n, curr_bins, patches) = axarr.hist(plt_data,
                                                             bins=hist_bins,
                                                             label = str(lab),
                                                             normed = norm,
                                                             alpha = 0.6)
                        axarr.set_title(feature_names[v])
                        if plot_legend:
                            if (j == 0) & (i == 0):
                                axarr.legend()
                    elif n_row == 1:
                        (n, curr_bins, patches) = axarr[j].hist(plt_data,
                                                                bins=hist_bins,
                                                                label = str(lab),
                                                                normed = norm,
                                                                alpha = 0.6)
                        axarr[j].set_title(feature_names[v])
                        if plot_legend:
                            if (j == 0) & (i == 0):
                                axarr[j].legend()
                    elif n_col == 1:
                        (n, curr_bins, patches) = axarr[i].hist(plt_data, 
                                                                bins = hist_bins, 
                                                                label = lab,
                                                                normed = norm, 
                                                                alpha = 0.6)
                        axarr[i].set_title(feature_names[v])
                        if plot_legend:
                            if (j == 0) & (i == 0):
                                axarr[j].legend()
                    else:
                        (n, curr_bins, patches) = axarr[i][j].hist(plt_data,
                                                                   bins=hist_bins,
                                                                   label = str(lab),
                                                                   normed = norm,
                                                                   alpha = 0.6)
                        axarr[i][j].set_title(feature_names[v])
                        if plot_legend:
                            if (j == 0) & (i == 0):
                                axarr[i,j].legend()
                v += 1
            else:
                plt_data = df_features.loc[:, feature_names[v]].values
                if (n_row == 1) and (n_col == 1):
                    (n, curr_bins, patches) = axarr.hist(plt_data,
                                                         bins=hist_bins,
                                                         label = str(lab),
                                                         normed = norm,
                                                         alpha = 0.6)
                    axarr.set_title(feature_names[v])
                    if plot_legend:
                        if (j == 0) & (i == 0):
                            axarr.legend()
                elif n_row == 1:
                    (n, curr_bins, patches) = axarr[j].hist(plt_data,
                                                            bins=hist_bins,
                                                            label = str(lab),
                                                            normed = norm,
                                                            alpha = 0.6)
                    axarr[j].set_title(feature_names[v])
                    if plot_legend:
                        if (j == 0) & (i == 0):
                            axarr[j].legend()
                elif n_col == 1:
                    (n, curr_bins, patches) = axarr[i].hist(plt_data,
                                                            bins=hist_bins,
                                                            label = str(lab),
                                                            normed = norm,
                                                            alpha = 0.6)
                    axarr[i].set_title(feature_names[v])
                    if plot_legend:
                        if (j == 0) & (i == 0):
                            axarr[j].legend()
                else:
                    (n, curr_bins, patches) = axarr[i][j].hist(plt_data,
                                                               bins=hist_bins,
                                                               label = str(lab),
                                                               normed = norm,
                                                               alpha = 0.6)
                    axarr[i][j].set_title(feature_names[v])
                    if plot_legend:
                        if (j == 0) & (i == 0):
                            axarr[i][j].legend()
                v += 1


    if n_col == 1:
        f.subplots_adjust(hspace = 1)
    else:
        f.subplots_adjust(hspace = 0.5)
    return f
    
    
    
def kde_plot(df, n_col = 3, outcome_col = None,
             plot_legend = False, cov_factor = 0.25,
             f_size = (15,15)):
    '''
    Function to generate KDE plots, optionally grouping by outcome_col

    Inputs:
    df: pandas dataframe of features to plot
    n_col: number of columns
    outcome_col: string specifying the outcome column. Optional
    plot_legend: Boolean. should a legend be included in the plot?
    cov_factor: parameter to pass to scipy.stats.guassian_kde
    f_size: tuple of (x size, y size) to specify size of plot
    '''

    if outcome_col is None:
        n_features = df.shape[1]
        feature_names = df.columns.values
        df_features = df
    else:
        n_features = df.shape[1] -1
        feature_names = [x for x in df.columns.values if x != outcome_col]
        df_features = df.loc[:, feature_names]
        outcome_values = sorted([x for x in df.loc[:, outcome_col].unique()])

    # determine number of rows from specified number of columns
    if n_features % n_col == 0:
        n_row = n_features // n_col
    else:
        n_row = n_features // n_col + 1

    # initialise plot
    f, axarr = plt.subplots(n_row, n_col, figsize = f_size, dpi=80)

    # iterate over subplots
    v = 0
    for i in np.arange(n_row):
        for j in np.arange(n_col):

            # turn off axis if empty grids
            if v >= n_features:
                if n_row == 1:
                    axarr[j].axis('off')
                    continue
                elif n_col == 1:
                    axarr[i].axis('off')
                    continue
                else:
                    axarr[i][j].axis('off')
                    continue

            if (n_row==1) and (n_col==1):
                axarr.spines['right'].set_visible(False)
                axarr.spines['top'].set_visible(False)
                axarr.tick_params(axis=u'both', which=u'both',length=5)
            elif n_row==1:
                axarr[j].spines['right'].set_visible(False)
                axarr[j].spines['top'].set_visible(False)
                axarr[j].tick_params(axis=u'both', which=u'both',length=5)
            elif n_col==1:
                axarr[i].spines['right'].set_visible(False)
                axarr[i].spines['top'].set_visible(False)
                axarr[i].tick_params(axis=u'both', which=u'both',length=5)
            else:
                axarr[i][j].spines['right'].set_visible(False)
                axarr[i][j].spines['top'].set_visible(False)
                axarr[i][j].tick_params(axis=u'both', which=u'both',length=5)

            xmin = df_features.loc[:, feature_names[v]].min()
            xmax = df_features.loc[:, feature_names[v]].max()

            if outcome_col is not None:
                for c in outcome_values:
                    lab = None if plot_legend is False else c
                    density = gaussian_kde(df_features.loc[df[outcome_col] == c , feature_names[v]])
                    density.covariance_factor = lambda : cov_factor
                    density._compute_covariance()
                    xs = np.arange(xmin, xmax, 0.1)
                    if (n_row == 1) and (n_col == 1):
                        axarr.plot(xs, density(xs), label = lab)
                        axarr.set_title(feature_names[v])
                        if plot_legend:
                            if (j == 0) & (i == 0):
                                axarr.legend()
                    elif n_row == 1:
                        axarr[j].plot(xs, density(xs), label = lab)
                        axarr[j].set_title(feature_names[v])
                        if plot_legend:
                            if (j == 0) & (i == 0):
                                axarr[j].legend()
                    elif n_col == 1:
                        axarr[i].plot(xs, density(xs), label = lab)
                        axarr[i].set_title(feature_names[v])
                        if plot_legend:
                            if (j == 0) & (i == 0):
                                axarr[i].legend()
                    else:
                        axarr[i][j].plot(xs, density(xs), label = lab)
                        axarr[i][j].set_title(feature_names[v])
                        if plot_legend:
                            if (j == 0) & (i == 0):
                                axarr[i,j].legend()
                v += 1
            else:
                density = gaussian_kde(df_features.loc[ : , feature_names[v]])
                density.covariance_factor = lambda : cov_factor
                density._compute_covariance()
                xs = np.arange(xmin, xmax, 0.1)
                if (n_row == 1) and (n_col == 1):
                    axarr.plot(xs, density(xs))
                    axarr.set_title(feature_names[v])
                    if plot_legend:
                        if (j == 0) & (i == 0):
                            axarr.legend()
                elif n_row == 1:
                    axarr[j].plot(xs, density(xs))
                    axarr[j].set_title(feature_names[v])
                    if plot_legend:
                        if (j == 0) & (i == 0):
                            axarr[j].legend()
                elif n_col == 1:
                    axarr[i].plot(xs, density(xs))
                    axarr[i].set_title(feature_names[v])
                    if plot_legend:
                        if (j == 0) & (i == 0):
                            axarr[i].legend()
                else:
                    axarr[i][j].plot(xs, density(xs))
                    axarr[i][j].set_title(feature_names[v])
                    if plot_legend:
                        if (j == 0) & (i == 0):
                            axarr[i][j].legend()
                v += 1

    if n_col == 1:
        f.subplots_adjust(hspace = 1)
    else:
        f.subplots_adjust(hspace = 0.5)
    return f


def cat_plot(df, outcome_col = None,
             n_col = 3, plot_legend = False,
             f_size = (15,15)):
    '''
    Function to generate stacked bar plots, for examining categorical data

    Inputs:
    df: pandas dataframe of features to plot
    outcome_col: string specifying the outcome column. Cannot be None
    n_col: number of columns
    plot_legend: Boolean. should a legend be included in the plot?
    f_size: tuple of (x size, y size) to specify size of plot
    '''

    if outcome_col is None:
        raise ValueError('outcome column cannot be None')
    else:
        n_features = df.shape[1] -1
        feature_names = [x for x in df.columns.values if x != outcome_col]
        outcome_values = df.loc[:, outcome_col].unique()

    if n_features % n_col == 0:
        n_row = n_features // n_col
    else:
        n_row = n_features // n_col + 1

    f, axarr = plt.subplots(n_row, n_col, figsize = f_size, dpi=80)

    v = 0
    for i in range(n_row):
        for j in range (n_col):

            if v >= n_features:
                if n_row == 1:
                    axarr[j].axis('off')
                    continue
                elif n_col == 1:
                    axarr[i].axis('off')
                    continue
                else:
                    axarr[i][j].axis('off')
                    continue

            if (n_row == 1) and (n_col==1):
                axarr.spines['right'].set_visible(False)
                axarr.spines['top'].set_visible(False)
                axarr.tick_params(axis=u'both', which=u'both',length=5)
            elif n_row == 1:
                axarr[j].spines['right'].set_visible(False)
                axarr[j].spines['top'].set_visible(False)
                axarr[j].tick_params(axis=u'both', which=u'both',length=5)
            elif n_col == 1:
                axarr[i].spines['right'].set_visible(False)
                axarr[i].spines['top'].set_visible(False)
                axarr[i].tick_params(axis=u'both', which=u'both',length=5)
            else:
                axarr[i][j].spines['right'].set_visible(False)
                axarr[i][j].spines['top'].set_visible(False)
                axarr[i][j].tick_params(axis=u'both', which=u'both',length=5)


            n_levels = len(df.loc[:, feature_names[v]].unique())
            width = 0.45
            ind = np.arange(n_levels)

            tmp = (df
            .loc[:, [feature_names[v], outcome_col]]
            .groupby([feature_names[v], outcome_col])[feature_names[v]]
            .count()
            .unstack(outcome_col)
            .fillna(0)
            .loc[:, outcome_values] )

            for d in range(len(outcome_values)):
                if d == 0:
                    d0 = tmp[outcome_values[d]].values
                    if (n_row==1) and (n_col==1):
                        axarr.bar(ind, d0, width, alpha = 0.8, label = outcome_values[d])
                    elif n_row == 1:
                        axarr[j].bar(ind, d0, width, alpha = 0.8, label = outcome_values[d])
                    elif n_col == 1:
                        axarr[i].bar(ind, d0, width, alpha = 0.8, label = outcome_values[d])
                    else:
                        axarr[i][j].bar(ind, d0, width, alpha = 0.8, label = outcome_values[d])
                else:
                    dd = tmp[outcome_values[d]].values
                    if (n_row==1) and (n_col==1):
                        axarr.bar(ind, dd, width, bottom = d0, alpha = 0.8, label = outcome_values[d])
                    elif n_row == 1:
                        axarr[j].bar(ind, dd, width, bottom = d0, alpha = 0.8, label = outcome_values[d])
                    elif n_col == 1:
                        axarr[i].bar(ind, dd, width, bottom = d0, alpha = 0.8, label = outcome_values[d])
                    else:
                        axarr[i][j].bar(ind, dd, width, bottom = d0, alpha = 0.8, label = outcome_values[d])
            if (n_row == 1) and (n_col == 1):
                axarr.set_xticks(ind)
                axarr.set_xticklabels(tmp.index.values.tolist(), rotation = 45)
                axarr.set_title(feature_names[v])
                if plot_legend:
                    if (j == 0) & (i == 0):
                        axarr.legend()
            elif n_row == 1:
                axarr[j].set_xticks(ind)
                axarr[j].set_xticklabels(tmp.index.values.tolist(), rotation = 45)
                axarr[j].set_title(feature_names[v])
                if plot_legend:
                    if (j == 0) & (i == 0):
                        axarr[j].legend()
            elif n_col == 1:
                axarr[i].set_xticks(ind)
                axarr[i].set_xticklabels(tmp.index.values.tolist(), rotation = 45)
                axarr[i].set_title(feature_names[v])
                if plot_legend:
                    if (j == 0) & (i == 0):
                        axarr[j].legend()
            else:
                axarr[i][j].set_xticks(ind)
                axarr[i][j].set_xticklabels(tmp.index.values.tolist(), rotation = 45)
                axarr[i][j].set_title(feature_names[v])
                if plot_legend:
                    if (j == 0) & (i == 0):
                        axarr[i][j].legend()
            v += 1
    if n_col == 1:
        f.subplots_adjust(hspace = 1)
    else:
        f.subplots_adjust(hspace = 0.5)
    return f


def cs(var):
    return (var - var.mean()) / var.max()


def pairwise_plot(df, outcome_col = None, center_scale = True,
                  plot_legend = False, f_size = (15,15)):
    '''
    docstring please
    '''

    if outcome_col is None:
        n_features = df.shape[1]
        feature_names = df.columns.values
        df_features = df
    else:
        n_features = df.shape[1] -1
        feature_names = [x for x in df.columns.values if x != outcome_col]
        df_features = df.loc[:, feature_names]
        outcome_values = sorted([x for x in df.loc[:, outcome_col].unique()])

    if center_scale:
        df_features = df_features.apply(lambda x: cs(x))

    n_features = df_features.shape[1]
    if n_features == 1:
        raise ValueError('not defined for n_features = 1')

    f, axarr = plt.subplots(n_features, n_features, figsize = f_size, dpi=80)

    v = 0
    for i in range(n_features):
        for j in range(n_features):
            axarr[i][j].spines['right'].set_visible(False)
            axarr[i][j].spines['top'].set_visible(False)

            if outcome_col is not None:
                for c in outcome_values:
                    tmp_i = cs(df_features.loc[:, feature_names[i]])
                    tmp_j = cs(df_features.loc[:, feature_names[j]])
                    if j <= i:
                        axarr[i][j].scatter(tmp_i[df[outcome_col] == c],
                                           tmp_j[df[outcome_col] == c],
                                           label = c,
                                           alpha = 0.5,
                                           s = 2)

                        axarr[i][j].set_yticklabels([])
                        axarr[i][j].set_xticklabels([])

                        if j == 0:
                            axarr[i][j].set_ylabel(feature_names[i], rotation = 0,ha='right')

                        if i== n_features - 1:
                            axarr[i][j].set_xlabel(feature_names[j], rotation = 90)
                    else:
                        axarr[i][j].axis('off')
            else:
                tmp_i = cs(df_features.loc[:, feature_names[i]])
                tmp_j = cs(df_features.loc[:, feature_names[j]])
                if j <= i:
                    axarr[i][j].scatter(tmp_i,
                                      tmp_j,
                                      label = c,
                                      alpha = 0.5,
                                      s = 2)

                    axarr[i][j].set_yticklabels([])
                    axarr[i][j].set_xticklabels([])

                    if j == 0:
                        print(len(feature_names[i]))
                        axarr[i][j].set_ylabel(feature_names[i], rotation = 0, ha='right')

                    if i== n_features - 1:
                        axarr[i][j].set_xlabel(feature_names[j], rotation = 90)
                else:
                    axarr[i][j].axis('off')

    f.subplots_adjust(hspace = 0.5)
    f.suptitle('(centered and scaled) pairwise plot', fontsize = 24)
    f.subplots_adjust(top=0.95)
    return f


def plot_prec_recall(y_lab, y_pred_prob, cross=False, figsize=(8,5)):
    '''
    plot precision recall curves

    inputs:
    ylab: np array, true outcome (must be binary)
    y_pred_prob: np array of positive prediction probabilities
    cross: boolean. should a line be drawn to highlight the crossing point?
    figsize: tuple of (x size, y_size) to be passed to the subplot object
    '''

    fig, ax = plt.subplots(figsize=figsize, facecolor='white', edgecolor='white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    prec, rec, thresh = precision_recall_curve(y_lab, y_pred_prob)

    plt.plot(thresh, rec[:-1], label = 'Recall')
    plt.plot(thresh, prec[:-1], label = 'Precision')
    if cross:
        point_ix = np.argmin(prec < rec)
        thresh_cross = thresh[point_ix]
        rec_cross = rec[point_ix]
        plt.plot([thresh_cross, thresh_cross],[0, rec_cross], 'k--', alpha = 0.4)
        plt.plot([0, thresh_cross],[rec_cross, rec_cross], 'k--', alpha = 0.4)

    plt.xlabel("Theshold")
    plt.ylim([0, 1.02])
    plt.xlim([0, 1])
    plt.legend()

    if cross:
        return(ax, (thresh_cross, rec_cross))
    else:
        return ax
