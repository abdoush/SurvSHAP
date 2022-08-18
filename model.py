import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import logging
import os
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import matthews_corrcoef as MCC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

from lifelines import CoxPHFitter

import pickle
from itertools import product
import shap

from sksurv.compare import compare_survival


class Loggable:
    @staticmethod
    def _configure_logger(results_folder, prefix_name, timestamp):
        name = f'Exp_{prefix_name}'
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s:%(message)s')
        file_handler = logging.FileHandler(f'{results_folder}/{prefix_name}_Experiment_{timestamp}.log')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        return logger

class XSurv(Loggable):
    def __init__(self, prefix_name, results_folder='Results', max_k=30, patience=3, z_explained_variance_ratio_threshold=0.99, curves_diff_significance_level=0.05, verbose=True):
        self.results_folder = results_folder + '/' + prefix_name
        if not os.path.exists(self.results_folder):
            os.makedirs(self.results_folder)

        self.prefix_name = prefix_name
        self.timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.logger = self._configure_logger(results_folder=self.results_folder, prefix_name=self.prefix_name, timestamp=self.timestamp)
        self.max_k = max_k
        self.patience = patience
        self.z_explained_variance_ratio_threshold=z_explained_variance_ratio_threshold
        self.curves_diff_significance_level = curves_diff_significance_level
        self.verbose = verbose

    def fit(self, xte_data, survival_curves, event_times, pretrained_clustering_model=None, k=None):

        # explain f(x)=s by explaining h(x): z=t(s), g(z)=c, h(x)=c  , h(x)=g(t(s))

        # data prepatation x, s
        (self.x_train, self.y_train, self.e_train,
         self.x_val, self.y_val, self.e_val,
         self.x_test, self.y_test, self.e_test) = xte_data
        # special for sksurv
        dt = np.dtype('bool,float')
        self.ey_train_surv = np.array([(bool(e), y) for e, y in zip(self.e_train, self.y_train)], dtype=dt)
        self.ey_val_surv = np.array([(bool(e), y) for e, y in zip(self.e_val, self.y_val)], dtype=dt)
        self.ey_test_surv = np.array([(bool(e), y) for e, y in zip(self.e_test, self.y_test)], dtype=dt)
        self.survival_curves_train, self.survival_curves_val, self.survival_curves_test = survival_curves
        self.event_times = event_times

        # transformation to lower dimentions (s -> z), z=t(s)
        self.z_train, self.z_val, self.z_test = self._get_z()

        # clustering (z -> c), c=g(z)
        if k is None:
            self.optimal_k = self._find_optimal_clusters_number()
        else:
            self.optimal_k = k

        self.labels_train, self.labels_val, self.labels_test = self._cluster_in_z(pretrained_model=pretrained_clustering_model)

        if self.verbose:
            #self._plot_z(z, labels, label='train')
            self._plot_z(self.z_train, self.labels_train, label='train')
            self._plot_z(self.z_val, self.labels_val, label='val')
            self._plot_z(self.z_test, self.labels_test, label='test')
            self._plot_curves(curves=self.survival_curves_train, clusters=self.labels_train, event_times=self.event_times, fig_name='Concepts')
            self._plot_context(curves=self.survival_curves_train, clusters=self.labels_train, event_times=self.event_times)

        # classification (x -> c), c=h(x)
        self.classication_model = self._classify()
        self.logger.info('Classification Scores')
        self._classification_evaluation(self.classication_model, self.x_train, self.labels_train, label='Train')
        self._classification_evaluation(self.classication_model, self.x_val, self.labels_val, label='Val')
        self._classification_evaluation(self.classication_model, self.x_test, self.labels_test, label='Test')

    def _classification_evaluation(self, clf, x, y, label):
        self.logger.info(f'{label} Accuracy: {clf.score(x, y)}')
        y_pred = clf.predict(x)

        self.logger.info(f'{label} MCC: {MCC(y, y_pred)}')

        cmx = confusion_matrix(y, y_pred)
        cmx_p = cmx / np.expand_dims(cmx.sum(axis=1), axis=1)

        self.logger.info(cmx)
        self.logger.info(cmx_p)

        all_scores = precision_recall_fscore_support(y, y_pred)
        for measure, scores in zip(['Prec', 'Recall', 'F1', 'Support'], all_scores):
            self.logger.info(measure)
            for i, v in enumerate(scores):
                self.logger.info(f'Pattern_{i}: {v}')
            self.logger.info(f'Avg: {np.mean(scores)}')

    def _get_z(self):
        pca = PCA(n_components=1)
        print("shape:",self.survival_curves_train.shape[1])
        for i in range(1, self.survival_curves_train.shape[1]+1):
            pca = PCA(n_components=i)
            pca.fit(self.survival_curves_train)
            if (pca.explained_variance_ratio_.sum() >= self.z_explained_variance_ratio_threshold) or (pca.explained_variance_ratio_.sum() >= 0.999):
                self.logger.info(f'Z Space Dimensions: {i}')
                break

        z_train = pca.transform(self.survival_curves_train)
        z_val = pca.transform(self.survival_curves_val)
        z_test = pca.transform(self.survival_curves_test)
        if self.verbose:
            if z_train.shape[1] >= 2:
                fig, ax = plt.subplots(1, 3, figsize=(15, 5))
                ax[0].scatter(z_train[:, 0], z_train[:, 1], c=self._get_colors(self.e_train.astype(int)), alpha=0.1)
                ax[1].scatter(z_val[:, 0], z_val[:, 1], c=self._get_colors(self.e_val.astype(int)), alpha=0.1)
                ax[2].scatter(z_test[:, 0], z_test[:, 1], c=self._get_colors(self.e_test.astype(int)), alpha=0.1)
            else:
                plt.figure()
                plt.hist(z_train[:, 0], bins=100)
            plt.savefig(f'{self.results_folder}/{self.prefix_name}_{self.timestamp}_z.pdf', format='pdf', bbox_inches='tight')
            plt.show()
        return z_train, z_val, z_test

    def _cluster_in_z(self, pretrained_model=None):
        if pretrained_model is None:
            self.clustering_mdl = self._base_clustering_model(number_clusters=self.optimal_k).fit(self.z_train)
            pickle.dump(self.clustering_mdl,
                        open(f'{self.results_folder}/{self.prefix_name}_clustering_model_{self.optimal_k}_{self.timestamp}.mdl', 'wb'))
        else:
            self.clustering_mdl = pickle.load(open(pretrained_model, 'rb'))

        labels_train = self.clustering_mdl.predict(self.z_train)
        labels_val = self.clustering_mdl.predict(self.z_val)
        labels_test = self.clustering_mdl.predict(self.z_test)

        return labels_train, labels_val, labels_test

    def _classify(self):
        pass

    def _find_optimal_clusters_number(self):
        diffs = []
        diffs_stds = []
        diffs_means = []
        ntrials = self.patience + 1
        for k in range(2, self.max_k):
            if ntrials == 0:
                break
            diffs_temp = []
            for i in range(10):
                d = self._count_logrank_diffs(self.z_train, ye=self.ey_train_surv, e=self.e_train, number_clusters=k, random_state=i)
                diffs_temp.append(d)
            diffs_means.append(np.mean(diffs_temp))
            diffs.append(np.median(diffs_temp))
            diffs_stds.append(np.std(diffs_temp))
            if diffs[-1] == 1:
                ntrials = self.patience
            else:
                ntrials -= 1


        diffs = list(diffs)
        max_k_with_max_diffs = len(diffs) - diffs[::-1].index(max(diffs)) + 1

        diffs = np.array(diffs)
        diffs_stds = np.array(diffs_stds)
        ks = list(range(2, len(diffs)+2))

        if self.verbose:
            plt.figure(figsize=(4, 4))
            plt.plot(ks, diffs)
            highs = diffs + diffs_stds
            highs = [1 if x>1 else x for x in highs]
            lows = diffs - diffs_stds
            plt.fill_between(ks, lows, highs, alpha=0.2)
            plt.vlines(max_k_with_max_diffs, ymin=min(diffs), ymax=1, color='C1', linestyle='--', label='Last Max')

            plt.xticks(ks, rotation=90)
            #plt.title(f'{max_k_with_max_diffs} Significanly Different Survival Patterns')
            plt.ylabel('% of Different Patterns')
            plt.xlabel('Number of Clusters')
            plt.legend()
            plt.savefig(f'{self.results_folder}/{self.prefix_name}_{self.timestamp}_clusters_number.pdf', format='pdf', bbox_inches='tight')
            plt.show()

        return max_k_with_max_diffs

    def _count_logrank_diffs(self, data, ye, e, number_clusters, random_state=None):
        mdl = self._base_clustering_model(number_clusters=number_clusters, random_state=random_state).fit(data)
        labels = mdl.predict(data)
        total_diffs = 0
        total_comps = 0
        for i, j in product(range(number_clusters), range(number_clusters)):
            if i < j:
                is_diff = self._is_different(i=i, j=j, ye=ye, e=e, group=labels, sig_threshold=self.curves_diff_significance_level)
                total_diffs += int(is_diff)
                total_comps += 1

        return total_diffs / total_comps

    def _def_clustering_model(self, number_clusters):
        pass

    @staticmethod
    def _is_different(i, j, ye, e, group, sig_threshold=0.05):
        f = (group == i) | (group == j)
        ye_sub = ye[f]
        g_sub = group[f]
        e_sub = e[f]

        if (e_sub == 1).any():  # there should be at least one event
            # logrank_test
            try:
                _, p = compare_survival(y=ye_sub, group_indicator=g_sub)
            except:
                p = 1
        else:
            p = 1
        return p <= sig_threshold

    def _plot_z(self, z, labels, label='train'):
        plt.figure(figsize=(5, 5))
        if self.z_train.shape[1] >= 2:
            plt.figure(figsize=(5, 5))
            plt.scatter(z[:, 0], z[:, 1], c=self._get_colors(labels), alpha=0.2)
            plt.xlabel('PC1')
            plt.ylabel('PC2')
        else:
            for i in set(labels):
                f = (labels == i)
                plt.hist(z[f, 0], bins=100, color=f'C{i}')

        plt.savefig(f'{self.results_folder}/{self.prefix_name}_{self.timestamp}_z_clusters_{label}.pdf', format='pdf', bbox_inches='tight')
        plt.show()

    def _plot_curves(self, curves, clusters, event_times, title='', alpha=0.02, sameplot=False, figsize=None, fig_name='curves'):
        if figsize is None:
            figsize = (3, 2)
        else:
            figsize = figsize

        if sameplot:
            n = 1
        else:
            n = len(set(clusters))

        fig, ax = plt.subplots(1, n, figsize=(figsize[0] * n, figsize[1]))
        if n > 1:
            for i, s in enumerate(zip(curves, clusters)):
                ax[s[1]].step(event_times, s[0], where="post", c='C' + str(s[1]), alpha=alpha)
                ax[s[1]].set_xlabel('Time')
                ax[s[1]].set_ylabel('S(t)')
                ax[s[1]].set_ylim(0, 1)
                ax[s[1]].set_title(f'Pattern {s[1]}')
                ax[s[1]].grid(True)
        else:
            for i, s in enumerate(zip(curves, clusters)):
                ax.step(event_times, s[0], where="post", c='C' + str(s[1]), alpha=alpha, label=f'Pattern {i}')
                ax.set_xlabel('Time')
                ax.set_ylabel('S(t)')
                ax.set_ylim(0, 1)
                ax.grid(True)
            lgnd = plt.legend(loc=(1, 0))
            for handle in lgnd.legendHandles:
                handle.set_alpha(1)

        if (not sameplot) and (n > 1):
            for i in set(clusters):
                x_avg = curves[clusters == i].mean(axis=0)
                ax[i].plot(event_times, x_avg, c='k', linestyle='dashed', alpha=1)

        fig.suptitle(title)
        plt.grid(True)
        plt.savefig(f'{self.results_folder}/{self.prefix_name}_{self.timestamp}_{fig_name}_s.pdf', format='pdf', bbox_inches='tight')
        plt.show()

    def _plot_context(self, curves, clusters, event_times):
        fig_name = 'Context'
        means = []
        ccolors = []
        for i in range(self.optimal_k):
            avg_ci = curves[clusters == i].mean(axis=0)
            means.append(avg_ci)
            ccolors.append(i)
        patterns = np.array(means)
        self._plot_curves(curves=patterns, clusters=ccolors, event_times=event_times, sameplot=True, alpha=0.5, figsize=(5, 3), fig_name=fig_name)

    @staticmethod
    def _get_colors(labels):
        return ['C' + str(i) for i in labels]


class ShapExplainer:
    def explain(self, x, features_names_list, classes=None, suffex='test', max_display_total=20, max_display_subgroups=7):
        if classes is None:
            classes = list(range(self.optimal_k))
        self.shap_values = shap.TreeExplainer(self.classication_model).shap_values(x)
        shap.summary_plot(self.shap_values, x, plot_type="bar", class_names=classes, feature_names=features_names_list, show=False, max_display=max_display_total)
        plt.savefig(f'{self.results_folder}/{self.prefix_name}_{self.timestamp}__SHAP_All_{suffex}.pdf', format='pdf', bbox_inches='tight')
        plt.show()
        for i in range(self.optimal_k):
            plt.title(f'Pattern {i}')
            shap.summary_plot(self.shap_values[i], x, feature_names=features_names_list, show=False, max_display=max_display_subgroups)
            # save the drawing to disk
            plt.savefig(f'{self.results_folder}/{self.prefix_name}_{self.timestamp}_SHAP_Class_{suffex}_{i}.pdf', format='pdf', bbox_inches='tight')
            plt.show()


class SurvSHAP(XSurv, ShapExplainer):
    def __init__(self, max_depth=10, *args, **kwargs):
        self.max_depth = max_depth
        super().__init__(*args, **kwargs)

    def _base_clustering_model(self, number_clusters, random_state=10):
        return KMeans(n_clusters=number_clusters, n_init=10, random_state=random_state)

    def _classify(self): # 22
        rf = RandomForestClassifier(max_depth=self.max_depth, random_state=0, oob_score=True, class_weight='balanced_subsample')
        rf.fit(self.x_train, self.labels_train)
        return rf
