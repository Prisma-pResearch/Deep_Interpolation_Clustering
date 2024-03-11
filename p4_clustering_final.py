#!/usr/bin/env
# -*- coding: utf-8 -*-

"""
clustering.py: 
"""
import copy
import os
import os.path as osp
import argparse
import numpy as np
from internal_eval import Sihouette, DBIndex, CHIndex, DunnIndex
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, OPTICS
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances, silhouette_score
from scipy.spatial.distance import cdist
import gc
import pandas as pd
from kneed import KneeLocator
from utils import print_dict_byline, logger
from info import LEGEND_INFO, COHORTS
np.random.seed(123)

##The Silhouette score is bounded from -1 to 1 and higher score means more distinct clusters.
##The Calinski-Harabasz index compares the variance between-clusters to the variance within each cluster. This measure is much simpler to calculate then the Silhouette score however it is not bounded. The higher the score the better the separation is.
##Davies-Bouldin index is the ratio between the within cluster distances and the between cluster distances and computing the average overall the clusters. It is therefore relatively simple to compute, bounded – 0 to 1, lower score is better. 

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cluster_method', default='kmeans', choices=['kmeans', 'dbscan', 'dl', 'optics', 'consensus'],
                        help='For consensus, the labels are generated outside, and then directly loaded.')
    parser.add_argument('--num_clusters', type=int, default=4, help="The number of cluster centers")
    parser.add_argument('--restore_metric', default=['ae_mse', 'loss', 'delta'])    
    parser.add_argument('--opt_eps', type=float, default=1.9, help="The optimal eps value for DBSCAN.")
    parser.add_argument('--dl_cluster_label_type', default='pred', choices=['label', 'pred'],
                        help='Only for dl clustering, use the cluster assignment of cluster_label or cluster_pred')    
    args = parser.parse_args()
    return args


class Cluster(object):
    def __init__(self, args):
        self.args = args
        self.exp_path = os.path.join(os.getcwd(), "Results", "Clustering")
    
    ###load hidden feature (numpy version)
    def load_data(self):
        cohort_lst = []
        for cohort in COHORTS:
            logger.info("*******Load the features for {}*******".format(cohort))
            full_data = np.load(osp.join(self.feat_path, "{}.npy".format(cohort)), allow_pickle=True).item()
            cohort_data = {kept_k: full_data[kept_k] for kept_k in ['encounter_id', 'hidden', 'ob', 'padding_mask']}
            cohort_lst.append(cohort_data)
            logger.debug(cohort_data.keys())
            encounter_id = cohort_data['encounter_id']
            logger.info('Cohort: {}, Sample: {}'.format(cohort, len(encounter_id)))
        self.train_data, self.valid_data, self.test_data = cohort_lst
        self.feat_dim = self.train_data['hidden'].shape[-1]
        
        ###align label based on sbp
    def generate_align_map(self, org_label, ob, padding, feat=None):
        """
        This is only valid for k-means, sort the training cluster id based on the average sbp value.
        For validation and test, use the sorted clustering center to assign cluster id.
        For other cluster algo, which needs to run individually on train/valid/test sets, individually align will not
        generate the same aligned results (special closes sample may assigned to different cluster id,), because the
        order relationship of sbp may differ across train/valid/test.
        ob: [B, num_variables(6), max_timestamp], the heart_rate locates at [:, 2, :]
        padding: [B, num_variables(6), max_timestamp]
        :return:
        """
        sorted_var = ob[:, 0, :]
        sorted_var_padding = padding[:, 0, :]
        sorted_var = sorted_var * sorted_var_padding
        avg_sorted_var = np.sum(sorted_var, axis=1) / np.sum(sorted_var_padding, axis=1)
        n_clusters = len(set(org_label)) - (1 if -1 in org_label else 0)

        cluster_sbp, cluster_idx = [], []
        for i in range(0, n_clusters):
            cluster_sbp.append(np.average(avg_sorted_var[org_label == i]))
            cluster_idx.append(np.where(org_label == i))
        sorted_cluster_ids = np.argsort(cluster_sbp)[::-1]
        align_map = {prev: cur for cur, prev in enumerate(sorted_cluster_ids)}
        align_map = {sorted_prev: align_map[sorted_prev] for sorted_prev in sorted(align_map)}
        logger.info('Align_map: {}'.format(align_map))

        for org_id, new_id in align_map.items():
            org_idx = cluster_idx[org_id]
            org_label[org_idx] = new_id

        new_feat_centers = []
        if feat is not None:
            # Using the aligned labels to generate the clustering center of training feat.
            for i in range(0, n_clusters):
                new_feat_centers.append(np.mean(feat[org_label == i], axis=0))
        return align_map, org_label, new_feat_centers

    ###base on the align map, convert the original label
    def align_labels(self, org_label, align_map):
        n_clusters = len(set(org_label)) - (1 if -1 in org_label else 0)
        cluster_idx = []
        for i in range(0, n_clusters):
            cluster_idx.append(np.where(org_label == i))

        for org_id, new_id in align_map.items():
            org_idx = cluster_idx[org_id]
            org_label[org_idx] = new_id
        return org_label

    ##generate the align map based on the center distance, and align label if able to align
    def align_labels_with_center(self, org_feat, org_label, aligned_feat_centers):
        n_clusters = len(set(org_label)) - (1 if -1 in org_label else 0)
        org_cluster_centers = []
        for i in range(0, n_clusters):
            org_cluster_centers.append(np.mean(org_feat[org_label == i], axis=0))

        # D_ij: dist(D_i from org and D_j from aligned)
        feat_center_dist = pairwise_distances(org_cluster_centers, aligned_feat_centers)
        min_dist_idx = np.argmin(feat_center_dist, axis=1)

        if len(set(min_dist_idx)) != n_clusters:
            logger.info(min_dist_idx)
            raise ValueError('Different org_feat_centers map to a same train_feat_center')

        align_map = dict()
        for org_id, new_id in enumerate(min_dist_idx):
            align_map[org_id] = new_id
        logger.info('Align_map: {}'.format(align_map))

        cluster_idx = []
        for i in range(0, n_clusters):
            cluster_idx.append(np.where(org_label == i))

        for org_id, new_id in align_map.items():
            org_idx = cluster_idx[org_id]
            org_label[org_idx] = new_id
        return org_label
    
    def pred(self, **kwargs):
        """
        This is train and eval the cluster model using the selected optimal number of cluster.
        :param kwargs:
        :return:
        """
        for metric in self.args.restore_metric:
            self.feat_path = osp.join(self.exp_path, "out_feat", metric)
            self.out_path = osp.join(self.exp_path, 'out_feat', '{}_{}'.format(metric, self.args.cluster_method))
            self.out_path = self.out_path + '_aligned'
            os.makedirs(self.out_path, exist_ok=True)
            self.load_data()
        
            if self.args.cluster_method == 'kmeans':
                opt_k = self.args.num_clusters
                overwrite = kwargs.get('overwrite', False)
                logger.info('==> Generate the k-means results with opt-k: {}'.format(opt_k))
        
                kmeans_model = KMeans(n_clusters=opt_k, init='k-means++', n_init=20).fit(self.train_data['hidden'])
                train_raw_label = kmeans_model.predict(self.train_data['hidden'])
                align_map, aligned_label, _ = self.generate_align_map(train_raw_label, self.train_data['ob'], self.train_data['padding_mask'])
        
                # Adjust the cluster center based on align_map
                idp_center = copy.deepcopy(kmeans_model.cluster_centers_)
                for org_id, new_id in align_map.items():
                    kmeans_model.cluster_centers_[new_id] = idp_center[org_id]
        
                for cohort, data in zip(COHORTS, [self.train_data, self.valid_data, self.test_data]):
                    cohort_f = osp.join(self.out_path, '{}_{}.npy'.format(cohort, opt_k))
                    if osp.exists(cohort_f) and not overwrite:
                        logger.info('Not Save for {}.'.format(cohort_f))
                        continue
                    feat = data['hidden']
                    data['cluster_id'] = kmeans_model.predict(feat)
                    del data['ob']
                    del data['padding_mask']
        
                    logger.info('Cohort clustering: {} is done. Save to {}'.format(cohort, cohort_f))
                    np.save(cohort_f, data)
    
            elif self.args.cluster_method == 'dbscan':
                opt_eps = self.args.opt_eps
                overwrite = kwargs.get('overwrite', False)
                logger.info('==> Generate the DBSCAN results with opt-eps: {}'.format(opt_eps))
                train_feat_centers = None
        
                for cohort, data in zip(COHORTS, [self.train_data, self.valid_data, self.test_data]):
                    cohort_f = osp.join(self.out_path, '{}_eps-{}.npy'.format(cohort, opt_eps))
                    if osp.exists(cohort_f) and not overwrite:
                        logger.info('Not Save for {}.'.format(cohort_f))
                        continue
        
                    logger.info('NEW DBSCAN model for {}'.format(cohort))
                    feat = data['hidden']
                    feat_dist = pairwise_distances(feat)
                    min_sample = feat.shape[-1]
                    db = DBSCAN(opt_eps, min_sample, metric='precomputed').fit(feat_dist)
                    raw_label = db.labels_
        
                    # Sort by sbp, but sometimes cannot align over train/valid/test
                    if cohort == 'training':
                        align_map, aligned_label, train_feat_centers = self.generate_align_map(raw_label, data['ob'], data['padding_mask'], feat)
                    else:
                        aligned_label = self.align_labels_with_center(feat, raw_label, train_feat_centers)
        
                    data['cluster_id'] = aligned_label
                    del data['ob']
                    del data['padding_mask']
        
                    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
                    core_samples_mask[db.core_sample_indices_] = True
                    logger.info('core_sample: {}'.format(sum(core_samples_mask)))
        
                    # Number of clusters in labels, ignoring noise if present.
                    n_clusters_ = len(set(aligned_label)) - (1 if -1 in aligned_label else 0)
                    n_noise_ = list(aligned_label).count(-1)
        
                    # Denoise samples
                    denoise_idx = aligned_label != -1
                    denoise_feat = feat[denoise_idx]
                    denoise_label = aligned_label[denoise_idx]
        
                    logger.info('Estimated number of clusters: %d' % n_clusters_)
                    logger.info('Estimated number of noise points: %d' % n_noise_)
                    if n_clusters_ == 0:
                        continue
                    elif n_clusters_ == 1:
                        logger.info('Skip the Silhouette Coefficient calculation.')
                    else:
                        logger.info('Orginal Sample: {} Silhouette Coefficient: {:.5f}'.
                                    format(len(aligned_label), silhouette_score(feat, aligned_label)))
                        logger.info('Denoise sample: {}, Denoise Silhouette Coefficient: {:.5f}'.
                                    format(len(denoise_label), silhouette_score(denoise_feat, denoise_label)))
        
                    logger.info('Cohort clustering: {} is done. Save to {}'.format(cohort, cohort_f))
                    np.save(cohort_f, data)
        
            elif self.args.cluster_method == 'optics':
                pass
        
            elif self.args.cluster_method == 'consensus':
                opt_k = self.args.num_clusters
                overwrite = kwargs.get('overwrite', False)
                logger.info('==> Generate the consensus clustering results with opt-k: {}'.format(opt_k))
        
                # TODO: align the training raw label
                train_raw_label_f = osp.join(self.exp_path, 'out_feat', 'raw_consensus_result', 'training_consensus.csv')
                train_raw_label_df = pd.read_csv(train_raw_label_f)
                train_raw_label = train_raw_label_df[f'k{self.args.num_clusters}'].values
        
                if not any(train_raw_label == 0):
                    # adjust label starting from 0, to be comparable to generate_align_map
                    train_raw_label -= 1
        
                align_map, aligned_label, _ = self.generate_align_map(train_raw_label, self.train_data['ob'],
                                                                      self.train_data['padding_mask'])
        
                # Adjust the cluster results based on align_map
                for cohort, data in zip(COHORTS, [self.train_data, self.valid_data]):
                    cohort_f = osp.join(self.out_path, '{}_{}.npy'.format(cohort, opt_k))
                    if osp.exists(cohort_f) and not overwrite:
                        logger.info('Not Save for {}.'.format(cohort_f))
                        continue
        
                    cohort_raw_label_f = osp.join(self.exp_path, 'out_feat', 'raw_consensus_result', f'{cohort}_consensus.csv')
                    cohort_raw_label_df = pd.read_csv(cohort_raw_label_f)
                    raw_label = cohort_raw_label_df[f'k{self.args.num_clusters}'].values
        
                    if not any(raw_label == 0):
                        # adjust label starting from 0, to be comparable to generate_align_map
                        raw_label -= 1
        
                    tag = max(align_map.keys()) + 10
                    for org_id, new_id in align_map.items():
                        if new_id == 0: # label the 0 as None
                            raw_label[raw_label==org_id] = tag
                        raw_label[raw_label==org_id] = -new_id
                    raw_label[raw_label==tag] = 0
                    new_label = abs(raw_label)
        
                    feat = data['hidden']
                    data['cluster_id'] = new_label
                    del data['ob']
                    del data['padding_mask']
        
                    logger.info('Cohort clustering: {} is done. Save to {}'.format(cohort, cohort_f))
                    np.save(cohort_f, data)
        
            elif self.args.cluster_method == 'dl':
                overwrite = kwargs.get('overwrite', False)
        
                for cohort, data in zip(COHORTS, [self.train_data, self.valid_data, self.test_data]):
                    if self.args.dl_cluster_label_type == 'label':
                        label_prob = data['cluster_label']
                    else:
                        label_prob = data['cluster_pred']
                    opt_k = label_prob.shape[1]
                    label_id = np.argmax(label_prob, 1)
                    data['cluster_id'] = label_id
                    del data['cluster_pred']
                    del data['cluster_label']
        
                    cohort_f = osp.join(self.out_path, '{}_{}.npy'.format(cohort, opt_k))
                    if osp.exists(cohort_f) and not overwrite:
                        logger.info('Not Save for {}.'.format(cohort_f))
                        continue
        
                    logger.info('Cohort clustering: {} is done. Save to {}'.format(cohort, cohort_f))
                    np.save(cohort_f, data)

def main(args):
    cluster = Cluster(args)
    cluster.pred()

if __name__ == '__main__':
    args = get_arguments()
    print_dict_byline(vars(args))
    main(args)
