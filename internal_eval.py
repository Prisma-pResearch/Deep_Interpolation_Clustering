#!/usr/bin/env 
# -*- coding: utf-8 -*-
"""
Created by yanjun.li at 12/17/19
Dunn index and DB index to ref https://gist.github.com/douglasrizzo/cd7e792ff3a2dcaf27f6
https://www.geeksforgeeks.org/dunn-index-and-db-index-cluster-validity-indices-set-1/
"""
import numpy as np
from sklearn import metrics
from sklearn.metrics.pairwise import euclidean_distances

DIAMETER_METHODS = ['mean_cluster', 'farthest']
CLUSTER_DISTANCE_METHODS = ['nearest', 'farthest']

class DunnIndex(object):
    """
    Dunn Index (DI)
    """
    def __init__(self):
        pass

    def inter_cluster_distances(self, labels, distances, method='nearest'):
        """Calculates the distances between the two nearest points of each cluster.
        :param labels: a list containing cluster labels for each of the n elements
        :param distances: an n x n numpy.array containing the pairwise distances between elements
        :param method: `nearest` for the distances between the two nearest points in each cluster, or `farthest`
        """
        if method not in CLUSTER_DISTANCE_METHODS:
            raise ValueError(
                'method must be one of {}'.format(CLUSTER_DISTANCE_METHODS))

        if method == 'nearest':
            return self._cluster_distances_by_points(labels, distances)
        elif method == 'farthest':
            return self._cluster_distances_by_points(labels, distances, farthest=True)

    def _cluster_distances_by_points(self, labels, distances, farthest=False):
        n_unique_labels = len(np.unique(labels))
        cluster_distances = np.full((n_unique_labels, n_unique_labels),
                                    float('inf') if not farthest else 0)

        np.fill_diagonal(cluster_distances, 0)

        for i in np.arange(0, len(labels) - 1):
            for ii in np.arange(i, len(labels)):
                if labels[i] != labels[ii] and (
                        (not farthest and
                         distances[i, ii] < cluster_distances[labels[i], labels[ii]])
                        or
                        (farthest and
                         distances[i, ii] > cluster_distances[labels[i], labels[ii]])):
                    cluster_distances[labels[i], labels[ii]] = cluster_distances[
                        labels[ii], labels[i]] = distances[i, ii]
        return cluster_distances

    def diameter(self, labels, distances, method='farthest'):
        """Calculates cluster diameters
        :param labels: a list containing cluster labels for each of the n elements
        :param distances: an n x n numpy.array containing the pairwise distances between elements
        :param method: either `mean_cluster` for the mean distance between all elements in each cluster, or `farthest` for the distance between the two points furthest from each other
        """
        if method not in DIAMETER_METHODS:
            raise ValueError('method must be one of {}'.format(DIAMETER_METHODS))

        n_clusters = len(np.unique(labels))
        diameters = np.zeros(n_clusters)

        if method == 'mean_cluster':
            for i in range(0, len(labels) - 1):
                for ii in range(i + 1, len(labels)):
                    if labels[i] == labels[ii]:
                        diameters[labels[i]] += distances[i, ii]

            for i in range(len(diameters)):
                diameters[i] /= sum(labels == i)

        elif method == 'farthest':
            for i in range(0, len(labels) - 1):
                for ii in range(i + 1, len(labels)):
                    if labels[i] == labels[ii] and distances[i, ii] > diameters[labels[i]]:
                        diameters[labels[i]] = distances[i, ii]
        return diameters

    def __call__(self, x, labels, *args, **kwargs):
        """
            Dunn index for cluster validation (larger is better).

            .. math:: D = \\min_{i = 1 \\ldots n_c; j = i + 1\ldots n_c} \\left\\lbrace \\frac{d \\left( c_i,c_j \\right)}{\\max_{k = 1 \\ldots n_c} \\left(diam \\left(c_k \\right) \\right)} \\right\\rbrace

            where :math:`d(c_i,c_j)` represents the distance between
            clusters :math:`c_i` and :math:`c_j`, and :math:`diam(c_k)` is the diameter of cluster :math:`c_k`.
            Inter-cluster distance can be defined in many ways, such as the distance between cluster centroids or between their closest elements. Cluster diameter can be defined as the mean distance between all elements in the cluster, between all elements to the cluster centroid, or as the distance between the two furthest elements.
            The higher the value of the resulting Dunn index, the better the clustering
            result is considered, since higher values indicate that clusters are
            compact (small :math:`diam(c_k)`) and far apart (large :math:`d \\left( c_i,c_j \\right)`).
            :param labels: a list containing cluster labels for each of the n elements
            :param distances: an n x n numpy.array containing the pairwise distances between elements
            :param diameter_method: see :py:function:`diameter` `method` parameter
            :param cdist_method: see :py:function:`diameter` `method` parameter

            .. [Kovacs2005] Kovács, F., Legány, C., & Babos, A. (2005). Cluster validity measurement techniques. 6th International Symposium of Hungarian Researchers on Computational Intelligence.
            """

        distances = euclidean_distances(x)
        ic_distances = self.inter_cluster_distances(labels, distances, "nearest")
        min_distance = min(ic_distances[ic_distances.nonzero()])
        max_diameter = max(self.diameter(labels, distances, "farthest"))

        return min_distance / max_diameter


class Sihouette(object):
    """
    The score is bounded between -1 for incorrect clustering and +1 for highly dense clustering.
    Scores around zero indicate overlapping clusters.
    """
    def __init__(self):
        pass

    def __call__(self, x, labels, *args, **kwargs):
        pairwise_distances_metric = kwargs.get("metrics", "euclidean")
        return metrics.silhouette_score(x, labels, metric=pairwise_distances_metric)


class CHIndex(object):
    """
    Calinski-Harabasz Index
    The score is defined as ratio between the within-cluster dispersion and the between-cluster dispersion.
    The score is higher when clusters are dense and well separated, which relates to a standard concept of a cluster.
    """
    def __init__(self):
        pass

    def __call__(self, x, labels, *args, **kwargs):
        return metrics.calinski_harabasz_score(x, labels)


class DBIndex(object):
    """
    Davies-Bouldin Index
    The minimum score is zero, with lower values indicating better clustering.
    """
    def __init__(self):
        pass

    def __call__(self, x, label, *args, **kwargs):
        return metrics.davies_bouldin_score(x, label)

