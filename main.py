"""
Student template code for Project 3
Student will implement five functions:

slow_closest_pair(cluster_list)
fast_closest_pair(cluster_list)
closest_pair_strip(cluster_list, horiz_center, half_width)
hierarchical_clustering(cluster_list, num_clusters)
kmeans_clustering(cluster_list, num_clusters, num_iterations)

where cluster_list is a 2D list of clusters in the plane
"""

import math
import alg_cluster



######################################################
# Code for closest pairs of clusters

def pair_distance(cluster_list, idx1, idx2):
    """
    Helper function that computes Euclidean distance between two clusters in a list

    Input: cluster_list is list of clusters, idx1 and idx2 are integer indices for two clusters

    Output: tuple (dist, idx1, idx2) where dist is distance between
    cluster_list[idx1] and cluster_list[idx2]
    """
    return (cluster_list[idx1].distance(cluster_list[idx2]), min(idx1, idx2), max(idx1, idx2))


def slow_closest_pair(cluster_list):
    """
    Compute the distance between the closest pair of clusters in a list (slow)

    Input: cluster_list is the list of clusters

    Output: tuple of the form (dist, idx1, idx2) where the centers of the clusters
    cluster_list[idx1] and cluster_list[idx2] have minimum distance dist.
    """
    distance = float('inf')
    idx1 = -1
    idx2 = -1

    for idx_u in range(len(cluster_list)):
        for idx_v in range(len(cluster_list)):
            if idx_u != idx_v:
                if pair_distance(cluster_list,idx_u,idx_v)[0] < distance:
                    distance = pair_distance(cluster_list,idx_u,idx_v)[0]
                    idx1 = idx_u
                    idx2 = idx_v
    return (distance, idx1, idx2)




def fast_closest_pair(cluster_list):
    """
    Compute the distance between the closest pair of clusters in a list (fast)

    Input: cluster_list is list of clusters SORTED such that horizontal positions of their
    centers are in ascending order

    Output: tuple of the form (dist, idx1, idx2) where the centers of the clusters
    cluster_list[idx1] and cluster_list[idx2] have minimum distance dist.
    """

    n_length = len(cluster_list)
    if n_length <= 3:
        result = slow_closest_pair(cluster_list)
    else:
        mid_index = n_length/2
        left_cluster_list = cluster_list[:mid_index]
        right_cluster_list = cluster_list[mid_index:]
        left_result = fast_closest_pair(left_cluster_list)
        right_result = fast_closest_pair(right_cluster_list)

        if left_result <= right_result:
            result = left_result
        else:
            result = (right_result[0], right_result[1] + mid_index, right_result[2] + mid_index)

        mid = 0.5 * (cluster_list[mid_index-1].horiz_center() + cluster_list[mid_index].horiz_center())
        result = min([result, closest_pair_strip(cluster_list,mid,result[0])],key= lambda my_result: my_result[0])
    return (result)


def closest_pair_strip(cluster_list, horiz_center, half_width):
    """
    Helper function to compute the closest pair of clusters in a vertical strip

    Input: cluster_list is a list of clusters produced by fast_closest_pair
    horiz_center is the horizontal position of the strip's vertical center line
    half_width is the half the width of the strip (i.e; the maximum horizontal distance
    that a cluster can lie from the center line)

    Output: tuple of the form (dist, idx1, idx2) where the centers of the clusters
    cluster_list[idx1] and cluster_list[idx2] lie in the strip and have minimum distance dist.
    """
    # create a list of indexes in range (mid, width) and sort it according to corresponding vertical coordinate
    s_list = []
    vertical_centers_list = []
    for idx_i in range(len(cluster_list)):
        if abs(cluster_list[idx_i].horiz_center() - horiz_center) < half_width:
            s_list.append(idx_i)
            vertical_centers_list.append(cluster_list[idx_i].vert_center())
    zipped_list = zip(vertical_centers_list,s_list)
    zipped_list.sort()
    s_list = [s_value for dummy, s_value in zipped_list]

    #initilize
    k_length = len(s_list)
    distance = float('inf')
    idx_i = -1
    idx_j = -1

    #cover boundry conditions
    if k_length < 2:
        return (distance, idx_i, idx_j)
    if k_length == 2:
        idx_i = s_list[0]
        idx_j = s_list[1]
        distance = cluster_list[idx_i].distance(cluster_list[idx_j])

    #meat of the algorithm
    for idx_u in range(k_length-1):
        for idx_v in range(idx_u+1, min([idx_u + 4, k_length])):
            distance_uv = cluster_list[s_list[idx_u]].distance(cluster_list[s_list[idx_v]])
            if distance_uv < distance:
                distance = distance_uv
                idx_i = s_list[idx_u]
                idx_j = s_list[idx_v]

    return (distance, min([idx_i,idx_j]), max([idx_i,idx_j]))


######################################################################
# Code for hierarchical clustering


def hierarchical_clustering(cluster_list, num_clusters):
    """
    Compute a hierarchical clustering of a set of clusters
    Note: the function may mutate cluster_list

    Input: List of clusters, integer number of clusters
    Output: List of clusters whose length is num_clusters
    """
    while len(cluster_list) > num_clusters:
        cluster_list.sort(key=lambda cluster: cluster.horiz_center())#needed because fast_closest_pair requires a horizonally sorted list
        closest_pair = fast_closest_pair(cluster_list)
        idx_i = closest_pair[1]
        idx_j = closest_pair[2]
        cluster_i = cluster_list[idx_i]
        cluster_j = cluster_list[idx_j]
        merged_cluster = cluster_i.merge_clusters(cluster_j)
        cluster_list.append(merged_cluster)
        cluster_list.remove(cluster_i)
        cluster_list.remove(cluster_j)

    return cluster_list


######################################################################
# Code for k-means clustering


def kmeans_clustering(cluster_list, num_clusters, num_iterations):
    """
    Compute the k-means clustering of a set of clusters
    Note: the function may not mutate cluster_list

    Input: List of clusters, integers number of clusters and number of iterations
    Output: List of clusters whose length is num_clusters
    """

    # position initial cluster centers at the location of clusters with largest populations
    cluster_list_copy = sorted(cluster_list, key= lambda cluster: cluster.total_population(), reverse=True)
    result_cluster_list = cluster_list_copy[:num_clusters]
    centers = [(cluster.horiz_center(), cluster.vert_center()) for cluster in result_cluster_list]

    for dummy_index in range(num_iterations):
        # create list of empty clusters at the centers
        new_clusters = [alg_cluster.Cluster(set([]),center[0],center[1],0,0) for center in centers]
        #add each county to the nearest empty cluster
        for county in cluster_list:
            nearest_cluster = [float('inf'), None]
            for index in range(len(result_cluster_list)):
                if inter_cluster_distance(county,result_cluster_list[index]) < nearest_cluster[0]:
                    nearest_cluster = [inter_cluster_distance(county,result_cluster_list[index]), index]
            new_clusters[nearest_cluster[1]].merge_clusters(county)
        #update the result_cluster_list
        result_cluster_list = new_clusters
    return result_cluster_list

def inter_cluster_distance(cluster1, cluster2):
    """
    Helper method to calculate the distance between two clusters
    :param cluster1:
    :param cluster2:
    :return: Euclidean distance
    """
    return math.sqrt(math.pow((cluster1.horiz_center()-cluster2.horiz_center()),2) + \
           math.pow((cluster1.vert_center()-cluster2.vert_center()),2))
