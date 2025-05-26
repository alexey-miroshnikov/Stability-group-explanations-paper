############################################## 
# Methods for hierarchical clustering  
############################################## 
# Code authors:
#   Alexey Miroshnikov
#   Konstandinos Kotsiopoulos
# Consultant:
#   Khashayar Filom
############################################## 
# version 1 (January 2025) 
# packages:
#	scipy 1.9.3
###############################################

from copy import copy
from scipy.cluster import hierarchy
##########################################################################################
# var hierarchichal clusterng

def var_clustering(data,var_dist,method='average'):

    assert len(data.shape)==2
    assert data.shape[0]>=1
    assert data.shape[1]>=1
    assert callable(var_dist), "var_dist must be callabl function (.,.)"

    linkage = hierarchy.linkage(data.T,metric=var_dist,method=method)

    root_node, node_list = hierarchy.to_tree(linkage,rd=True)

    node_leaves_list = [None]*len(node_list)
    for j in range(len(node_list)):
        node_leaves_list[j] = node_list[j].pre_order(lambda x: x.id)		
    
    # partition of singletons:
    partitions_info = [{'partition':[],'height': 0.0}]
    for id in node_leaves_list[-1]:
        partitions_info[0]['partition'].append({id})
    
    n_root_leaves = len(node_leaves_list[-1])

    assert n_root_leaves<len(node_leaves_list), \
        "[Error] n_root_leaves must be < the number of nodes in the tree."

    # construct remaining partitions:
    for j in range(n_root_leaves,len(node_leaves_list)):

        partitions_info.append({'partition':[],'height':None} ) 
        partitions_info[-1]['height'] = node_list[j].dist
        merged_set = set(node_leaves_list[j])

        for e in partitions_info[-2]['partition']:
            if not e.issubset(merged_set):
                partitions_info[-1]['partition'].append(copy(e))
        partitions_info[-1]['partition'].append(copy(merged_set))

    # convert elements to lists:
    for j in range(len(partitions_info)):
        partition = partitions_info[j]['partition']
        partition_ = [None]*len(partition)
        for k in range(len(partition)):
            partition_[k] = list(partition[k])
        partitions_info[j]['partition'] = partition_
        
    result = { 'linkage': linkage, 
               'partitions_info': partitions_info, 
               'node_leaves' : node_leaves_list }

    return result


##########################################################################################
# get sliced partition from partitions_info (which is the output of var_clustering)
def get_sliced_partition(partitions_info, threshold):

    assert len(partitions_info)>=1

    # special case (height of leaves level)
    if threshold<=partitions_info[0]['height']: 
        return (copy(partitions_info[0]['partition']),0)
    else: # find index:
        j=1 
        while j<(len(partitions_info)):
            if partitions_info[j-1]['height']<threshold and threshold<=partitions_info[j]['height']:					
                return ( copy(partitions_info[j-1]['partition']), j-1 )
            else:
                j+=1
        # special case: if threshold>partition_info[-1]['height']
        return (copy(partitions_info[-1]['partition']),len(partitions_info)-1)
    
##########################################################################################
# get sliced partition from partitions_info (which is the output of var_clustering)
def _get_sliced_partition(partitions_info, threshold):

    assert len(partitions_info)>=1

    # special case (height of leaves level)
    if threshold<=partitions_info[0]['height']: 
        return copy(partitions_info[0]['partition'])
    else: # find index:
        j=1 
        while j<(len(partitions_info)):
            if partitions_info[j-1]['height']<threshold and threshold<=partitions_info[j]['height']:					
                return copy(partitions_info[j-1]['partition'])
            else:
                j+=1
        # special case: if threshold>partition_info[-1]['height']
        return copy(partitions_info[-1]['partition'])
   
##########################################################################################
# get sliced partition from a partitions_tree_list:
# partition_tree_list[i][0] = list of partitions for i-th item
# partition_tree_list[i][1] = height
def get_sliced_partition_(partitions_tree_list, threshold):

    assert len(partitions_tree_list)>=1

    # special case (height of leaves level)
    if threshold<=partitions_tree_list[0][1]: 
        return (copy(partitions_tree_list[0][0]),0)
    else: # find index:
        j=1 
        while j<(len(partitions_tree_list)):
            if partitions_tree_list[j-1][1]<threshold and threshold<=partitions_tree_list[j][1]:					
                return ( copy(partitions_tree_list[j-1][0]), j-1 )
            else:
                j+=1
        # special case: if threshold>partition_info[-1]['height']
        return ( copy(partitions_tree_list[-1][0]), len(partitions_tree_list)-1 )