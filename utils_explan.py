import numpy as np
import pandas as pd
from utils import L2_norm_rv
from utils import L2_norm_vec

def make_coal_expl_rand_filename(model_number,
                                    value_type,
                                    partition_idx,
                                    folder_name = "."):
    return folder_name + \
        "/expl_rand_model_{0}_{1}_value_partition_{2}.csv".format(model_number,value_type,partition_idx)				

def make_coal_expl_filename(value_type,
                            partition_idx,
                            folder_name = "."):
    return folder_name + \
        "/expl_model_{1}_value_partition_{2}.csv".format('',value_type,partition_idx)	


def load_coal_expl_filename(value_type,
                            partition_idx, 
                            folder_name,
                            header=None, 
                            verbose=True):
    expl_filename = make_coal_expl_filename(value_type,
                                            partition_idx,
                                            folder_name)
    if verbose:
        print("Loading explanations: model {0}, partition {1}, {2}"\
            .format( '', partition_idx, expl_filename) )
    return np.array(pd.read_csv(expl_filename, header=header))


def load_coal_expl_rand_filename(model_number,
                                    value_type,
                                    partition_idx,
                                    folder_name, 
                                    header=None, verbose=True):
    
    expl_filename = make_coal_expl_rand_filename(model_number,
                                                 value_type,
                                                 partition_idx,
                                                 folder_name)
    if verbose:
        print("Loading explanations: random model, partition {1}, {2}"\
            .format( '', partition_idx, expl_filename) )
    return np.array(pd.read_csv(expl_filename, header=header))


##################################################################### 

def compute_glob_value(value_list):

    beta     = [None] * len(value_list)
    beta_sub = [None] * len(value_list)
    beta_tot = [None] * len(value_list)
        
    for m in range(len(value_list)):        
        beta[m]  = L2_norm_rv( value_list[m], axis=0 )        
        beta_tot[m] = float(L2_norm_vec(beta[m]))
        beta[m] = list(beta[m])

    return beta, beta_tot

##################################################################### 

def compute_aggr_glob_value(glob_value_list,partition_list):
    assert len(glob_value_list)==len(partition_list)
    glob_aggr_value_list = [None] * len(glob_value_list)
    for m in range(len(partition_list)):
        glob_aggr_value_list[m] = aggregate_vector(glob_value_list[m],partition_list[m])
    return glob_aggr_value_list

##################################################################### 

def aggregate_vector(vector,partition):

    vector_ = np.array(vector)
    assert len(vector_.shape)==1, "vector must be 1-dimensional."
    subvector = [ L2_norm_vec(vector_[partition[j]]) for j in range(len(partition)) ]
    
    return subvector

##################################################################### 

def check_efficiency(value, response, response_mean, silent=True ):
    
    assert len(value.shape)==2, "value must be 2-dimensional."

    assert len(response.shape)==1, "response must be 1-dimensional."
    
    assert value.shape[0]==response.shape[0], "value.shape[0] and response.shape[0] must be the same."

    diff = np.sum(value,axis=1) + response_mean - response

    if not silent:
        print("Efficiency check: ", np.max(np.abs(diff)))

    return diff


##################################################################### 

def trivial_group_value(value, partition):
    
    assert len(value.shape)==2, "value must be 2-dimensional"

    group_value = np.zeros( shape = (value.shape[0],len(partition)))            
    for j in range(len(partition)):				
        Sj = partition[j]
        group_value[:,j] = np.sum(value[:,Sj],axis=1)

    return group_value