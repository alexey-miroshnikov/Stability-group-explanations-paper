import os, sys
import random
import json
import time
import numpy as np
import numbers
import pandas as pd
from argparse import ArgumentParser
import matplotlib.pyplot as plt; plt.style.use('ggplot')
import scipy

script_dirname = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dirname)
main_dirname = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(main_dirname)

from utils_clust import var_clustering

##########################################################################################
# metrics for hierarchichal clusterng

import minepy

def mic_dist(x,y):
    dist = 1-minepy.pstats([x,y], est = "mic_e")[0][0]    
    if dist < 0:
        return 0
    else:
        return dist

def abs_corr(x,y):
    dist = 1 - np.abs( np.corrcoef(x,y) )[0,1]
    if dist < 0:
        return 0
    else:
        return dist
     
def var_dist(x,y,metric):

    metric_dict = { "mic"     : mic_dist,
                    "abs_corr": abs_corr  } 

    # metric_dict = { "mic"     : abs_corr,
    #                 "abs_corr": abs_corr  } 

    assert isinstance(metric,str), \
        "[Error] metric must be callable or string."
    assert metric in metric_dict.keys(), \
        "[Error] Metric {0} is not defined.".format(metric)
    _metric = metric_dict[metric]

    return _metric(x,y)

#########################################################################################
#########################################################################################
## Main
#########################################################################################
#########################################################################################

def main( ):

    # Remember to change default in parser and all_examples to number of examples in script
    parser = ArgumentParser( description = 'script for variable clustering: UCI Default Credit.')	
    parser.add_argument('-j',   '--json', default = None, help = '[path to json]')
    parser.add_argument('-s',   '--silent', action = "store_true", help = 'if prints are silent')
    args = parser.parse_args()


    ###########################################
    # loading json with parameters
    ############################################

    if args.json is None:
        filename_json = script_dirname + '/' + "super_conductivity.json"
    else:
        filename_json = args.json
    assert os.path.exists(filename_json), "{0} does not exist".format(filename_json)

    with open(filename_json, 'r') as f:
        json_dict = json.load(f)

    ###########################################
    # create a folder for preprocessed data
    ###########################################

    output_dir = script_dirname + '/partitions'

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    silent = args.silent

    folder_name_pics = output_dir + "/pics"

    if not os.path.exists(folder_name_pics):
        os.mkdir(folder_name_pics)

    data_train_proc_filename = script_dirname + '/dataset/data_train.csv'
    data_test_proc_filename =  script_dirname + '/dataset/data_test.csv'
    var_clustering_json = output_dir + '/partitions.json'

    ###########################################################################################
    # variable clustering
    ###########################################################################################

    seeds = json_dict["seeds"]

    n_samples_clustering = json_dict["n_samples_clustering"]

    if not silent:
        print("\n**********************")
        print("\n[Variable clustering]")
        print("\n**********************")

    if not silent:
        print("\n[loading processed train data]:", data_train_proc_filename )

    df_train = pd.read_csv( data_train_proc_filename )	

    if not silent:
        print("\n[loading processed test data]:", data_test_proc_filename )


    df_test = pd.read_csv( data_test_proc_filename )

    X_train = np.array( df_train.iloc[:,:-1] )

    X_test = np.array( df_test.iloc[:,:-1] )

    # generate numpy seed:
    np.random.seed(seeds["var_clust"])
    # set a random seed for random.sample
    random_state = np.random.randint(0,999999999) 
    random.seed(random_state)

    if not silent:
        print("\n numpy-seed = ", seeds["var_clust"])
    if not silent:
        print("\n random-seed = ", random_state)


    X = np.concatenate((X_train,X_test))
    
    if n_samples_clustering is None:

            X_clust = X

    else:

        assert isinstance(n_samples_clustering,numbers.Integral) \
                and n_samples_clustering > 0, \
            "n_samples_clustering must be positive integer."

        if n_samples_clustering>=X.shape[0]:

            X_clust = X

        else:

            idx = random.sample( range(X.shape[0]),n_samples_clustering )

            X_clust = X[idx,:]


    start_time = time.time()

    if not silent:

        print("\n[clustering predictors ({0} samples)]".format(X_clust.shape[0]))

    dep_metric = json_dict["dep_metric"]

    assert dep_metric in ["abs_corr","mic"],\
        "dependency metric {0} is not defined".format(dep_metric)
        
    def _var_dist(x,y):

        return var_dist(x,y,metric=dep_metric)

    pt = var_clustering(data=X_clust, var_dist=_var_dist, method='average')

    end_time = time.time()

    print("clustering time = ", end_time - start_time, "\n")

    pred_labels = list(df_train.columns)[:-1]

    #printing nested partitions into json:
    linkage_matrix = pt['linkage']
    heights = [None]*len(pt['partitions_info'])
    partition_dict = dict()
    for j in range(len(pt['partitions_info'])):
        partition_j = pt['partitions_info'][j]['partition']
        heights[j]  = pt['partitions_info'][j]['height']
        max_group_size = int(np.max([len(S) for S in partition_j]))
        partition_dict[j] = (heights[j],(len(partition_j),max_group_size),partition_j)

    if not silent:
        print("heights = ", heights )

    # test printing nodes on the screen:
    if not silent:
        print("\n\n [linkage]")
        print(pt['linkage'])

        print("\n\n [nested partitions]")
        for j in range(len(pt['partitions_info'])):
            partition_j   = pt['partitions_info'][j]['partition']
            sub_count_list = [ len(S) for S in partition_j ]
            max_group_size = np.max(np.array(sub_count_list))
            print("\n[{0}]:".format(j), \
                    (len(partition_j),max_group_size), \
                    ", h=", heights[j], ", " , partition_j, sep="")
    
    if not silent:
        print("\n[saving variable clustering json]:", var_clustering_json)

    linkage_filename_base = "linkage.csv"

    linkage_filename = output_dir + "/" + linkage_filename_base

    var_clustering_dict = {  
                        "coales_heights"  : heights,
                        "labels"          : pred_labels,
                        "partition_dict"  : partition_dict,
                        "linkage_filename_base" : linkage_filename_base,
                        "dep_metric" : dep_metric
                        }
    
    # save partition tree info into json
    with open(var_clustering_json, "w") as outfile:
        json.dump(var_clustering_dict, outfile, indent=4, separators=(',',':'), skipkeys=True)

    pd.DataFrame(linkage_matrix).to_csv(linkage_filename,index=False,header=None)

    pred_labels_adj = [""] * len(pred_labels)
    for i in range(len(pred_labels_adj)):
        pred_labels_adj[i] = r"$X_{" + r"{0}".format(i+1) + r"}$ = " + str(pred_labels[i])
        
    # plotting dedrogram:
    plt.figure(figsize=(10,6))    
    scipy.cluster.hierarchy.dendrogram(linkage_matrix, 
                                       labels=pred_labels_adj,
                                       orientation='right')
    plt.ylabel('features')

    if dep_metric=="mic":
        plt.xlabel('$1 - MIC_e$')
        plt.title('MIC$_e$ Hierarchical Clustering')
    elif dep_metric=="abs_corr":
        plt.xlabel('$1 - |corr|$')
        plt.title('|corr| Hierarchical Clustering')
    else:
        plt.xlabel('dependency')
        plt.title('Hierarchical Clustering')
    
    plt.tight_layout()					
    plt.savefig(fname = folder_name_pics+'/hierclust_average.png')        
    
    plt.close()

if __name__=="__main__":
        main()