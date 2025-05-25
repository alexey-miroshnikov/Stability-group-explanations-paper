import os, sys
script_dirname = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dirname)
main_dirname = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(main_dirname)

import json
import pickle
import numpy as np
import pandas as pd
from argparse import ArgumentParser
import catboost

# from libcatboostcoalexpl import CatBoostCoalGameValue  
from utils_clust import get_sliced_partition

from utils_explan import make_expl_rand_filename
from utils_explan import make_expl_filename


#########################################################################################
## ## explanation analysis
#########################################################################################

def main( ):

    parser = ArgumentParser( description = 'Script for computing explanations.')	

    parser.add_argument('-j', '--json', 
                        default = None, 
                        help = '[either full path to json or json name if example_tag is provided]')
    parser.add_argument('-s', '--silent', 
                        action = "store_true", 
                        help = '[either full path to json or json name if example_tag is provided]')
    args = parser.parse_args()

    ###########################################
    # loading json
    ###########################################

    silent = args.silent

    # default json:
    if args.json is None:
        filename_json = script_dirname + '/' + "default_credit.json"
    else:
        filename_json = args.json
    assert os.path.exists(filename_json), "{0} does not exist".format(filename_json)

    with open(filename_json, 'r') as f:
        json_dict = json.load(f)

    models_folder_name = script_dirname + "/models"
    if not os.path.exists(models_folder_name):
        os.mkdir(models_folder_name)

    explan_folder_name = script_dirname + "/explanations"
    if not os.path.exists(explan_folder_name):
        os.mkdir(explan_folder_name)

    data_proc_train_filename  = script_dirname + '/dataset/data_train.csv'
    data_proc_test_filename   = script_dirname + '/dataset/data_test.csv'	
    
    model_filename = models_folder_name + '/model.dat'
    models_json    = models_folder_name + "/models.json"

    partitions_json    = script_dirname + "/partitions/partitions.json"
    pred_expl_filename = explan_folder_name + "/pred_expl_values.csv"
    explanations_json  = explan_folder_name + "/explanations.json"
    
    ###########################################################################################
    # Compute explanations
    ###########################################################################################

    # load seed:
    seeds=json_dict['seeds']
    np.random.seed(seeds['expl_comp'])
    print("\nseed =", seeds['expl_comp'])

    # load type of the game value:
    value_type = "shapley" #json_dict["value_type"]

    # load partitions:
    with open(partitions_json, 'r') as f:
        partitions_dict = json.load(f)

    # load seed from the previous step:
    with open(models_json, 'r') as f:
        models_dict = json.load(f)

    # load reference model:
    if not silent:
        print("\n[loading model]:", model_filename )
    models_dat_dict = pickle.load(open(model_filename,"rb"))

    # extract partition info:
    thresholds_expl = json_dict['thresholds']
    pt = partitions_dict['partition_dict']

    # partition_tree_list  = [ [pt[str(i)][2],pt[str(i)][0]] for i in range(len(pt)) ]
    # partition_list_expl = [None] * len(thresholds_expl)
    # for j in range(len(partition_list_expl)):
    #     partition_list_expl[j] = \
    #     get_sliced_partition_(partition_tree_list, thresholds_expl[j])            
 
    partition_tree_info  = [{"height":pt[str(i)][0],"partition":pt[str(i)][2]} \
                            for i in range(len(pt))]

    partition_list_expl = [None] * len(thresholds_expl)
    for j in range(len(partition_list_expl)):
        partition_list_expl[j] = \
        get_sliced_partition(partition_tree_info, thresholds_expl[j])            

    # extract model info:
    model = models_dat_dict["model"]        
    rand_model_list = models_dat_dict["random_models"]
    model_type = models_dict["model_type"]        
    pred_names = models_dict["pred_names"]

    # load data:
    if not silent:
        print("\n[loading processed train data]:", data_proc_train_filename )
    df_train = pd.read_csv( data_proc_train_filename )		

    if not silent:
        print("\n[loading processed test data]:", data_proc_test_filename )
    df_test = pd.read_csv( data_proc_test_filename )		

    pred_names = list(df_train.columns)[:-1]
    X_train = np.array( df_train.iloc[:,:-1] ) 
    X_test  = np.array( df_test.iloc[:,:-1] )

    # number of threads to split computation of explanations:
    n_threads_expl = json_dict["n_threads_expl"]
    expl_test = json_dict.get("expl_test", False)

    # pick the dataset to explain:
    if expl_test:
        X_expl = X_test
        if not silent:
            print( "Computing explanations for the test dataset X_test" )
    else:
        X_expl = X_train
        if not silent:
            print( "Computing explanations for the training dataset X_train" )
    
    Y_raw_expl_pred = model.predict( X_expl, prediction_type='RawFormulaVal')

    #################################### 
    # explanations for a reference model
    ####################################

    if not silent:
        print("\n\n*****************************************")            
        print("[Computing {1} value for reference model f*, type {0} ]".format(model_type,value_type))
        print("*********************************************")

    # catcoal = CatBoostCoalGameValue(ml_model=model)
    print(" computing Shapley explanations using native catboost with PreCalc ...")
    catcoal_vals = model.get_feature_importance( data=catboost.Pool(X_expl),
                                          type = 'ShapValues',
                                          shap_mode = "UsePreCalc",
                                          shap_calc_type = "Exact" )    
    print(" done ... ")

    # split explanatins and expectation in the last column:
    Y_train_raw_mean = catcoal_vals[:,-1]
    catcoal_vals = catcoal_vals[:,0:-1]

    # sanity check of efficiency property:
    if not silent: 
        # diff = np.sum( catcoal_vals, axis=1 ) + catcoal.mean - Y_raw_expl_pred_j
        diff = np.sum( catcoal_vals, axis=1 ) + Y_train_raw_mean - Y_raw_expl_pred
        print("\nEfficiency check: ", np.max(np.abs(diff)))

    # save explanations:
    coal_expl_filename = make_expl_filename(value_type,                                                
                                            explan_folder_name)

    if not silent:
        print("\n[saving {0} explanations]:".format(value_type), coal_expl_filename )
    if os.path.exists(coal_expl_filename):
        os.remove(coal_expl_filename)
    pd.DataFrame(catcoal_vals, columns=pred_names).to_csv(coal_expl_filename,index=False)


    # # compute explanations for other partitions
    # for m in range(-1,len(partition_list_expl)):
        
    #     if m==-1: # separate construction for singletons
    #         partition = partition_tree_info[0]["partition"]
    #         height    = partition_tree_info[0]["height"]
    #         # partition = partition_tree_list[0][0]
    #         # height    = partition_tree_list[0][1]
    #         partition_idx = 0
    #     else:
    #         height    = thresholds_expl[m]
    #         partition = partition_list_expl[m][0]
    #         partition_idx = partition_list_expl[m][1]
        
    #     if not silent:					
    #         print("\n[Partition P[{0}]]: ".format(partition_idx), \
    #                 "\nh=", height, ", P[{0}]=".format(partition_idx), partition, "\n", sep="" )

    #     # catcoal.precompute_ensemble_values( value_type, partition, n_threads=n_threads_expl )
    #     # catcoal_vals = catcoal(X_expl)
        
    #     if not silent: # sanity check of efficiency property (catcoal_vals includes mean in the last column):
    #         # diff = np.sum( catcoal_vals, axis=1 ) + catcoal.mean - Y_raw_expl_pred_j
    #         diff = np.sum( catcoal_vals, axis=1 ) + Y_train_raw_mean - Y_raw_expl_pred
    #         print("\nEfficiency check: ", np.max(np.abs(diff)))

    #     # coal_expl_filename = make_coal_expl_filename(value_type, 
    #     #                                              partition_idx,
    #     #                                              explan_folder_name)

    #     coal_expl_filename = make_expl_filename(value_type,                                                
    #                                             explan_folder_name)

    #     if not silent:
    #         print("\n[saving {0} explanations]:".format(value_type), coal_expl_filename )
    #     if os.path.exists(coal_expl_filename):
    #         os.remove(coal_expl_filename)
    #     pd.DataFrame(catcoal_vals, columns=pred_names).to_csv(coal_expl_filename,index=False)

    #################################### 
    # explanations for random models
    ####################################

    if not silent:
        print("\n\n*****************************************")
        print("[Computing {0} explanations for random models]".format(value_type))
        print("*****************************************")

    for j in range(len(rand_model_list)):

        if not silent:
            print("\n explanations for random model [{0}]  ...".format(j))

        Y_raw_expl_pred_j = rand_model_list[j].predict( X_expl, prediction_type='RawFormulaVal')

        # catcoal = CatBoostCoalGameValue(ml_model=rand_model_list[j])

        catcoal_vals = rand_model_list[j].get_feature_importance( data=catboost.Pool(X_expl),
                                                                  type = 'ShapValues',
                                                                  shap_mode = "UsePreCalc",
                                                                  shap_calc_type = "Exact" )        
        if not silent:
            print(" done ...")


        # split explanatins and expectation in the last column:
        Y_train_raw_mean = catcoal_vals[:,-1]
        catcoal_vals = catcoal_vals[:,0:-1]

        # sanity check of efficiency property:
        if not silent: 
            # diff = np.sum( catcoal_vals, axis=1 ) + catcoal.mean - Y_raw_expl_pred_j
            diff = np.sum( catcoal_vals, axis=1 ) + Y_train_raw_mean - Y_raw_expl_pred_j
            print("\nEfficiency check: ", np.max(np.abs(diff)))

        coal_expl_filename = make_expl_rand_filename(j, value_type,
                                                     explan_folder_name)

        if not silent:
            print("\n[saving {0} explanations]:".format(value_type), coal_expl_filename )
        if os.path.exists(coal_expl_filename):
            os.remove(coal_expl_filename)
        pd.DataFrame(catcoal_vals, columns=pred_names).to_csv(coal_expl_filename,index=False)


        # for m in range(len(partition_list_expl)):
            
        #     height    = thresholds_expl[m]
        #     partition = partition_list_expl[m][0]
        #     partition_idx = partition_list_expl[m][1]
            
        #     if not silent:					
        #         print("\n[Partition P[{0}]]: ".format(partition_idx), \
        #                 "\nh=", height, ", P[{0}]=".format(partition_idx), partition, "\n", sep="" )

        #     # catcoal.precompute_ensemble_values( "owen", partition, n_threads=n_threads_expl )
        #     # catcoal_vals = catcoal(X_expl)
            
        #     if not silent: # sanity check of efficiency property:
        #         # diff = np.sum( catcoal_vals, axis=1 ) + catcoal.mean - Y_raw_expl_pred_j
        #         diff = np.sum( catcoal_vals, axis=1 ) + Y_train_raw_mean - Y_raw_expl_pred_j
        #         print("\nEfficiency check: ", np.max(np.abs(diff)))

        #     coal_expl_filename = make_coal_expl_rand_filename(j, value_type,
        #                                                       partition_idx, 
        #                                                       explan_folder_name)

        #     if not silent:
        #         print("\n[saving {0} explanations]:".format(value_type), coal_expl_filename )
        #     if os.path.exists(coal_expl_filename):
        #         os.remove(coal_expl_filename)
        #     pd.DataFrame(catcoal_vals, columns=pred_names).to_csv(coal_expl_filename,index=False)

    #################################################### 
    # saving data X_expl (where X_ave=X_train by design)
    ####################################################

    print("\n[saving X_expl]:", pred_expl_filename)
    if os.path.exists(pred_expl_filename): os.remove(pred_expl_filename)
    pd.DataFrame(X_expl, columns=pred_names).to_csv(pred_expl_filename,index=False)

    explanations_dict = {
            "thresholds" : thresholds_expl,
            "partitions" : partition_list_expl,
            "value_type" : value_type
            }
    
    # save partition tree info into json
    with open(explanations_json, "w") as outfile:
        json.dump(explanations_dict, outfile, indent=4, separators=(',',':'), skipkeys= True )

if __name__=="__main__":
        main()

