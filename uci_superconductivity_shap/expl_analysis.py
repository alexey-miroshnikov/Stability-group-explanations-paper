import os, sys
import json
import pickle
import numpy as np
import pandas as pd
from argparse import ArgumentParser

script_dirname = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dirname)
main_dirname = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(main_dirname)

from utils_explan import load_expl_filename  
from utils_explan import load_expl_rand_filename

from utils_explan import compute_glob_value
from utils_explan import compute_glob_value_triv
from utils_explan import compute_aggr_glob_value_triv
from utils_explan import check_efficiency
from utils_explan import trivial_group_value

from utils import L2_norm_rv
from utils import L2_norm_vec




#########################################################################################
## explanation analysis
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
        filename_json = script_dirname + '/' + "super_conductivity.json"
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

    explain_analysis_json = explan_folder_name + "/explain_analysis.json"   

    model_filename = models_folder_name + '/model.dat'
    models_json    = models_folder_name + "/models.json"   

    pred_expl_filename = explan_folder_name + "/pred_expl_values.csv"
    explanations_json  = explan_folder_name + "/explanations.json"
    

    ###########################################################################################
    # Compute explanations
    ###########################################################################################


    if not silent: 
        print("\n**********************")
        print("[ Step 5: explanations postprocessing ]")
        print("**********************")


    if not silent: 
        print("\n**********************")
        print("[ stability analysis ]")
        print("**********************")

    with open(models_json, 'r') as f:
        models_dict = json.load(f)

    with open( explanations_json, 'r') as f:
        explanations_dict = json.load(f)

    # extract partitions an thresholds:
    partition_list_expl = explanations_dict["partitions"]
    thresholds_expl = explanations_dict["thresholds"]        
    value_type = explanations_dict["value_type"]
    n_partitions_expl = len(partition_list_expl)
    partition_list_expl_ = [None] * n_partitions_expl
    for j in range(n_partitions_expl):
        partition_list_expl_[j] = partition_list_expl[j][0] # pointer only to partition

    # loading models:
    if not silent:
        print("\n[loading models]:", model_filename )
    model_dict = pickle.load(open(model_filename,"rb"))

    # extract model info:
    model = model_dict["model"]
    rand_model_list = model_dict["random_models"]    
    model_mean_train = models_dict["model_mean_train"]
    rand_model_mean_train = models_dict["rand_model_mean_train"]    
    n_rand_models = len(rand_model_list)


    # loading X_expl:
    X_expl = np.array(pd.read_csv(pred_expl_filename,header = 0))        
    # predicting Y values at X_expl (reference model)
    Y_raw_expl_pred = model.predict(X_expl,prediction_type='RawFormulaVal')
    # predicting Y values at X_expl (random models)
    Y_raw_expl_pred_rand = [None] * len(rand_model_list)
    for k in range(len(rand_model_list)):
        Y_raw_expl_pred_rand[k] = rand_model_list[k].predict(X_expl,prediction_type='RawFormulaVal')

    # first load shapley value for the reference model:
    shap_vals_singles = load_expl_filename(value_type,explan_folder_name,header=0)

    # efficiency check (reference model):    
    check_efficiency(shap_vals_singles,
                     Y_raw_expl_pred,
                     model_mean_train, 
                     silent=silent)

    # then compute for non-trivial partitions
    shap_group_vals_list = [None] * n_partitions_expl
    for m in range(n_partitions_expl):
        partition     = partition_list_expl[m][0]
        shap_group_vals_list[m] = trivial_group_value(shap_vals_singles,partition)


    # (random models) load owen values and construct group owen values:
    
    shap_vals_rm_list = [None] * n_rand_models
    
    shap_group_vals_rm_list = [None] * n_rand_models

    for k in range(n_rand_models):

        if not silent: print("\n")

        shap_group_vals_rm_list[k] = [None] * n_partitions_expl

        # load values for k-th model:
        shap_vals_rm_list[k] = load_expl_rand_filename(k,value_type,
                                                        explan_folder_name,
                                                        header=0)

        for m in range(n_partitions_expl):
            partition = partition_list_expl[m][0]
            shap_group_vals_rm_list[k][m] =\
                trivial_group_value(shap_vals_rm_list[k],partition)                

        # efficiency check (random models):                            
        check_efficiency(shap_vals_rm_list[k],
                         Y_raw_expl_pred_rand[k], 
                         rand_model_mean_train[k], 
                         silent=silent)

    # compute differences of explanations:
    if not silent:
        print("\n computions of difference of explanations ...")

    diff_shap_vals_list = [None] * n_rand_models
    diff_shap_group_vals_list = [None] * n_rand_models

    for k in range(n_rand_models):
        diff_shap_group_vals_list[k] = [None] * n_partitions_expl
        diff_shap_vals_list[k] = shap_vals_singles-shap_vals_rm_list[k]
        for m in range(n_partitions_expl):
            diff_shap_group_vals_list[k][m] =\
                shap_group_vals_list[m]-shap_group_vals_rm_list[k][m]

    # random model global explanations beta(f_i):
    if not silent:
        print("\n computations of explanations  beta(f_i) for random models ...")        

    # difference-global explanations beta(f* - f_i):
    if not silent:
        print("\n computations of explanations  beta(f*-f_i) ...")        

    diff_glob_shap_expl       = [None] * n_rand_models
    diff_glob_shap_expl_tot   = [None] * n_rand_models
    diff_glob_group_shap_expl = [None] * n_rand_models
    diff_glob_group_shap_expl_tot = [None] * n_rand_models
    

    # find partitions of singeltons for every partition:
    singletons_idx = [ None ] * n_partitions_expl        
    for m in range(n_partitions_expl):
        partition = partition_list_expl_[m]
        singletons_idx[m] = []            
        for j in range(len(partition)):
            if len(partition[j])==1:
                singletons_idx[m].append(j)

    # global feature attribution for the model difference
    for k in range(n_rand_models):

        diff_glob_shap_expl[k], diff_glob_shap_expl_tot[k] \
            = compute_glob_value_triv(diff_shap_vals_list[k])

        diff_glob_group_shap_expl[k], diff_glob_group_shap_expl_tot[k] = \
            compute_glob_value(diff_shap_group_vals_list[k])

    singletons_energy = [ None ] * n_rand_models
    # energy of global model difference for singletons for every model and every partition
    for k in range(n_rand_models):
        singletons_energy[k] = [None] * n_partitions_expl
        for m in range(n_partitions_expl):
            singletons_energy[k][m] \
            = np.sum( [ np.power(diff_glob_group_shap_expl[k][m][idx],2) for idx in singletons_idx[m] ] )

    diff_glob_shap_expl_nonsingl_tot   = [None] * n_rand_models
    diff_glob_group_shap_expl_nonsingl_tot = [None] * n_rand_models

    # compute subvector lengths without singletons:
    for k in range(n_rand_models):
        diff_glob_shap_expl_nonsingl_tot[k] = [None] * n_partitions_expl
        diff_glob_group_shap_expl_nonsingl_tot[k] = [None] * n_partitions_expl
        for m in range(n_partitions_expl):
            diff_glob_shap_expl_nonsingl_tot[k][m] = \
                np.sqrt( np.power(diff_glob_shap_expl_tot[k],2) - singletons_energy[k][m] )                
            diff_glob_group_shap_expl_nonsingl_tot[k][m] = \
                np.sqrt( np.power(diff_glob_group_shap_expl_tot[k][m],2) - singletons_energy[k][m] )   

    if not silent:
        print("\n computations of aggregated explanations  beta_aggr(f*-f_i) ...")        

    diff_glob_aggr_shap_expl = [None] * n_rand_models
    for k in range(n_rand_models):
        diff_glob_aggr_shap_expl[k] = \
            compute_aggr_glob_value_triv(diff_glob_shap_expl[k], partition_list_expl_)

    # max difference-global explanations max_i beta(f* - f_i):
    max_diff_glob_shap_expl_nonsingl_tot = [None] * n_partitions_expl
    max_diff_glob_group_shap_expl_tot = [None] * n_partitions_expl
    max_diff_glob_group_shap_expl_nonsingl_tot = [None] * n_partitions_expl
    
    L2_norm_group_shap_expl = [None] * n_partitions_expl

    print("\nformatting:")

    max_diff_glob_shap_expl_tot = \
        np.max([ diff_glob_shap_expl_tot[k] for k in range(n_rand_models) ])        

    for m in range(n_partitions_expl):

        max_diff_glob_group_shap_expl_tot[m] = \
            np.max([ diff_glob_group_shap_expl_tot[k][m] for k in range(n_rand_models)])
        
        max_diff_glob_shap_expl_nonsingl_tot[m] = \
            np.max([diff_glob_shap_expl_nonsingl_tot[k][m] for k in range(n_rand_models)])
        
        max_diff_glob_group_shap_expl_nonsingl_tot[m] = \
            np.max([diff_glob_group_shap_expl_nonsingl_tot[k][m] for k in range(n_rand_models)])
        	
        L2_norm_group_shap_expl[m] = L2_norm_rv( shap_group_vals_list[m], axis=0 )			        
        L2_norm_group_shap_expl[m] = list(L2_norm_group_shap_expl[m])
    
    L2_norm_shap_expl = list(L2_norm_rv(shap_vals_singles,axis=0))
    L2_norm_shap_expl_tot = L2_norm_vec(np.array(L2_norm_shap_expl))

    explain_analysis_dict = 	{

        "diff_glob_shap_expl" 	        : diff_glob_shap_expl,
        "diff_glob_shap_expl_tot"       : diff_glob_shap_expl_tot,
        "diff_glob_group_shap_expl"     : diff_glob_group_shap_expl,
        "diff_glob_group_shap_expl_tot" : diff_glob_group_shap_expl_tot,

        "diff_glob_aggr_shap_expl" : diff_glob_aggr_shap_expl,

        "max_diff_glob_shap_expl_tot"       : max_diff_glob_shap_expl_tot,
        "max_diff_glob_group_shap_expl_tot" : max_diff_glob_group_shap_expl_tot,

        "singletons_idx" : singletons_idx,
        "singletons_energy" : singletons_energy,
        "diff_glob_shap_expl_nonsingl_tot" : diff_glob_shap_expl_nonsingl_tot,
        "diff_glob_group_shap_expl_nonsingl_tot" : diff_glob_group_shap_expl_nonsingl_tot,
        "max_diff_glob_shap_expl_nonsingl_tot" : max_diff_glob_shap_expl_nonsingl_tot,
        "max_diff_glob_group_shap_expl_nonsingl_tot" : max_diff_glob_group_shap_expl_nonsingl_tot,

        "L2_norm_shap_expl" 	  : L2_norm_shap_expl,
        "L2_norm_group_shap_expl" : L2_norm_group_shap_expl,
        "L2_norm_shap_expl_tot"   : L2_norm_shap_expl_tot    

    }

    # save partition tree info into json
    with open(explain_analysis_json, "w") as outfile:
        json.dump(explain_analysis_dict, outfile, indent=4, separators=(',',':'), skipkeys= True )


if __name__=="__main__":
        main()

