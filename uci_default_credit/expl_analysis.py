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

from utils_explan import load_coal_expl_filename  
from utils_explan import load_coal_expl_rand_filename
from utils_explan import compute_glob_value
from utils_explan import compute_aggr_glob_value
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

    # (reference model) load owen values and construct group owen values:

    # first compute singletons
    coal_vals_singles = load_coal_expl_filename(value_type,0,explan_folder_name,header=0)
    # then compute for non-trivial partitions
    shap_group_vals_list = [None] * n_partitions_expl
    coal_vals_list = [None] * n_partitions_expl			
    coal_group_vals_list = [None] * n_partitions_expl
    for m in range(n_partitions_expl):
        if not silent: print("\n")            
        partition     = partition_list_expl[m][0]
        partition_idx = partition_list_expl[m][1]
        coal_vals_list[m] = load_coal_expl_filename( value_type, 
                                                    partition_idx,
                                                    explan_folder_name, 
                                                    header=0)
        coal_group_vals_list[m] = trivial_group_value(coal_vals_list[m],partition)
        shap_group_vals_list[m] = trivial_group_value(coal_vals_singles,partition)

    # efficiency check (reference model):
    for m in range(n_partitions_expl):
        if not silent: 
            print("\nP[{0}], h={1:.4f}".format(partition_list_expl[m][1], thresholds_expl[m]))            
        check_efficiency(coal_vals_list[m],Y_raw_expl_pred, model_mean_train, silent=silent)

    # (random models) load owen values and construct group owen values:
    coal_vals_rm_list = [None] * n_rand_models
    coal_group_vals_rm_list = [None] * n_rand_models
    for k in range(n_rand_models):
        if not silent: print("\n")
        coal_vals_rm_list[k] = [None] * n_partitions_expl
        coal_group_vals_rm_list[k] = [None] * n_partitions_expl

        for m in range(n_partitions_expl):
            partition     = partition_list_expl[m][0]
            partition_idx = partition_list_expl[m][1]
            coal_vals_rm_list[k][m] = load_coal_expl_rand_filename(k,value_type,
                                                                   partition_idx,
                                                                   explan_folder_name,
                                                                   header=0)
            coal_group_vals_rm_list[k][m] =\
                  trivial_group_value(coal_vals_rm_list[k][m],partition)

        # efficiency check (random models):            
        for m in range(n_partitions_expl):                
            if not silent: 
                print("\nrandom model {0}, P[{1}], h={2:.4f}".format(k,partition_list_expl[m][1], thresholds_expl[m]))
            check_efficiency(coal_vals_rm_list[k][m], 
                             Y_raw_expl_pred_rand[k], 
                             rand_model_mean_train[k], 
                             silent=silent)


    # compute differences of explanations:
    if not silent:
        print("\n computions of difference of explanations ...")

    diff_coal_vals_list = [None] * n_rand_models
    diff_coal_group_vals_list = [None] * n_rand_models

    for k in range(n_rand_models):
        diff_coal_vals_list[k] = [None] * n_partitions_expl
        diff_coal_group_vals_list[k] = [None] * n_partitions_expl
        for m in range(n_partitions_expl):
            diff_coal_vals_list[k][m] = coal_vals_list[m]-coal_vals_rm_list[k][m]
            diff_coal_group_vals_list[k][m] = coal_group_vals_list[m]-coal_group_vals_rm_list[k][m]

    # reference model norm computations beta(f*):
    if not silent:
        print("\n computions of global explanations beta(f*) and beta^P(f*) for reference model ...")

    glob_coal_expl, glob_coal_expl_tot = compute_glob_value(coal_vals_list)
    glob_group_coal_expl, glob_group_coal_expl_tot = compute_glob_value(coal_group_vals_list)

    # random model global explanations beta(f_i):
    if not silent:
        print("\n computations of explanations  beta(f_i) for random models ...")        

    glob_coal_expl_rm       = [None] * n_rand_models
    glob_coal_expl_rm_tot   = [None] * n_rand_models
    glob_group_coal_expl_rm = [None] * n_rand_models
    glob_group_coal_expl_rm_tot = [None] * n_rand_models

    for k in range(n_rand_models):
        glob_coal_expl_rm[k], glob_coal_expl_rm_tot[k] = compute_glob_value(coal_vals_rm_list[k])
        glob_group_coal_expl_rm[k], glob_group_coal_expl_rm_tot[k] = compute_glob_value(coal_group_vals_rm_list[k])

    # difference-global explanations beta(f* - f_i):
    if not silent:
        print("\n computations of explanations  beta(f*-f_i) ...")        

    diff_glob_coal_expl       = [None] * n_rand_models
    diff_glob_coal_expl_tot   = [None] * n_rand_models
    diff_glob_group_coal_expl = [None] * n_rand_models
    diff_glob_group_coal_expl_tot = [None] * n_rand_models
    

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
        diff_glob_coal_expl[k], diff_glob_coal_expl_tot[k] \
            = compute_glob_value(diff_coal_vals_list[k])
        diff_glob_group_coal_expl[k], diff_glob_group_coal_expl_tot[k] = \
            compute_glob_value(diff_coal_group_vals_list[k])

    singletons_energy = [ None ] * n_rand_models
    # energy of global model difference for singletons for every model and every partition
    for k in range(n_rand_models):
        singletons_energy[k] = [None] * n_partitions_expl
        for m in range(n_partitions_expl):
            singletons_energy[k][m] = np.sum( [ np.power(diff_glob_group_coal_expl[k][m][idx],2) for idx in singletons_idx[m] ] )

    diff_glob_coal_expl_nonsingl_tot   = [None] * n_rand_models
    diff_glob_group_coal_expl_nonsingl_tot = [None] * n_rand_models

    # compute subvector lengths without singletons:
    for k in range(n_rand_models):
        diff_glob_coal_expl_nonsingl_tot[k] = [None] * n_partitions_expl
        diff_glob_group_coal_expl_nonsingl_tot[k] = [None] * n_partitions_expl
        for m in range(n_partitions_expl):
            diff_glob_coal_expl_nonsingl_tot[k][m] = \
                np.sqrt( np.power(diff_glob_coal_expl_tot[k][m],2) - singletons_energy[k][m] )
            diff_glob_group_coal_expl_nonsingl_tot[k][m] = \
                np.sqrt( np.power(diff_glob_group_coal_expl_tot[k][m],2) - singletons_energy[k][m] )


    if not silent:
        print("\n computations of aggregaetd explanations  beta_aggr(f*-f_i) ...")        

    diff_glob_aggr_coal_expl = [None] * n_rand_models
    for k in range(n_rand_models):
        diff_glob_aggr_coal_expl[k] = compute_aggr_glob_value(diff_glob_coal_expl[k],partition_list_expl_)

    # max difference-global explanations max_i beta(f* - f_i):
    max_diff_glob_coal_expl_tot = [None] * n_partitions_expl
    max_diff_glob_group_coal_expl_tot = [None] * n_partitions_expl
    max_diff_glob_coal_expl_nonsingl_tot = [None] * n_partitions_expl
    max_diff_glob_group_coal_expl_nonsingl_tot = [None] * n_partitions_expl

    L2_norm_owen_expl       = [None] * n_partitions_expl			
    L2_norm_owen_expl_tot   = [None] * n_partitions_expl			
    L2_norm_group_owen_expl = [None] * n_partitions_expl
    L2_norm_group_shap_expl = [None] * n_partitions_expl

    print("\nformatting:")

    for m in range(n_partitions_expl):

        max_diff_glob_coal_expl_tot[m] = \
            np.max([ diff_glob_coal_expl_tot[k][m] for k in range(n_rand_models) ])
        
        max_diff_glob_group_coal_expl_tot[m] = \
            np.max([ diff_glob_group_coal_expl_tot[k][m] for k in range(n_rand_models)])
        
        max_diff_glob_coal_expl_nonsingl_tot[m] = \
            np.max([diff_glob_coal_expl_nonsingl_tot[k][m] for k in range(n_rand_models)])
        
        max_diff_glob_group_coal_expl_nonsingl_tot[m] = \
            np.max([diff_glob_group_coal_expl_nonsingl_tot[k][m] for k in range(n_rand_models)])

        L2_norm_owen_expl[m]       = L2_norm_rv( coal_vals_list[m], axis=0 )
        L2_norm_group_owen_expl[m] = L2_norm_rv( coal_group_vals_list[m], axis=0 )			
        L2_norm_group_shap_expl[m] = L2_norm_rv( shap_group_vals_list[m], axis=0 )			
        L2_norm_owen_expl_tot[m]   = L2_norm_vec( L2_norm_owen_expl[m] )			
        L2_norm_owen_expl[m]       = list(L2_norm_owen_expl[m])
        L2_norm_group_owen_expl[m] = list(L2_norm_group_owen_expl[m])
        L2_norm_group_shap_expl[m] = list(L2_norm_group_shap_expl[m])
        L2_norm_owen_expl_tot[m]   = float(L2_norm_owen_expl_tot[m])
    
    L2_norm_shap_expl = list(L2_norm_rv(coal_vals_singles,axis=0))

    explain_analysis_dict = 	{

        "glob_coal_expl" 	       : glob_coal_expl,
        "glob_coal_expl_tot"       : glob_coal_expl_tot,
        "glob_group_coal_expl"     : glob_group_coal_expl,
        "glob_group_coal_expl_tot" : glob_group_coal_expl_tot,

        "glob_coal_expl_rm" 	      : glob_coal_expl_rm,
        "glob_coal_expl_tot_rm"       : glob_coal_expl_rm_tot,
        "glob_group_coal_expl_rm"     : glob_group_coal_expl_rm,
        "glob_group_coal_expl_rm_tot" : glob_group_coal_expl_rm_tot,

        "diff_glob_coal_expl" 	        : diff_glob_coal_expl,
        "diff_glob_coal_expl_tot"       : diff_glob_coal_expl_tot,
        "diff_glob_group_coal_expl"     : diff_glob_group_coal_expl,
        "diff_glob_group_coal_expl_tot" : diff_glob_group_coal_expl_tot,

        "diff_glob_aggr_coal_expl" : diff_glob_aggr_coal_expl,

        "max_diff_glob_coal_expl_tot"       : max_diff_glob_coal_expl_tot,
        "max_diff_glob_group_coal_expl_tot" : max_diff_glob_group_coal_expl_tot,

        "singletons_idx" : singletons_idx,
        "singletons_energy" : singletons_energy,
        "diff_glob_coal_expl_nonsingl_tot" : diff_glob_coal_expl_nonsingl_tot,
        "diff_glob_group_coal_expl_nonsingl_tot" : diff_glob_group_coal_expl_nonsingl_tot,
        "max_diff_glob_coal_expl_nonsingl_tot" : max_diff_glob_coal_expl_nonsingl_tot,
        "max_diff_glob_group_coal_expl_nonsingl_tot" : max_diff_glob_group_coal_expl_nonsingl_tot,

        "L2_norm_shap_expl" 	  : L2_norm_shap_expl,
        "L2_norm_owen_expl" 	  : L2_norm_owen_expl,
        "L2_norm_owen_expl_tot"	  : L2_norm_owen_expl_tot,
        "L2_norm_group_owen_expl" : L2_norm_group_owen_expl,
        "L2_norm_group_shap_expl" : L2_norm_group_shap_expl

    }

    # save partition tree info into json
    with open(explain_analysis_json, "w") as outfile:
        json.dump(explain_analysis_dict, outfile, indent=4, separators=(',',':'), skipkeys= True )

    ################### 
    # ranking analysis
    ###################

    # L2_norm_shap_expl = list(L2_norm_rv(coal_vals_singles,axis=0))

    # explain_analysis_ranking_dict = 	{

    #     "L2_norm_shap_expl" 	  : L2_norm_shap_expl,
    #     "L2_norm_owen_expl" 	  : L2_norm_owen_expl,
    #     "L2_norm_owen_expl_tot"	  : L2_norm_owen_expl_tot,
    #     "L2_norm_group_owen_expl" : L2_norm_group_owen_expl,
    #     "L2_norm_group_shap_expl" : L2_norm_group_shap_expl
    # }

    # save partition tree info into json
    # with open(explain_analysis_rank_json, "w") as outfile:
    #     json.dump(explain_analysis_ranking_dict, outfile, indent=4, separators=(',',':'), skipkeys= True )



    # if not silent: 
    #     print("\n**********************")
    #     print("[ Ranking analysis ]")
    #     print("**********************")

    # L2_norm_owen_expl       = [None] * n_partitions_expl			
    # L2_norm_owen_expl_tot   = [None] * n_partitions_expl			
    # L2_norm_group_owen_expl = [None] * n_partitions_expl
    # L2_norm_group_shap_expl = [None] * n_partitions_expl
    
    # for m in range(n_partitions_expl):
    #     L2_norm_owen_expl[m]       = L2_norm_rv( coal_vals_list[m], axis=0 )
    #     L2_norm_group_owen_expl[m] = L2_norm_rv( coal_group_vals_list[m], axis=0 )			
    #     L2_norm_group_shap_expl[m] = L2_norm_rv( shap_group_vals_list[m], axis=0 )			
    #     L2_norm_owen_expl_tot[m]   = L2_norm_vec( L2_norm_owen_expl[m] )			
    #     L2_norm_owen_expl[m]       = list(L2_norm_owen_expl[m])
    #     L2_norm_group_owen_expl[m] = list(L2_norm_group_owen_expl[m])
    #     L2_norm_group_shap_expl[m] = list(L2_norm_group_shap_expl[m])
    #     L2_norm_owen_expl_tot[m]   = float(L2_norm_owen_expl_tot[m])


    # load Shapley value (Owen for singeltons):
    # if not silent:
    #     print("Loading explanations for model f*, partition 0 (singletons)")

    # owen_vals_singl = load_coal_expl_filename(value_type, 0, explan_folder_name, header=0)

    # load owen values across given partitions and compute group owen values:
    # owen_vals_list = [None] * n_partitions_expl
    # owen_group_vals_list = [None] * n_partitions_expl
    # shap_group_vals_list = [None] * n_partitions_expl

    # for m in range(len(partition_list_expl)):

    #     partition = partition_list_expl[m][0]
    #     partition_idx = partition_list_expl[m][1]

        # owen_vals_list[m] = load_coal_expl_filename(value_type, 
        #                                             partition_idx, 
        #                                             explan_folder_name, 
        #                                             header=0)
        # compute group explanations:
        # owen_group_vals_list[m] = np.zeros( shape = (owen_vals_list[m].shape[0],len(partition)))
        # shap_group_vals_list[m] = np.zeros( shape = (owen_vals_list[m].shape[0],len(partition)))
        # shap_group_vals_list[m] = np.zeros( shape = (coal_vals_list[m].shape[0],len(partition)))
        # for j in range(len(partition)):				
        #     Sj = partition[j]
            # owen_group_vals_list[m][:,j] = np.sum(owen_vals_list[m][:,Sj],axis=1)            
            # shap_group_vals_list[m][:,j] = np.sum(owen_vals_singl[:,Sj],axis=1)
            # shap_group_vals_list[m][:,j] = np.sum(coal_vals_singles[:,Sj],axis=1)

    # for m in range(n_partitions_expl):
    #     L2_norm_owen_expl[m]       = L2_norm_rv( owen_vals_list[m], axis=0 )
    #     L2_norm_group_owen_expl[m] = L2_norm_rv( owen_group_vals_list[m], axis=0 )			
    #     L2_norm_group_shap_expl[m] = L2_norm_rv( shap_group_vals_list[m], axis=0 )			
    #     L2_norm_owen_expl_tot[m]   = L2_norm_vec( L2_norm_owen_expl[m] )			
    #     L2_norm_owen_expl[m]       = list(L2_norm_owen_expl[m])
    #     L2_norm_group_owen_expl[m] = list(L2_norm_group_owen_expl[m])
    #     L2_norm_group_shap_expl[m] = list(L2_norm_group_shap_expl[m])
    #     L2_norm_owen_expl_tot[m]   = float(L2_norm_owen_expl_tot[m])
    
    # L2_norm_shap_expl = list(L2_norm_rv(owen_vals_singl,axis=0))

    # norm computations:
    # if not silent:
    #     print("\n[computions of explanation norms]")


if __name__=="__main__":
        main()

