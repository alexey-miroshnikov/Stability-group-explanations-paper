############################################## 
############################################## 
# Code authors:
#   Alexey Miroshnikov
#   Konstandinos Kotsiopoulos
# Consultant:
#   Khashayar Filom
############################################## 
# Version 1: May 2025
############################################## 

import os, sys
import pickle
from copy import deepcopy
import catboost
import numpy as np
import pandas as pd
import json
import sklearn.metrics as metrics
from argparse import ArgumentParser

script_dirname = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dirname)
main_dirname = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(main_dirname)

from utils import L2_norm_rv

#########################################################################################

def build_model(pred,resp,model_type,model_params):
    
    ml_model_constr_dict = { "CatBoostRegressor": catboost.CatBoostRegressor,
                             "CatBoostClassifier": catboost.CatBoostClassifier }

    if model_type in ml_model_constr_dict:
        regr = ml_model_constr_dict[model_type](**model_params)
        regr.fit(pred,resp)
    else:
        raise ValueError("This type of models does not exist")

    return regr


#########################################################################################
# model training
#########################################################################################

def main( ):

    parser = ArgumentParser( description = 'Model training.')
    parser.add_argument('-j', '--json',
                        default = None, 
                        help = '[either full path to json or json name if example_tag is provided]')
    parser.add_argument('-s', '--silent',
                        action = "store_true",
                        help = '[either full path to json or json name if example_tag is provided]')
    args = parser.parse_args() # quits if --help is used

    ###########################################
    # loading json
    ###########################################

    silent = args.silent

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

    data_train_filename = script_dirname + '/dataset/data_train.csv'
    data_test_filename  = script_dirname + '/dataset/data_test.csv'	
    
    model_filename = models_folder_name + '/model.dat'
    models_json = models_folder_name + "/models.json"

    
    ###########################################################################################
    # Training
    ###########################################################################################
        
    seeds = json_dict["seeds"]
    np.random.seed(seeds['mod_train'])
    print("\nseed =", seeds['mod_train'])

    # load data:
    if not silent:
        print("\n[loading processed train data]:", data_train_filename )
    df_train = pd.read_csv( data_train_filename )

    if not silent:
        print("\n[loading processed test data]:", data_test_filename )
    df_test = pd.read_csv( data_test_filename )

    pred_names = list(df_train.columns)[:-1]
    X_train = np.array( df_train.iloc[:,:-1], dtype="float" )
    Y_train = np.array( df_train.iloc[:,-1],  dtype="float" )
    X_test  = np.array( df_test.iloc[:,:-1],  dtype="float" )
    Y_test  = np.array( df_test.iloc[:,-1],   dtype="float" )        
    pred_dim = X_train.shape[1]

    # loading models:
    if not silent:
        print("\n[loading models]:", model_filename )
    model_dict = pickle.load(open(model_filename,"rb"))        

    model_type = json_dict["model_params_dict"]['type']
    model_params = json_dict["model_params_dict"]['params']	
    max_ignored_features = json_dict.get("max_ignored_features",None)
    if max_ignored_features is None:
        max_ignored_features = 0
    
    assert model_type in ["CatBoostRegressor", "CatBoostClassifier"]
    assert max_ignored_features>=0
    assert max_ignored_features<=pred_dim
    
    if model_type == "CatBoostClassifier":
        model_type_ = "classifier"
    else:
        model_type_ = "regressor"

    random_state_mlm = np.random.randint(0,999999999)

    model_params.update({"random_state" : random_state_mlm} )
    
    if not silent:
        print("\n\n[Training model]")
        print("\nModel parameters:")
        print("model: {0}, ".format(model_type), model_params)
        print("num_train_samples: {0}, ".format(X_train.shape[0]))
        print("num_test_samples: {0}, ".format(X_test.shape[0]))

    ####################################################
    # build reference model f*(x)=logistic(g_*(x))
    ####################################################

    if not silent:
        print("\n[Training a reference model f*] \n")	

    model = model_dict["model"]

    if model_type_=="classifier":

        Y_pred_prob_train = model.predict_proba(X_train)[:,1] # predictions on train 
        Y_pred_prob_test  = model.predict_proba(X_test)[:,1]  # predictions on test
        # compute classification error:
        auc_train = metrics.roc_auc_score(y_true=Y_train, y_score=Y_pred_prob_train)
        auc_test  = metrics.roc_auc_score(y_true=Y_test,  y_score=Y_pred_prob_test)

        log_loss_train = metrics.log_loss(y_true=Y_train, y_pred = Y_pred_prob_train,normalize=True)
        log_loss_test  = metrics.log_loss(y_true=Y_test,  y_pred = Y_pred_prob_test, normalize=True)
    
        if not silent:
            print("\n[model parameters]\n")	
            print("\t reference model, type {0} :", model_params,"\n")
            print("\n[model performance]\n")	
            print("\n[metrics for reference model, type {0}]:\n".format(model_type))
            print("\t f auc_train ", auc_train )
            print("\t f auc_test ",  auc_test  )
            print("\t f log_loss_train ", log_loss_train )
            print("\t f log_loss_test",   log_loss_test  )

        if not silent:
            print("\n[computing the norm of probabilities scores]\n")			
        L2_norm_score_train = L2_norm_rv(Y_pred_prob_train)
        L2_norm_score_test  = L2_norm_rv(Y_pred_prob_test)
        if not silent:
            print("\tL2-norm (train) of f* {0:.8f}".format(L2_norm_score_train))
            print("\tL2-norm (test) of f* {0:.8f}".format(L2_norm_score_test))		

    Y_pop_min_train = model.predict( data = X_train, prediction_type = "RawFormulaVal")  # predictions on train
    Y_pop_min_test  = model.predict( data = X_test,  prediction_type = "RawFormulaVal")  # predictions on test

    if model_type_=="regressor":
        L2_error_train = L2_norm_rv(Y_pop_min_train-Y_train)
        L2_error_test  = L2_norm_rv(Y_pop_min_test-Y_test)


    if not silent:
        print("\n[computing the norm of population scores]\n")        
    pop_min_mean_train = np.mean(Y_pop_min_train)
    pop_min_mean_test  = np.mean(Y_pop_min_test)
    L2_norm_pop_min_train  = L2_norm_rv(Y_pop_min_train)
    L2_norm_pop_min_test   = L2_norm_rv(Y_pop_min_test)
    if not silent:
        print("\tL2-norm (train) of g_*: {0:.8f}".format(L2_norm_pop_min_train))
        print("\tL2-norm (test) of g_*: {0:.8f}".format(L2_norm_pop_min_test))

    metr_dict = {}
    
    if model_type_=="classifier":

        metr_dict.update( {  "auc_train": float(auc_train),
                    "auc_test" : float(auc_test),
                    "log_loss_train" : float(log_loss_train),
                    "log_loss_test"  : float(log_loss_test),
                    "L2_norm_score_train"   : float(L2_norm_score_train),
                    "L2_norm_score_test"   : float(L2_norm_score_test),
                    } )

    metr_dict.update( { "L2_norm_pop_min_train" : float(L2_norm_pop_min_train),
                        "L2_norm_pop_min_test"  : float(L2_norm_pop_min_test),
                        "L2_dist_train" : [],
                        "L2_dist_test"  : []
                        } )

    if model_type_=="regressor":

        metr_dict.update( {"L2_error_train": float(L2_error_train),
                            "L2_error_test" : float(L2_error_test) } ) 

    ##########################
    # build random models
    ##########################

    if not silent:

        print("\n[Training random models] \n")	

    n_accepted_random_models = json_dict["n_accepted_random_models"]

    assert n_accepted_random_models>=1, "n_accepted_random_models must be >= 1"

    n_iterations_random_models = json_dict["n_iterations_random_models"]

    assert n_iterations_random_models>=n_accepted_random_models, \
        "n_iterations_random_models must be >= n_accepted_random_models"

    rashomon_ball_rel_radius = json_dict["rashomon_ball_rel_radius"]

    if not silent:
        print("\n rashomon_ball_rel_radius =", rashomon_ball_rel_radius)	

    bounds_dict = json_dict["bounds_dict"]
    
    # add ignored features:
    bounds_dict["ignored_features"]=[[0,pred_dim,max_ignored_features], "list"]

    rand_model_list = model_dict["random_models"]
    rand_model_pop_min_mean_train = []
    rand_model_pop_min_mean_test  = []

    for j in range(len(rand_model_list)):

        if not silent:
            print("\n analysis of random model {0}  \n".format(j))	

        model_j = rand_model_list[j]
        
        Y_pop_min_train_j = model_j.predict( data = X_train,  prediction_type = "RawFormulaVal")
        Y_pop_min_test_j = model_j.predict( data = X_test,    prediction_type = "RawFormulaVal")

        pop_min_mean_train_j = np.mean(Y_pop_min_train_j)
        pop_min_mean_test_j  = np.mean(Y_pop_min_test_j)

        L2_dist_pop_min_train_j = L2_norm_rv(Y_pop_min_train_j-Y_pop_min_train)
        L2_dist_pop_min_test_j  = L2_norm_rv(Y_pop_min_test_j-Y_pop_min_test)

        rel_L2_dist_pop_min_train_j = L2_dist_pop_min_train_j/L2_norm_pop_min_train
        rel_L2_dist_pop_min_test_j  = L2_dist_pop_min_test_j/L2_norm_pop_min_test

        if not silent:
            print("\trel-L2-distance (train) of g_*-g_{0}: {1:.8f}".format(j,rel_L2_dist_pop_min_train_j))
            print("\trel-L2-distance (test) of g_*-g_{0}: {1:.8f}".format(j,rel_L2_dist_pop_min_test_j))

        metr_dict["L2_dist_train"].append(L2_dist_pop_min_train_j)
        metr_dict["L2_dist_test"].append(L2_dist_pop_min_test_j)
        rand_model_pop_min_mean_train.append(pop_min_mean_train_j)
        rand_model_pop_min_mean_test.append(pop_min_mean_test_j)

    if not silent: 
        print("\n[saving models_json]:", models_json)

    models_dict = {                
            "model_type"      : model_type,                
            "model_mean_train": pop_min_mean_train,
            "model_mean_test" : pop_min_mean_test,
            "rand_model_mean_train" : rand_model_pop_min_mean_train,
            "rand_model_mean_test"  : rand_model_pop_min_mean_test,
            "model_params"    : model_params,
            "metr_dict"       : metr_dict,
            "pred_names"	  : pred_names,
            "model_filename"  : model_filename,
            "data_train_filename" : data_train_filename,
            "data_test_filename"  : data_test_filename
            }
    
    # save models:
    with open(models_json, "w") as outfile:
        json.dump(models_dict, outfile, indent=4, separators=(',',':'), skipkeys= True )

if __name__=="__main__":
        main()