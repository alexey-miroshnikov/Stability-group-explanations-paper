################################################ 
# Computing marginal shapley under dependencies.
# Exploring the Rashomon effect and instability
# of marginal explanations in $L^2(P_X)$.
################################################ 
# Code authors:
#   Alexey Miroshnikov
#   Konstandinos Kotsiopoulos
# Consultant:
#   Khashayar Filom
################################################
# packages:
#	numpy 1.22.4
#	xgboost 1.7.5
#   scikit-learn 1.2.2
#   shap 0.41.0
#################################################################################
# version 1 (January 2025)
#################################################################################
# To run from command line:
# python instab_data.py --json instab_data.json
################################################################################

import os
import sys
script_dirname  = os.path.dirname(os.path.abspath(__file__))
main_dirname    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(script_dirname)
sys.path.append(main_dirname)

################################################################################
# To run from command line:
# python instab_data.py --json instab_data.json
################################################################################

import sys
import pickle
import os
import numpy as np
import random
import json
import matplotlib.pyplot as plt; plt.style.use('ggplot')
from copy import deepcopy
import sklearn.metrics as metrics

from argparse import ArgumentParser
from libgame import MarginalProbGame
from libshapvalue import DirectQuotShapley
from libowenvalue import DirectOwen
from utils import build_model, L2_norm_rv,L2_norm_vec
from utils import true_response_model,data_generator_model

#########################################################################################
#########################################################################################
## Main
#########################################################################################
#########################################################################################


def main( ):

    # command line parser:
    parser = ArgumentParser( description = \
        'Computing marginal shapley values for data with dependencies.')
    parser.add_argument('-j','--json', 
                        default = None,
                        help = '[either full path to json or json name if example_tag is provided]')
    args = parser.parse_args()

    # creating folders for the results:
    folder_name = script_dirname + '/results'

    if not os.path.exists(folder_name):
            os.mkdir(folder_name)

    # creating a folder for pictures
    pics_folder_name = folder_name + '/pics'
    if not os.path.exists(pics_folder_name):
        os.mkdir(pics_folder_name)

    ###########################################
    # loading json and copy to a "run folder"
    ###########################################

    if args.json is None:
        # default json filename
        filename_json = script_dirname \
            + '/instab_data.json'
    else:
        filename_json = args.json

    assert os.path.exists(filename_json), "{0} does not exist".format(filename_json)
    
    with open(filename_json, 'r') as f:
        json_dict = json.load(f)

    models_filename  = folder_name + '/models.dat'
    shapley_filename = folder_name + '/shapley_explans.dat'

    # pipeline steps:
    pipeline 	   = json_dict["pipeline"]
    step_1_train   = pipeline["step_1_train"]
    step_2_explain = pipeline["step_2_explain"]
    step_3_plot    = pipeline["step_3_plot"]

    progress_bar_status = json_dict["progress_bar_status"]
    
    eps_pred_list = json_dict["eps_pred_list"] # noise level in predictors

    #########################################################################################
    #########################################################################################
    ## STEP 1
    #########################################################################################
    #########################################################################################

    if step_1_train:

        #############################################
        #############################################

        # random seed
        seed = json_dict["seed"]
        if seed is None:
            seed = 123
        np.random.seed(seed)

        model_type    = json_dict["model_params_dict"][0]
        model_params  = json_dict["model_params_dict"][1]

        # generate a list of synthetic data:
        n_sample_train = json_dict["sample_train_size"]
        n_sample_test  = json_dict["sample_test_size"]
       
        ###########################################
        # Model training
        ###########################################

        n_models = len(eps_pred_list)

        model_data_list = [None] * (n_models)

        for k in range(n_models):

            generator = data_generator_model( eps_pred = eps_pred_list[k], gap=1.0 )
            
            X_train,Y_train = generator.dataset_sampler( size = n_sample_train )						
            X_test, Y_test  = generator.dataset_sampler( size = n_sample_test  )		

            random_state_mlm = np.random.randint(0,999999999)

            model_params.update({"random_state" : random_state_mlm} )
            
            print("\n[Training model {0}]".format(k))

            model = build_model( pred = X_train, 
                                 resp = Y_train, 
                                 model_type   = model_type,
                                 model_params = model_params )		

            Y_pred_train = model.predict(X_train)  # predictions on train 
            Y_pred_test  = model.predict(X_test)   # predictions on test 

            # compute error:
            L2_norm_Y_train = L2_norm_rv(Y_train)
            L2_norm_Y_test  = L2_norm_rv(Y_test)

            # compute means:
            Y_pred_mean_train = np.mean(Y_pred_train)
            Y_pred_mean_test  = np.mean(Y_pred_test)
            
            # compute centered norms (deviations) for train dataset:
            L2_norm_centered_train = L2_norm_rv(Y_pred_train- Y_pred_mean_train)
            L2_norm_centered_test  = L2_norm_rv(Y_pred_test - Y_pred_mean_test )
            L2_norm_train = L2_norm_rv(Y_pred_train)
            L2_norm_test  = L2_norm_rv(Y_pred_test)

            # compute errors:
            L2_error_train  = np.sqrt( metrics.mean_squared_error( Y_train, Y_pred_train) ) 
            L2_error_test   = np.sqrt( metrics.mean_squared_error( Y_test,  Y_pred_test)  ) 
            L2_rerror_train = L2_error_train / L2_norm_Y_train				
            L2_rerror_test  = L2_error_test  / L2_norm_Y_test		

            print("\nL2_norm_train:",  L2_norm_train  )
            print("L2_norm_test:",     L2_norm_test   )
            print("L2_error_train:",   L2_error_train )
            print("L2_error_test:",    L2_error_test  )
            print("L2_rerror_train:",  L2_rerror_train)
            print("L2_rerror_test:",   L2_rerror_test )

            metr_dict = {
                    "Y_pred_mean_train"      : Y_pred_mean_train,
                    "L2_norm_train"          : L2_norm_train,
                    "L2_norm_centered_train" : L2_norm_centered_train,
                    "L2_norm_centered_test"  : L2_norm_centered_test,
                    "L2_norm_test"			 : L2_norm_test,
                    "L2_error_train"		 : L2_error_train,
                    "L2_error_test"			 : L2_error_test,
                    "L2_rerror_train"        : L2_rerror_train,
                    "L2_rerror_test"         : L2_rerror_test }

            model_data_list[k] = { "model":      deepcopy(model),
                                   "metr_dict":  deepcopy(metr_dict),
                                   "X_train":    deepcopy(X_train),
                                   "Y_train":    deepcopy(Y_train),
                                   "X_test":     deepcopy(X_test),
                                   "Y_test":     deepcopy(Y_test),
                                   "generator":  deepcopy(generator)
                                   }

        # generate common train mixture dataset:
        dim = model_data_list[0]["X_train"].shape[1]
        XX_train = np.zeros( shape = (n_models, n_sample_train, dim ) )
        YY_train = np.zeros( shape = (n_models, n_sample_train ) )
        for k in range(n_models):
            XX_train[k] = model_data_list[k]["X_train"]
            YY_train[k] = model_data_list[k]["Y_train"]

        # multinomial coin
        probs = [1/n_models]*n_models
        mcoin_train = np.random.multinomial( 1, probs, size = n_sample_train )

        # mixture:
        X_train_mix = np.sum( mcoin_train * XX_train.T, axis = 2).T
        Y_train_mix = np.sum( mcoin_train * YY_train.T, axis = 1).T

        # generate common test mixture datasets:		
        XX_test = np.zeros( shape = (n_models, n_sample_test, dim ) )
        YY_test = np.zeros( shape = (n_models, n_sample_test ) )
        for k in range(n_models):
            XX_test[k] = model_data_list[k]["X_test"]
            YY_test[k] = model_data_list[k]["Y_test"]

        # multinomial coin:		
        mcoin_test = np.random.multinomial( 1, probs, size = n_sample_test )

        # mixture:
        X_test_mix = np.sum( mcoin_test * XX_test.T, axis = 2).T
        Y_test_mix = np.sum( mcoin_test * YY_test.T, axis = 1).T


        # construct a true model:
        model_true_dict = {  "model":      true_response_model(),
                               "metr_dict":  dict(),
                             "X_train":    None,
                             "Y_train":    None,
                             "X_test":     None,
                             "Y_test":     None,
                             "generator":  None
                          }

        # compute errors and differences between model k and 0 on mixture datasets
        Y_pred_train_mix_true_model = model_true_dict["model"].predict(X_train_mix)
        Y_pred_test_mix_true_model  = model_true_dict["model"].predict(X_test_mix)
    
        # append the true model to the end of the list:
        model_data_list.append( deepcopy(model_true_dict))

        for k in range(len(model_data_list)):

            if k<len(model_data_list)-1:
                Y_pred_train_mix  = model_data_list[k]["model"].predict(X_train_mix)
                Y_pred_test_mix   = model_data_list[k]["model"].predict(X_test_mix)
            else:
                Y_pred_train_mix  = model_data_list[k]["model"](X_train_mix)
                Y_pred_test_mix   = model_data_list[k]["model"](X_test_mix)

            L2_norm_train_mix  = L2_norm_rv(Y_pred_train_mix)
            L2_norm_test_mix   = L2_norm_rv(Y_pred_test_mix)

            L2_error_train_mix  = L2_norm_rv(Y_pred_train_mix - Y_train_mix)
            L2_error_test_mix   = L2_norm_rv(Y_pred_test_mix  - Y_test_mix )
            
            L2_diff_train_mix  = L2_norm_rv(Y_pred_train_mix - Y_pred_train_mix_true_model)
            L2_diff_test_mix   = L2_norm_rv(Y_pred_test_mix  - Y_pred_test_mix_true_model)

            metr_dict = model_data_list[k]["metr_dict"]

            metr_dict.update( {"L2_norm_train_mix"   : L2_norm_train_mix} )
            metr_dict.update( {"L2_norm_test_mix"    : L2_norm_test_mix } )
            metr_dict.update( {"L2_error_train_mix"  : L2_error_train_mix} )
            metr_dict.update( {"L2_error_test_mix"   : L2_error_test_mix } )
            metr_dict.update( {"L2_rerror_train_mix" : L2_error_train_mix/L2_norm_train_mix} )
            metr_dict.update( {"L2_rerror_test_mix"  : L2_error_test_mix/L2_norm_test_mix } )
            metr_dict.update( {"L2_diff_train_mix"   : L2_diff_train_mix}  )
            metr_dict.update( {"L2_diff_test_mix"    : L2_diff_test_mix }  )

            print("\n[Errors and differences on the mixture dataset, model {0}]:".format(k))
            print("\nL2_error_train_mix:",  L2_error_train_mix  )
            print("L2_error_test_mix:",     L2_error_test_mix )
            print("L2_diff_train_mix:",     L2_diff_train_mix  )
            print("L2_diff_test_mix:",      L2_diff_test_mix)

            for key in metr_dict:
                metr_dict[key]=float(metr_dict[key])

            with open(folder_name+"/metrics_model_"+str(k)+".json", "w") as outfile:
                json.dump(metr_dict, outfile, indent=4 )

        # get random state:
        stop_seed_state = np.random.get_state()

        # save the model and data:
        data = {
                "stop_seed"         : stop_seed_state[1][0],
                "model_data_list"   : model_data_list,			
                "model_type"        : model_type,
                "X_train_mix"		: X_train_mix,
                "Y_train_mix"		: Y_train_mix,
                "X_test_mix"		: X_test_mix,
                "Y_test_mix"		: Y_test_mix
                }

        print("\n[saving models]:", models_filename)			
        if os.path.exists(models_filename):
            os.remove(models_filename)
        pickle.dump(data,open(models_filename,"wb"))


    #########################################################################################
    #########################################################################################
    ## Step 2: explain
    #########################################################################################
    #########################################################################################

    if step_2_explain:

        #############################################
        # Loading model and data
        #############################################

        group_partition = json_dict["partition"]
        print("\ngroup_partition", group_partition)

        print("\n[loading model]:", models_filename )
        model_dict = pickle.load(open(models_filename,"rb"))

        seed = model_dict["stop_seed"]
        np.random.seed(seed)

        model_type      = model_dict["model_type"]
        model_data_list = model_dict["model_data_list"]

        X_train_mix = model_dict["X_train_mix"]
        Y_train_mix = model_dict["Y_train_mix"]

        # let us use the mixture dataset for both averaging and explanations:

        n_expl = json_dict["expl_size"] # number of explanations to compute
        if n_expl>X_train_mix.shape[0]:
            n_expl = X_train_mix.shape[0]

        n_ave = json_dict["ave_size"]  # number of samples for averaging
        if n_ave>X_train_mix.shape[0]:
            n_ave = X_train_mix.shape[0]

        # random seed for subsampling:
        random_state_ave  = np.random.randint(0,999999999)
        random.seed(random_state_ave)

        # # subsampling the trained mixture dataset to get a background dataset
        idx_subsample_ave = random.sample( range(X_train_mix.shape[0]), n_ave )
        X_ave = X_train_mix[idx_subsample_ave,:]
        
        # subsampling the mixture trained dataset for individuals for which explanations are computed
        idx_subsample_expl = random.sample( range(X_train_mix.shape[0]), n_expl )
        X_expl = X_train_mix[idx_subsample_expl,:]
        Y_expl = Y_train_mix[idx_subsample_expl]

        n_models = len(model_data_list)

        expl_data_list = [None] * n_models

        # iterating in reversed order:
        for k in reversed(range(n_models)):

            print("\n\n*****************************************")
            print("[Computing explanations for model {0}]".format(k))
            print("*****************************************")

            model = model_data_list[k]["model"]			

            # construction of marginal games:			
            vme = MarginalProbGame( func = model.predict, pred = X_ave )
            
            # construction of individual Shapley:		
            print("\n[Computing individual Shapley for model {0}]".format(k))
            mshap_sing      = DirectQuotShapley( game=vme )
            mshap_sing_vals = mshap_sing( X_expl, progress=progress_bar_status, out_format="np.array" )

            # construction of group Shapley:			
            
            print("\n[Computing group Shapley for model {0}]".format(k))
            mshap_group      = DirectQuotShapley( game=vme, index_partition = group_partition )
            # mshap_group      = DirectOwen( game=vme, index_partition = group_partition )
            mshap_group_vals = mshap_group( X_expl, progress=progress_bar_status, out_format="np.array" )

            # predictions on X_expl:
            Y_expl_pred = model.predict(X_expl)

            expl_data_list[k] = { "shapley_group_values" : deepcopy(mshap_group_vals),								  
                                  "shapley_sing_values"  : deepcopy(mshap_sing_vals),		
                                  "Y_expl_pred" : deepcopy(Y_expl_pred) }			
            
             # compute norms of individual Shapley:
            L2_mshap_sing_norm = L2_norm_rv(mshap_sing_vals,axis = 0)
            print("\nL2_mshap_sing_norm {0} :".format(k), L2_mshap_sing_norm)
                
             # compute norms of group Shapley:
            L2_mshap_group_norm = L2_norm_rv(mshap_group_vals,axis = 0)
            
            print("\nL2_mshap_group_norm {0} :".format(k), L2_mshap_group_norm)

            # compute differences between individual Shapley of model k and the true model (n_models-1)
            mshap_sing_vals_ref    = expl_data_list[-1]["shapley_sing_values"]
            L2_mshap_sing_diff     = L2_norm_rv(mshap_sing_vals - mshap_sing_vals_ref, axis = 0)
            L2_mshap_sing_diff_tot = L2_norm_vec(L2_mshap_sing_diff)

            print("\nL2_mshap_sing_diff ({0},0) :".format(k), L2_mshap_sing_diff)
            print("\nL2_mshap_sing_diff_tot ({0},0) :".format(k), L2_mshap_sing_diff_tot)
        
            # compute differences between group Shapley of model k and the true model:
            mshap_group_vals_ref    = expl_data_list[-1]["shapley_group_values"]
            L2_mshap_group_diff     = L2_norm_rv(mshap_group_vals-mshap_group_vals_ref,axis = 0)
            L2_mshap_group_diff_tot = L2_norm_vec(L2_mshap_group_diff)

            print("\nL2_mshap_group_diff ({0},0) :".format(k), L2_mshap_group_diff)
            print("\nL2_mshap_group_diff_tot ({0},0) :".format(k), L2_mshap_group_diff_tot)

            # Compute aggregated sing Shapley differences with group differences:
            L2_aggr_sing_diff_shap = [0]*len(group_partition)
            for j in range(len(group_partition)):			
                Sj=group_partition[j]
                L2_aggr_sing_diff_shap[j] = L2_norm_vec(L2_mshap_sing_diff[Sj])
                print("\n [group {0}] L2_shap_aggr_sing_diff_Sj:".format(j), L2_aggr_sing_diff_shap[j])
                print("\n [group {0}] L2_shap_group_diff:".format(j),  L2_mshap_group_diff[j])
        
            # save errors in a json file:
            shap_metr_dict = {  "L2_mshap_sing_norm"     : list(L2_mshap_sing_norm),
                                "L2_mshap_group_norm"    : list(L2_mshap_group_norm),
                                "L2_mshap_group_norm"    : list(L2_mshap_group_norm),
                                "L2_mshap_sing_diff"     : list(L2_mshap_sing_diff),
                                "L2_mshap_sing_diff_tot" : float(L2_mshap_sing_diff_tot),
                                "L2_mshap_group_diff"    : list(L2_mshap_group_diff),
                                "L2_mshap_group_diff_tot": float(L2_mshap_group_diff_tot),
                                "L2_aggr_sing_diff_shap" : list(L2_aggr_sing_diff_shap) }

            with open(folder_name+"/metrics_shap_"+str(k)+".json", "w") as outfile:
                json.dump(shap_metr_dict, outfile, indent=4)

            expl_data_list[k].update( {"shap_metr_dict" : deepcopy(shap_metr_dict) } )

        # save the data:
        data = {
                "expl_data_list" : expl_data_list,
                "X_expl" : X_expl,
                "Y_expl" : Y_expl,
                "group_partition" : group_partition
                }

        print("\n[saving explanations]:", shapley_filename)			
        if os.path.exists(shapley_filename):
            os.remove(shapley_filename)
        pickle.dump(data,open(shapley_filename,"wb"))

    #########################################################################################
    #########################################################################################
    ## STEP 3
    #########################################################################################
    #########################################################################################


    if step_3_plot:

        if not os.path.exists(models_filename):
            raise ValueError("file {0} does not exist".format(models_filename))
        if not os.path.exists(shapley_filename):
            raise ValueError("file {0} does not exist".format(shapley_filename))

        #############################################	 
        # Loading model and explanations
        #############################################
        
        print("[loading models]:", models_filename )
        model_dict = pickle.load(open(models_filename,"rb"))
        
        print("[loading shapley explanations]:", shapley_filename )
        shapley_dict = pickle.load(open(shapley_filename,"rb"))

        # unpacking model data:
        model_type      = model_dict["model_type"]
        model_data_list = model_dict["model_data_list"]

        # unpacking dictionary for models:
        expl_data_list  = shapley_dict["expl_data_list"]
        X_expl = shapley_dict["X_expl"]
        Y_expl = shapley_dict["Y_expl"]
        group_partition = shapley_dict["group_partition"]

        dim = X_expl.shape[1]
        n_groups = len(group_partition)
        n_models = len(model_data_list)

        ###########################################
        # Set plot parameters
        ###########################################

        cmap_resp   = plt.get_cmap("Greys") # cmap which we will use
        resp_colors = cmap_resp(np.linspace(0.4,1,n_models))

        cmap_shap   = plt.get_cmap("jet")  # cmap for explanations
        shap_colors = cmap_shap(np.linspace(0.0,1,n_models))

        cmap_shap_diff   = plt.get_cmap("plasma") # cmap for differences
        shap_diff_colors = cmap_shap_diff(np.linspace(0.3,1,n_models))

        cmap_shap_diff         = plt.get_cmap("Greens") # cmap for differences
        shap_diff_group_colors = cmap_shap_diff(np.linspace(0.3,1,n_models))

        cmap_shap_aggr_diff = plt.get_cmap("Blues")
        shap_diff_aggr_colors = cmap_shap_aggr_diff(np.linspace(0.3,1,n_models))

        alpha = 0.5	
        alpha_bars = 1
        bar_edgecolor = 'black'
        figsize = (10,8)
        markersize_shap = 5
        markersize_data = 5
        markeredgewidth = 0.5
        ticks_fontsize_bars = 18
        legend_fontsize = 18
        label_fontsize  = 18

        # ###########################################
        # # Plotting
        # ###########################################

        # plot predictions of each model:
        for i in range(dim):
            fig, ax = plt.subplots( figsize = figsize )			
            for k in range(n_models):
                if k<(n_models-1):                    
                    plot_label = "$f_{0}$".format(k+1)
                else:
                    plot_label = "$f_*$"

                plt.plot( X_expl[:,i], expl_data_list[k]["Y_expl_pred"],
                            marker='o',
                            markersize=markersize_data,
                            linestyle='',
                            color = resp_colors[n_models-k-1],
                            label = plot_label,
                            markeredgecolor = 'black',
                            markeredgewidth = markeredgewidth,
                            alpha=alpha,zorder=1)
            plt.xlabel( "$X_{0}$".format(i+1), fontsize=14 )
            plt.ylabel( "Explanations", fontsize=14 )
            plt.legend( fontsize=legend_fontsize, loc='center left', bbox_to_anchor=(1, 0.5))
            plt.tight_layout()
            plt.savefig( fname = pics_folder_name + '/resp_train_versus_X{0}.png'.format(i+1) )
            plt.close()

        # plot explanations of each model:
        for i in range(dim):
            fig, ax = plt.subplots( figsize = figsize )	
            for k in range(n_models):
                if k<(n_models-1):
                    plot_label = r'''$\phi_{0}$'''.format(i+1)+"$(f_{0})$".format(k+1)  
                else:
                    plot_label = r'''$\phi_{0}$'''.format(i+1)+"$(f_*)$"
                plt.plot( X_expl[:,i], expl_data_list[k]["shapley_sing_values"][:,i],
                            marker='o',
                            markersize=markersize_shap,
                            linestyle='',
                            color = shap_colors[k],
                            markeredgecolor='black',
                            markeredgewidth=markeredgewidth,
                            label=plot_label,
                            alpha=alpha, zorder=k)
            plt.xlabel("$X_{0}$".format(i+1), fontsize=label_fontsize)
            plt.ylabel("Explanations", fontsize=label_fontsize)
            plt.legend( fontsize=legend_fontsize, loc='center left', bbox_to_anchor=(1,0.5))
            plt.tight_layout()
            plt.savefig( fname = pics_folder_name + '/shap_train_output_versus_X{0}.png'.format(i+1))			
            plt.close()

        # plot group explanations of each model:
        for j in range(n_groups):			
            Sj = group_partition[j]
            for i in Sj:
                fig, ax = plt.subplots( figsize = figsize )	
                for k in range(n_models):
                    if k<(n_models-1):
                        plot_label = r"$\phi^{\mathcal{P}}_{S_"+ str(j+1) + r"}(f_" + str(k+1) + r")$"
                    else:
                        plot_label = r"$\phi^{\mathcal{P}}_{S_"+ str(j+1) + r"}(f_*)$"
                    plt.plot( X_expl[:,i], expl_data_list[k]["shapley_group_values"][:,j],
                                marker='o',
                                markersize=markersize_shap,
                                linestyle='',
                                color = shap_colors[k],
                                markeredgecolor='black',
                                markeredgewidth=markeredgewidth,
                                label=plot_label,
                                alpha=alpha, zorder=k)
                plt.xlabel("$X_{0}$".format(i+1), fontsize=label_fontsize)
                plt.ylabel("Explanations", fontsize=label_fontsize)
                plt.legend( fontsize=legend_fontsize, loc='center left', bbox_to_anchor=(1, 0.5))
                plt.tight_layout()
                plt.savefig( fname = pics_folder_name + '/shap_group_train_output_versus_X{0}.png'.format(i+1))			
                plt.close()

        # plot differences of individual explanations between model 0 and true model:
        # also find the scale
        expl_min=[None]*dim
        expl_max=[None]*dim
        for i in range(dim):				            
            fig, ax = plt.subplots( figsize = figsize )			
            for k in range(n_models-1): # true model is stored last
                if k==0:
                    expl_min[i]=np.min(expl_data_list[k]["shapley_sing_values"][:,i]-expl_data_list[-1]["shapley_sing_values"][:,i])
                    expl_max[i]=np.max(expl_data_list[k]["shapley_sing_values"][:,i]-expl_data_list[-1]["shapley_sing_values"][:,i])
                else:
                    expl_min_ = np.min(expl_data_list[k]["shapley_sing_values"][:,i]-expl_data_list[-1]["shapley_sing_values"][:,i])
                    expl_max_ = np.max(expl_data_list[k]["shapley_sing_values"][:,i]-expl_data_list[-1]["shapley_sing_values"][:,i])
                    if expl_min_<expl_min[i]: expl_min[i]=expl_min_
                    if expl_max_>expl_max[i]: expl_max[i]=expl_max_
                plt.plot( X_expl[:,i],
                        expl_data_list[k]["shapley_sing_values"][:,i]-expl_data_list[-1]["shapley_sing_values"][:,i],
                        marker='o',
                        markersize=markersize_shap-1,
                        linestyle='',
                        color = shap_colors[k],
                        markeredgecolor='black',
                        markeredgewidth=markeredgewidth,
                        label = r'''$\phi_{0}(\Delta f_{1})$'''.format(i+1,k+1),
                        alpha = 0.7, zorder=k+1)

            for k in range(n_models-1): # true model is stored last
                plt.plot( X_expl[:,i], expl_data_list[k]["Y_expl_pred"]-expl_data_list[-1]["Y_expl_pred"],
                            marker='o',
                            markersize=markersize_data-2,
                            linestyle='',
                            color = resp_colors[n_models-k-1],
                            label = r'''$\Delta f_{0}$'''.format(k+1),
                            markeredgecolor='black',
                            markeredgewidth=markeredgewidth,
                            alpha=0.7,zorder=n_models+k)

            plt.xlabel("$X_{0}$".format(i+1), fontsize=label_fontsize)
            plt.ylabel("Explanations", fontsize=label_fontsize)
            plt.legend( fontsize=legend_fontsize, loc='center left', bbox_to_anchor=(1,0.5))
            plt.tight_layout()				
            plt.savefig( fname = pics_folder_name + '/diff_shap_train_output_versus_X{1}.png'.format(k+1,i+1) )
            plt.close()

        # plot differences of responses between model 0 and true model:
        for i in range(dim):				
            fig, ax = plt.subplots( figsize = figsize )			
            for k in range(n_models-1): # true model is stored last
                plt.plot( X_expl[:,i], expl_data_list[k]["Y_expl_pred"]-expl_data_list[-1]["Y_expl_pred"],
                            marker='o',
                            markersize=markersize_data-1,
                            linestyle='',
                            color = resp_colors[n_models-k-1],
                            label = r'''$\Delta f_{0}$'''.format(k+1),
                            markeredgecolor='black',
                            markeredgewidth=markeredgewidth,
                            alpha=1,zorder=n_models+k)
            plt.ylim((expl_min[i],expl_max[i]))
            plt.xlabel("$X_{0}$".format(i+1), fontsize=label_fontsize)
            plt.ylabel("Explanations", fontsize=label_fontsize)
            plt.legend( fontsize=legend_fontsize, loc='center left', bbox_to_anchor=(1,0.5))
            plt.tight_layout()				
            plt.savefig( fname = pics_folder_name + '/diff_resp_train_output_versus_X{1}.png'.format(k+1,i+1) )
            plt.close()

        # plot individual shapley values norms for each model: 
        fig, ax   = plt.subplots( figsize = figsize )
        position  = 1.5*np.arange(dim)
        bar_width = 1/n_models
        norm_labels = [""]*dim
        for i in range(dim):
            norm_labels[i] = r'''$\phi_{0}$'''.format(i+1)
            for k in range(n_models):
                if i==0: 
                    if k<(n_models-1):
                        bar_label = r'''$\phi(f_{0})$'''.format(k+1)
                    else:
                        bar_label = r'''$\phi(f_*)$'''
                else:    
                    bar_label = None

                plt.bar( position+(k-((n_models-1)/2))*bar_width,
                        expl_data_list[k]["shap_metr_dict"]["L2_mshap_sing_norm"],
                        color=shap_colors[k], 
                        alpha = alpha_bars, 
                        width = bar_width,
                        align = 'center',
                        label = bar_label,
                        edgecolor = bar_edgecolor)
        plt.legend( fontsize=legend_fontsize, loc='center left', bbox_to_anchor=(1,0.5))		
        plt.xticks( position, norm_labels, fontsize=ticks_fontsize_bars )
        plt.ylabel( ylabel = r'''$L^2(\mathbb{P})$-norm''', fontsize=legend_fontsize)
        plt.tight_layout()				
        plt.savefig( fname = pics_folder_name + '/shap_norms_train.png')		
        plt.close()

        # plot individual shapley values norms for each model: 
        fig, ax   = plt.subplots( figsize = figsize )
        position  = 1.5*np.arange(n_groups)
        bar_width = 1/n_models
        norm_labels = [""]*n_groups
        for j in range(n_groups):
            norm_labels[j] = r'''$\phi_{0}$'''.format(j+1)
            for k in range(n_models):
                if j==0: 
                    if k<(n_models-1):
                        bar_label = r'''$\phi(f_{0})$'''.format(k+1)
                    else:
                        bar_label = r'''$\phi(f_*)$'''
                else:    
                    bar_label = None

                plt.bar( position+(k-((n_models-1)/2))*bar_width,
                        expl_data_list[k]["shap_metr_dict"]["L2_mshap_group_norm"],
                        color=shap_colors[k], 
                        alpha = alpha_bars, 
                        width = bar_width,
                        align = 'center',
                        label = bar_label,
                        edgecolor = bar_edgecolor)
        plt.legend( fontsize=legend_fontsize, loc='center left', bbox_to_anchor=(1,0.5))		
        plt.xticks( position, norm_labels, fontsize=ticks_fontsize_bars )
        plt.ylabel( ylabel = r'''$L^2(\mathbb{P})$-norm''', fontsize=legend_fontsize)
        plt.xlim((position[0]-n_models * bar_width , position[-1]+n_models * bar_width))
        plt.tight_layout()				
        plt.savefig( fname = pics_folder_name + '/shap_norms_group_train.png')		
        plt.close()


        # plot differences of individual shapley values between modek k and true model (n_models-1): 
        fig, ax   = plt.subplots( figsize = figsize )
        scaler    = 1.5
        position  = scaler*np.arange(dim+1)
        bar_width = 1/(n_models-1)
        norm_labels = [""]*(dim+1)
        for i in range(dim):
            norm_labels[i] = r'''$\Delta \phi_{0}$'''.format(i+1)
            for k in range(n_models-1):

                if i==0: 
                    bar_label = r'''$\phi(\Delta f_{0})$'''.format(k+1)
                else:    
                    bar_label = None

                plt.bar( position[0:-1]+(k-((n_models-1)/2))*bar_width,
                         expl_data_list[k]["shap_metr_dict"]["L2_mshap_sing_diff"],
                         color=shap_diff_colors[k], 
                         alpha = alpha_bars, 
                         width = bar_width,
                         align = 'center',
                          label = bar_label,
                         edgecolor = bar_edgecolor)		
        norm_labels[dim] = r'''$\Delta f$'''
        for k in range(n_models-1):
            plt.bar( position[-1]+(k-((n_models-1)/2))*bar_width,
                     model_data_list[k]["metr_dict"]["L2_diff_train_mix"],
                     color = resp_colors[n_models-k-1], 
                     alpha = alpha_bars, 
                     width = bar_width,
                     align = 'center',
                     label = r'''$\Delta f_{0}$'''.format(k+1),
                     edgecolor = bar_edgecolor)
        plt.legend( fontsize=legend_fontsize, loc='center left', bbox_to_anchor=(1,0.5))		
        plt.xticks( position, norm_labels, fontsize=ticks_fontsize_bars )
        plt.ylabel( ylabel = r'''$L^2(\mathbb{P})$-norm''', fontsize=legend_fontsize)
        plt.tight_layout()				
        plt.savefig( fname = pics_folder_name + '/shap_diff_train.png')
        plt.close()

        # plot differences of group shapley values between modek k and true model (n_models-1): 
        fig, ax   = plt.subplots( figsize = figsize )
        scaler    = 1.5
        position  = scaler*np.arange(n_groups+1)
        bar_width = 1/(n_models-1)
        norm_labels = [""]*(n_groups+1)
        for j in range(n_groups):
            norm_labels[j] = r"$\Delta \phi^{\mathcal{P}}_{S_"+ str(j+1) + r"}$"

            for k in range(n_models-1):

                if j==0: 
                    bar_label = r'''$\phi^{\mathcal{P}}$''' + r'''$(\Delta f_{0})$'''.format(k+1)
                else:    
                    bar_label = None

                plt.bar( position[0:-1]+(k-((n_models-1)/2))*bar_width,
                         expl_data_list[k]["shap_metr_dict"]["L2_mshap_group_diff"],
                         color=shap_diff_colors[k], 
                         alpha = alpha_bars, 
                         width = bar_width,
                         align = 'center',
                          label = bar_label,
                         edgecolor = bar_edgecolor)		
        norm_labels[n_groups] = r'''$\Delta f$'''
        for k in range(n_models-1):
            plt.bar( position[-1]+(k-((n_models-1)/2))*bar_width,
                     model_data_list[k]["metr_dict"]["L2_diff_train_mix"],
                     color = resp_colors[n_models-k-1], 
                     alpha = alpha_bars, 
                     width = bar_width,
                     align = 'center',
                     label = r'''$\Delta f_{0}$'''.format(k+1),
                     edgecolor = bar_edgecolor)
        plt.legend( fontsize=legend_fontsize, loc='center left', bbox_to_anchor=(1,0.5))		
        plt.xticks( position, norm_labels, fontsize=ticks_fontsize_bars )
        plt.ylabel( ylabel = r'''$L^2(\mathbb{P})$-norm''', fontsize=label_fontsize)
        plt.tight_layout()				
        plt.savefig( fname = pics_folder_name + '/shap_diff_group_train.png')
        plt.close()


        # plot differences of group shapley values between modek k and true model (n_models-1): 
        fig, ax   = plt.subplots( figsize = figsize )
        scaler    = 1.5
        position  = scaler*np.arange(n_groups+1)
        bar_width = 1/(2*(n_models-1))
        group_labels = [""]*(n_groups+1)

        group_labels = [""]*(n_groups+1)
        for j in range(n_groups):	
            index_set = str(list(np.array(group_partition[j])+1))
            index_set = index_set[1:-1]
            index_set = r"\{" + index_set + r"\}"
            group_labels[j] = r"$X_{S_" + str(j+1) +r"}=" + r"X_{" + r"{0}".format(index_set) + r"}$"
        group_labels[n_groups] = r"$\Delta f$"

        for j in range(n_groups):
            norm_labels[j] = r"$\Delta \phi_{0}$".format(j+1)+r"$[v_{ME}^{\mathcal{P}}]$"

            # plot aggregetad values:
            for k in range(n_models-1):
                if j==0: 
                    bar_label_aggr  = r"$|\phi_{S_j}(\Delta f_" + str(k+1) +r")|$"
                else:    
                    bar_label_aggr  = None

                plt.bar( position[0:-1]+(k-(n_models-1))*bar_width,
                         expl_data_list[k]["shap_metr_dict"]["L2_aggr_sing_diff_shap"],
                         color = shap_diff_aggr_colors[k], 
                         alpha = alpha_bars, 
                         width = bar_width,
                         align = 'center',
                          label = bar_label_aggr,
                         edgecolor = bar_edgecolor)
            
            # plot quotient values:
            for k in range(n_models-1):

                if j==0: 
                    bar_label_group = r"$\phi^{\mathcal{P}}_{S_j}(\Delta f_" + str(k+1) + ")$"
                else:    
                    bar_label_group = None

                plt.bar( position[0:-1]+k*bar_width,
                         expl_data_list[k]["shap_metr_dict"]["L2_mshap_group_diff"],
                         color = shap_diff_group_colors[k], 
                         alpha = alpha_bars, 
                         width = bar_width,
                         align = 'center',
                          label = bar_label_group,
                         edgecolor = bar_edgecolor)

        # norm of model difference	
        for k in range(n_models-1):
            plt.bar( position[-1]+(k-((n_models-1)))*bar_width,
                        model_data_list[k]["metr_dict"]["L2_diff_train_mix"],
                        color = resp_colors[n_models-k-1], 
                        alpha = alpha_bars, 
                        width = bar_width,
                        align = 'center',					
                        label = None,
                        edgecolor = bar_edgecolor)

        plt.legend( fontsize=legend_fontsize, loc='center left', bbox_to_anchor=(1,0.5))		
        plt.xticks( position, group_labels, fontsize=ticks_fontsize_bars )
        plt.ylabel( ylabel = r"$L^2(\mathbb{P})$-norm", fontsize=label_fontsize)
        plt.tight_layout()				
        plt.savefig( fname = pics_folder_name + '/shap_gain_stab_train_abs.png')
        plt.close()

        # plot differences of group shapley values between modek k and true model (n_models-1): 
        fig, ax   = plt.subplots( figsize = figsize )
        
        bar_width = 1/(2*(n_models-1))
        scaler    = 1.5
        position_clust = scaler*np.arange(n_groups+1)		
        position = np.zeros( shape=(2*n_groups+1,) )
        
        # indiv shap positions:
        position[0:n_groups] = position_clust[0:n_groups] - (n_models/2+2)*bar_width
        # quot shap positions:
        position[n_groups:(2*n_groups)] = position_clust[0:n_groups] + 1 * bar_width
        # model positions:
        position[-1] = position_clust[-1] - (n_models/2+2)*bar_width

        
        group_labels = [""]*(2*n_groups+1)
        for j in range(n_groups):
            group_labels[j] = r"$|\Delta \phi_{S_" + str(j+1) + r"}|$"
            group_labels[n_groups+j] = r"$|\Delta \phi^{\mathcal{P}}_{S_" + str(j+1) + r"}|$"
        group_labels[-1] = r"$\Delta f$"

            
        # plot aggregetad values:
        for k in range(n_models-1):

            plt.bar( position[0:n_groups]+k*bar_width,
                        expl_data_list[k]["shap_metr_dict"]["L2_aggr_sing_diff_shap"],
                        color = shap_diff_aggr_colors[k], 
                        alpha = alpha_bars, 
                        width = bar_width,
                        align = 'center',
                        label = r"$|\phi_{S_j}(\Delta f_" + str(k+1) +r")|$",
                        edgecolor = bar_edgecolor)
        
        # plot quotient values:
        for k in range(n_models-1):

            plt.bar( position[n_groups:(2*n_groups)] + k * bar_width,
                        expl_data_list[k]["shap_metr_dict"]["L2_mshap_group_diff"],
                        color = shap_diff_group_colors[k], 
                        alpha = alpha_bars, 
                        width = bar_width,
                        align = 'center',
                        label = r"$\phi^{\mathcal{P}}_{S_j}(\Delta f_" + str(k+1) + ")$",
                        edgecolor = bar_edgecolor)

        # plot norm of model difference	
        for k in range(n_models-1):
            plt.bar( position[-1] + k * bar_width,
                        model_data_list[k]["metr_dict"]["L2_diff_train_mix"],
                        color = resp_colors[n_models-k-1], 
                        alpha = alpha_bars, 
                        width = bar_width,
                        align = 'center',					
                        label = None, 
                        edgecolor = bar_edgecolor)

        plt.legend( fontsize=legend_fontsize, loc='center left', bbox_to_anchor=(1,0.5))		
        plt.xticks( position+(n_models/2-1)*bar_width, group_labels, fontsize=ticks_fontsize_bars )
        plt.ylabel( ylabel = r"$L^2(\mathbb{P})$-norm", fontsize=label_fontsize)
        plt.tight_layout()				
        plt.savefig( fname = pics_folder_name + '/shap_gain_stab_train_abs_2.png')
        plt.close()

        # plot differences of group shapley values between modek k and true model (n_models-1): 
        fig, ax   = plt.subplots( figsize = figsize )
        scaler    = 1.5
        position  = scaler*np.arange(n_groups)
        bar_width = 1/(2*(n_models-1))
        group_labels = [""]*(n_groups)

        group_labels = [""]*(n_groups)
        for j in range(n_groups):			
            index_set = str(list(np.array(group_partition[j])+1))
            index_set = index_set[1:-1]
            index_set = r"\{" + index_set + r"\}"
            group_labels[j] = r"$X_{S_" + str(j+1) +r"}=" + r"X_{" + r"{0}".format(index_set) + r"}$"

        for j in range(n_groups):
            norm_labels[j] = r"$\Delta \phi_{0}$".format(j+1)+r"$[v_{ME}^{\mathcal{P}}]$"

            # plot aggregetad values:
            for k in range(n_models-1):
                if j==0: 
                    bar_label_aggr  = r"$|\phi_{S_j}(\Delta f_" + str(k+1) +r")|$"
                else:    
                    bar_label_aggr  = None

                plt.bar( position+(k-(n_models-1))*bar_width,
                         expl_data_list[k]["shap_metr_dict"]["L2_aggr_sing_diff_shap"],
                         color = shap_diff_aggr_colors[k],
                         alpha = alpha_bars, 
                         width = bar_width,
                         align = 'center',
                          label = bar_label_aggr,
                         edgecolor = bar_edgecolor)
            
            # plot quotient values:
            for k in range(n_models-1):

                if j==0: 
                    bar_label_group = r"$\phi^{\mathcal{P}}_{S_j}(\Delta f_" + str(k+1) + r")$"
                else:    
                    bar_label_group = None

                plt.bar( position+k*bar_width,
                         expl_data_list[k]["shap_metr_dict"]["L2_mshap_group_diff"],
                         color = shap_diff_group_colors[k],
                         alpha = alpha_bars, 
                         width = bar_width,
                         align = 'center',
                          label = bar_label_group,
                         edgecolor = bar_edgecolor)


        plt.legend( fontsize=legend_fontsize, loc='center left', bbox_to_anchor=(1,0.5))		
        plt.xticks( position, group_labels, fontsize=ticks_fontsize_bars )
        plt.ylabel( ylabel = r"$L^2(\mathbb{P})$-norm", fontsize=label_fontsize)
        plt.tight_layout()				
        plt.savefig( fname = pics_folder_name + '/shap_gain_stab_train_abs_3.png')
        plt.close()

        # plot differences of group shapley values between modek k and true model (n_models-1): 
        fig, ax   = plt.subplots( figsize = figsize )
        scaler    = 1.5
        position  = scaler*np.arange(3)
        bar_width = 1/(n_models-1)
        norm_labels = [""]*(3)

        norm_labels[0] = r"$|\Delta \phi|$"
        norm_labels[1] = r"$|\Delta \phi^{\mathcal{P}}|$"
        norm_labels[2] = r"$\Delta f$"

        
        for k in range(n_models-1):
            
            # total norm of indiidual shapley 
            plt.bar( position[0]+(k-((n_models-1)/2))*bar_width,
                        expl_data_list[k]["shap_metr_dict"]["L2_mshap_sing_diff_tot"],
                        color = shap_diff_aggr_colors[k], 
                        alpha = alpha_bars, 
                        width = bar_width,
                        align = 'center',
                        label = r"$\phi$" + r"$(\Delta f_{0})$".format(k+1),
                        edgecolor = bar_edgecolor)

        for k in range(n_models-1):
            
            # total norm of quotient shapley
            plt.bar( position[1]+(k-((n_models-1)/2))*bar_width,
                    expl_data_list[k]["shap_metr_dict"]["L2_mshap_group_diff_tot"],
                    color = shap_diff_group_colors[k], 
                    alpha = alpha_bars, 
                    width = bar_width,
                    align = 'center',
                    label = r"$\phi^{\mathcal{P}}$" + r"$(\Delta f_{0})$".format(k+1),
                    edgecolor = bar_edgecolor)

        # norm of model difference	
        for k in range(n_models-1):
            plt.bar( position[-1]+(k-((n_models-1)/2))*bar_width,
                        model_data_list[k]["metr_dict"]["L2_diff_train_mix"],
                        color = resp_colors[n_models-k-1], 
                        alpha = alpha_bars, 
                        width = bar_width,
                        align = 'center',					
                        label = None,
                        edgecolor = bar_edgecolor)
        
        plt.legend( fontsize=legend_fontsize, loc='center left', bbox_to_anchor=(1,0.5))		
        plt.xticks( position, norm_labels, fontsize=ticks_fontsize_bars )
        plt.ylabel( ylabel = r"$L^2(\mathbb{P})$-norm", fontsize=label_fontsize)
        plt.tight_layout()				
        plt.savefig( fname = pics_folder_name + '/shap_gain_tot_stab_train_abs.png')
        plt.close()
      

if __name__=="__main__":
        main()
