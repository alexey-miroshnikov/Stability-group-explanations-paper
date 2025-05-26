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
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt; plt.style.use('ggplot')
from scipy.cluster.hierarchy import dendrogram
from argparse import ArgumentParser

script_dirname = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dirname)
main_dirname = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(main_dirname)

#########################################################################################
## plotting results
#########################################################################################

def main( ):

    
    parser = ArgumentParser( description = 'Plotting results.')
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

    pics_folder_name = script_dirname + "/pictures"
    if not os.path.exists(pics_folder_name):
        os.mkdir(pics_folder_name)

    partitions_json  = script_dirname + "/partitions/partitions.json"
    linkage_filename = script_dirname + "/partitions/linkage.csv"

    models_json = models_folder_name + "/models.json"
    explanations_json = explan_folder_name + "/explanations.json"
    explain_analysis_json = explan_folder_name + "/explain_analysis.json"

    ###########################################################################################
    # plotting
    ###########################################################################################

    # load partitions:
    with open(partitions_json, 'r') as f:
        partitions_dict = json.load(f)

    # load seed from the previous step:
    with open(models_json, 'r') as f:
        models_dict = json.load(f)

    # load explanations:
    with open(explanations_json, 'r') as f:
        explanations_dict = json.load(f)

    # load explanations analysis stability:
    with open(explain_analysis_json, 'r') as f:
        explain_analysis_dict = json.load(f)

    # extract partition info:
    if not silent:
        print("\n[loading clustering tree]:", partitions_json )
    pt = partitions_dict['partition_dict']
    partition_tree_list = [ [pt[str(i)][2],pt[str(i)][0]] for i in range(len(pt)) ]
    heights = partitions_dict['coales_heights']
    pred_labels = partitions_dict["labels"]        
    linkage = np.array( pd.read_csv(linkage_filename, header=None))

    # extract model info:
    metr_dict  = models_dict["metr_dict"]
    pred_names  = models_dict["pred_names"]
    L2_dist_models_train = metr_dict["L2_dist_train"]
    dim = len(pred_names)


    # extract explanations info:
    partition_list_expl = explanations_dict["partitions"]
    thresholds_expl     = explanations_dict["thresholds"]
    thresholds_idx_plot = json_dict["thresholds_idx_plot"]
    if thresholds_idx_plot is None:
        thresholds_idx_plot = [i for i in range(len(thresholds_expl))]
    n_partitions_expl   = len(partition_list_expl)

    # create labels for all explained partitions:
    partition_labels_expl = [None] * n_partitions_expl
    precision = 2
    for m in range(n_partitions_expl):           
        threshold_label   = "{0:.2f}".format( np.ceil(thresholds_expl[m]*(10**precision))/(10**precision) )            
        partition_labels_expl[m] = r'''$\mathcal{P}$''' + r'$_{' + threshold_label + r'}$'
    
    # diff-global values:
    max_diff_glob_shap_expl_tot=\
        explain_analysis_dict["max_diff_glob_shap_expl_tot"]
    max_diff_glob_group_shap_expl_tot=\
        explain_analysis_dict["max_diff_glob_group_shap_expl_tot"]
    max_diff_glob_shap_expl_nonsingl_tot=\
        explain_analysis_dict["max_diff_glob_shap_expl_nonsingl_tot"]
    max_diff_glob_group_shap_expl_nonsingl_tot=\
        explain_analysis_dict["max_diff_glob_group_shap_expl_nonsingl_tot"]

    # plotting:
    bar_edgecolor = 'black'
    figsize = (8,7)
    ticks_fontsize  = 14
    label_fontsize  = 16

    n_leaves = len(partition_tree_list) 


    pred_labels_adj = [""] * len(pred_labels)
    for i in range(len(pred_labels_adj)):
        pred_labels_adj[i] = r"$X_{" + r"{0}".format(i+1) + r"}$ = " + str(pred_labels[i])               

    if dim<=25:
        leaf_font_size = 14
    elif dim<=50:
        leaf_font_size = 12
    else:
        leaf_font_size = 10
    c = 1.5
    
    # plotting dendrogram:
    plt.figure(figsize=(20,12))
    dendrogram(linkage,labels=pred_labels_adj, orientation='right', leaf_font_size=leaf_font_size)
    plt.xlabel('1 - MIC$_e$', fontsize=label_fontsize * c)
    plt.ylabel('features', fontsize = label_fontsize * c)
    plt.xticks(fontsize=ticks_fontsize * c )
    plt.title('MIC$_e$ Hierarchical Clustering', fontsize = label_fontsize * c)
    plt.tight_layout()					
    plt.savefig(fname = pics_folder_name+'/hierclust_MIC_average.png')
    plt.close()

    # plotting dedrogram:
    plt.figure(figsize=(20,12))
    dendrogram(linkage,labels=pred_labels_adj, orientation='right', leaf_font_size=leaf_font_size)
    plt.xlabel('1 - MIC$_e$', fontsize=label_fontsize * c)
    plt.ylabel('features', fontsize = label_fontsize * c)
    plt.xticks(fontsize = ticks_fontsize * c )        
    plt.title('MIC$_e$ Hierarchical Clustering', fontsize = label_fontsize * c)
    for idx in thresholds_idx_plot:                
        plt.plot( [thresholds_expl[idx],thresholds_expl[idx]], [0,n_leaves*10],
                    color='black',
                    marker='o',
                    markersize=6,
                    markeredgewidth=0.5,                        
                    linestyle="solid", alpha=0.5) 
    plt.tight_layout()
    plt.savefig(fname = pics_folder_name+'/hierclust_MIC_average_.png')
    plt.close()
    
    plt.figure(figsize=figsize)
    heights_ = heights[1:]
    position = [i for i in range(len(heights_))]		
    height_labels = [ str(i + n_leaves) for i in range(len(heights_))]		
    bar_width = 1/2
    plt.bar( position,
            heights_,
            color="blue", 
            alpha = 0.5, 
            width = bar_width,
            label = height_labels,
            align = 'center',				
            edgecolor = "black")
    plt.xticks( position, height_labels, fontsize=ticks_fontsize )				
    plt.tight_layout()					
    plt.savefig(fname = pics_folder_name+'/hierclust_heights.png')
    plt.close()

    ###############################################################
    # plot differences of total individual versus total 
    # group explanations for a list of models (including singletons):
    ###############################################################

    n_partitions_plot = len(thresholds_idx_plot)
    diff_ticks_labels = [None] * (n_partitions_plot+1)
    for m in range(n_partitions_plot):
        diff_ticks_labels[m] = partition_labels_expl[thresholds_idx_plot[m]]
    
    diff_ticks_labels[-1] = r'''$\max_k \|\Delta f_k\|$'''

    fig, ax   = plt.subplots( figsize = figsize )
    position  = 2.0 * np.arange(n_partitions_plot+1)
    bar_width = 1/2

    plt.bar( position[:-1]-(1/2)*bar_width,
            max_diff_glob_shap_expl_tot,
            # np.array(max_diff_glob_coal_expl_tot)[thresholds_idx_plot],
            color = "magenta",
            alpha = 1, 
            width = bar_width,
            label = r'''$|Shap(\Delta f)|$''',
            edgecolor = bar_edgecolor)

    plt.bar( position[:-1]+(1/2)*bar_width,
            np.array(max_diff_glob_group_shap_expl_tot)[thresholds_idx_plot],
            color = "lime",
            alpha = 1, 
            width = bar_width,
            label = r'''$|Shap^{\mathcal{P}}(\Delta f)|$''',
            edgecolor = bar_edgecolor)

    plt.bar( position[-1],
            np.max(L2_dist_models_train),
            color = "gray",
            alpha = 1,
            width = bar_width,
            label = r"$\Delta f$",
            edgecolor = bar_edgecolor)

    plt.legend( fontsize=14)		
    plt.xticks( position, diff_ticks_labels, fontsize=ticks_fontsize, rotation=0 )
    plt.ylabel( ylabel = r"$L^2(\mathbb{P})$-norm", fontsize=label_fontsize)        
    plt.tight_layout()            
    plt.savefig( fname = pics_folder_name + '/diff_expl_tot_model.png')		
    plt.close()		

    ###############################################################
    # plot differences of total individual versus total 
    # group explanations for a list of models (excluding singletons)
    ###############################################################

    n_partitions_plot = len(thresholds_idx_plot)
    diff_ticks_labels = [None] * (n_partitions_plot+1)
    for m in range(n_partitions_plot):
        diff_ticks_labels[m] = partition_labels_expl[thresholds_idx_plot[m]]

    diff_ticks_labels[-1] = r'''$\max_k \|\Delta f_k\|$'''

    fig, ax   = plt.subplots( figsize = figsize )
    position  = 2.0 * np.arange(n_partitions_plot+1)
    bar_width = 1/2

    plt.bar( position[:-1]-(1/2)*bar_width,
            np.array(max_diff_glob_shap_expl_nonsingl_tot)[thresholds_idx_plot],
            color = "magenta",
            alpha = 1, 
            width = bar_width,
            label = r'''$|Shap(\Delta f)|$''',
            edgecolor = bar_edgecolor)

    plt.bar( position[:-1]+(1/2)*bar_width,
            np.array(max_diff_glob_group_shap_expl_nonsingl_tot)[thresholds_idx_plot],
            color = "lime",
            alpha = 1, 
            width = bar_width,
            label = r'''$|Shap^{\mathcal{P}}(\Delta f)|$''',
            edgecolor = bar_edgecolor)

    plt.bar( position[-1],
            np.max(L2_dist_models_train),
            color = "gray",
            alpha = 1,
            width = bar_width,
            label = r"$\Delta f$",
            edgecolor = bar_edgecolor)

    plt.legend( fontsize=14)		
    plt.xticks( position, diff_ticks_labels, fontsize=ticks_fontsize, rotation=0 )
    plt.ylabel( ylabel = r"$L^2(\mathbb{P})$-norm", fontsize=label_fontsize)
    plt.tight_layout()            
    plt.savefig( fname = pics_folder_name + '/diff_expl_tot_nonsingl_model.png')
    plt.close()		


    ############################################################### 
    # plotting explanation norms for ranking analysis
    ###############################################################

    # ranking

    heights = partitions_dict['coales_heights']
    pt      = partitions_dict['partition_dict']
    pred_labels = partitions_dict["labels"]

    if True:

        heights_expl = \
            [ heights[partition_list_expl[j][1]] for j in range(len(partition_list_expl)) ]
        partition_order_idx_list_expl = [None] * n_partitions_expl
        partition_labels_expl = [None] * n_partitions_expl
        precision = 2
        for m in range(n_partitions_expl):
            partition = partition_list_expl[m][0]
            idx       = partition_list_expl[m][1]
            # combine all lists together
            partition_order_idx_list_expl[m] = sum(partition,[]) 
            partition_label   = r'''$\mathcal{P}$''' + r'$^{(' + str(idx) + r')}$'
            threshold_label   = "{0:.2f}".format( np.ceil(heights_expl[m]*(10**precision))/(10**precision) )
            partition_label_h = r'''$\mathcal{P}$''' + r'$_{' + threshold_label + r'}$'
            partition_labels_expl[m] = partition_label + "=" + partition_label_h


    pred_names  = models_dict["pred_names"]
    dim = len(pred_names)

    model_info = json_dict["model_params_dict"]["names"]

    if not silent:
        print("\n[loading clustering tree linkage]:", linkage_filename)

    linkage = np.array( pd.read_csv( linkage_filename, header=None ) )
    
    L2_norm_shap_expl     = explain_analysis_dict["L2_norm_shap_expl"]
    L2_norm_shap_expl_tot = explain_analysis_dict["L2_norm_shap_expl_tot"]    
    L2_norm_group_shap_expl = explain_analysis_dict["L2_norm_group_shap_expl"]

    L2_norm_pop_min = models_dict["metr_dict"]["L2_norm_pop_min_train"]

    # plot parameters:
    bar_edgecolor = 'black'
    figsize = (8,7)
    ticks_fontsize  = 16
    label_fontsize  = 16
    
    n_cols  = 2
    cmap_explan   = plt.get_cmap("jet")  # cmap for explanations
    explan_colors = cmap_explan(np.linspace(0.3,0.7,n_cols))

    cmap_explan_group    = plt.get_cmap("BuGn")  # cmap for explanations
    explan_group_colors  = cmap_explan_group(np.linspace(0.3,0.7,n_cols))

    ticks_labels_expl = []        
    for m in range(n_partitions_expl):            
        idx = partition_list_expl[m][1]            
        ticks_labels_expl.append(r'$\mathcal{P}_{' + str(idx) + r'}$')
    ticks_labels_expl.append("")            
    ticks_labels_expl[-1] = r"$\^f$"

    model_name = model_info["model_name"]

    ############################################################### 
    # plots
    ###############################################################

    # plot the norms for explained partitions:

    fig, ax = plt.subplots( figsize = (10,7))
    
    position  = 2.0 * np.arange(n_partitions_expl+1)
    
    bar_width = 1/2          
            
    expl_label = r"$||Shap(\^f)||$"
    
    f_label    = r"$||\^f||$"

    plt.bar( position[:-1],                                    
            np.array([L2_norm_shap_expl_tot]*n_partitions_plot),
            color = explan_colors[0],
            alpha = 1, 
            width = bar_width,                    
            label = expl_label,
            edgecolor = bar_edgecolor)

    plt.bar( position[-1],
            L2_norm_pop_min,
            color = "gray",
            alpha = 1, 
            width = bar_width,                    
            label   = f_label,
            edgecolor = bar_edgecolor)

    plt.legend( fontsize=14)		
    plt.xticks( position, ticks_labels_expl, fontsize=14, rotation=0 )
    plt.ylabel( ylabel = r"$L^2(\mathbb{P})$-norm", fontsize=14)			
    plt.title(" Total energy of Shapley explanations")
    plt.tight_layout()
    plt.savefig( fname = pics_folder_name + '/expl_tot_models.png')
    plt.close()


    # plot individual and sums of shapley explanation norms             
    
    for m in range(n_partitions_expl):
        
        fig, ax   = plt.subplots( figsize = (10,7) )               

        dim = len(L2_norm_shap_expl)

        position  = 1 * np.arange(dim)
        
        bar_width = 1/4

        norm_labels = [""]*dim

        partition = partition_list_expl[m][0]
        idx       = partition_list_expl[m][1]
        partition_order_idx = partition_order_idx_list_expl[m]

        norm_labels = np.array(pred_names)[partition_order_idx]

        bar_label = r'''$\varphi(f_*)$'''

        bar_group_label = r'''$\varphi_{S_j}(f_*)$'''

        partition_label = partition_labels_expl[m]
        
        plt.bar( position,
                np.array(L2_norm_shap_expl)[partition_order_idx],
                color = explan_colors[1],
                alpha = 1, 
                width = bar_width,                        
                label = bar_label,
                edgecolor = bar_edgecolor, zorder=2)
        
        count           = np.array([len(partition[j]) for j in range(len(partition))])
        pos_partition   = np.cumsum(count)-1 - count/2 + (1/2)
        position_group  = pos_partition
        bar_width_group = count-(1/2) 

        plt.bar( position_group,                        
                L2_norm_group_shap_expl[m],
                color = explan_group_colors[1],
                alpha = 0.75, 
                width = bar_width_group,                        
                label = bar_group_label,
                edgecolor = bar_edgecolor, zorder=1)
        
        plt.legend( fontsize=14)
        plt.xticks( position, norm_labels, fontsize=14, rotation=90 )
        plt.ylabel( ylabel = r"$L^2(\mathbb{P})$-norm", fontsize=14)
        plt.title(" Shapley explanations and group-Shap with partition {0} for {1} model".format(partition_label,model_name))
        plt.tight_layout()
        plt.savefig( fname = pics_folder_name + '/expl_norms_shap_vs_sum_shap_model_part_{0}.png'.format(idx))		
        plt.close()

if __name__=="__main__":
        main()

