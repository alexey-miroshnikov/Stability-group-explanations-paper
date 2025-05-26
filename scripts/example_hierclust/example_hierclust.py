############################################## 
# Hierarchical clustering example for
# features with dependencies
############################################## 
# Code authors:
#   Alexey Miroshnikov
#   Konstandinos Kotsiopoulos
# Consultant:
#   Khashayar Filom
############################################## 
# Version 1: May 2025
############################################## 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import minepy
import sys
import os
from scipy.cluster.hierarchy import dendrogram, linkage
from argparse import ArgumentParser
from statsmodels.distributions.empirical_distribution import ECDF

script_dirname = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dirname)

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

# MAIN ######################################
def main( ):

    parser = ArgumentParser( description = 'Script for hierarchical clustering using MIC or PCA.')    
    arg = parser.parse_args()

    res_folder_name = script_dirname + "/results"

    if not os.path.exists(res_folder_name):
        os.mkdir(res_folder_name)

    print("generating dataset...")
    # Specify distribution
    N = 10000
    np.random.seed(seed=1234)
    x0 = np.random.uniform(-4*np.pi,4*np.pi,N)			
    x1_quad = np.power(x0,2) + np.random.normal(0,1,N)
    x2_sin = np.sin(x0) + np.random.normal(0,0.25,N)
    x3_lin = 0.5*x0 + np.random.normal(0,0.25,N)
    x4_uni = np.random.uniform(0,10,N)

    theta = np.random.uniform(0,2*np.pi,N)
    radius = 2
    eps5 = np.random.normal( 0.0, scale = 0.05 * radius, size = N )
    eps6 = np.random.normal( 0.0, scale = 0.05 * radius, size = N )
    x5_circ = radius*np.cos(theta)+eps5
    x6_circ = radius*np.sin(theta)+eps6

    data = np.array([x0,x1_quad,x2_sin,x3_lin,x4_uni,x5_circ,x6_circ])
        
    # save dataset:
    pd.DataFrame(data.T).to_csv(res_folder_name + '/dataset.csv',
                                index=False,header=None)

    # Plotting:
    figsize=(8,8)
    alpha_scatter = 0.6
    
    group1_color='red'
    group2_color='blue'
    group3_color='green'

    label_fontsize=16
    title_fontsize=20
    ticks_fontsize=14


    print("plotting features...")

    # Plotting

    # plot X0
    plt.figure(figsize=figsize)
    xi=np.linspace(-5*np.pi,5*np.pi,1001)
    cdf_x0=ECDF(x0)
    plt.plot(xi,cdf_x0(xi), color=group1_color)
    plt.title(r'CDF of $X_0$', fontsize = title_fontsize)
    plt.xticks(fontsize=ticks_fontsize)
    plt.yticks(fontsize=ticks_fontsize)
    plt.xlabel('$x_0$',fontsize=label_fontsize)
    plt.ylabel('$F_{X_0}$',fontsize=label_fontsize)
    plt.tight_layout()
    plt.savefig(fname = res_folder_name+'/cdf_x0.png')

    # plot X0 vs X1
    plt.figure(figsize=figsize)
    plt.scatter(x0, x1_quad, s=0.3, 
                linewidths = 0.5, 
                alpha = alpha_scatter,
                color = group1_color)
    plt.title(r'$X_0 \ vs \ X_1$', fontsize = title_fontsize)
    plt.xticks(fontsize=ticks_fontsize)
    plt.yticks(fontsize=ticks_fontsize)
    plt.xlabel('$X_0$',fontsize=label_fontsize)
    plt.ylabel('$X_1$',fontsize=label_fontsize)
    plt.tight_layout()
    plt.savefig(fname = res_folder_name+'/data_quadratic.png')
    plt.close()

    # plot X0 vs X2
    plt.figure(figsize=figsize)
    plt.scatter(x0, x2_sin, s=0.3, 
                linewidths = 0.5, 
                alpha=alpha_scatter,
                color = group1_color)
    plt.xticks(fontsize=ticks_fontsize)
    plt.yticks(fontsize=ticks_fontsize)
    plt.xlabel('$X_0$',fontsize=label_fontsize)
    plt.ylabel('$X_2$',fontsize=label_fontsize)
    plt.title(r'$X_0 \ vs \ X_2$', fontsize = title_fontsize )
    plt.tight_layout()
    plt.savefig(fname = res_folder_name+'/data_sine.png')
    plt.close()

    # plot X0 vs X3
    plt.figure(figsize=figsize)
    plt.scatter(x0, x3_lin, s=0.3,
                linewidths = 0.5, 
                alpha=alpha_scatter,
                color = group1_color)
    plt.xticks(fontsize=ticks_fontsize)
    plt.yticks(fontsize=ticks_fontsize)
    plt.xlabel('$X_0$',fontsize=label_fontsize)
    plt.ylabel('$X_3$',fontsize=label_fontsize)
    plt.title(r'$X_0 \ vs \ X_3$', fontsize = title_fontsize)
    plt.tight_layout()
    plt.savefig(fname = res_folder_name+'/data_line.png')
    plt.close()

    # plot X0 vs X4
    plt.figure(figsize=figsize)
    plt.scatter(x0, x4_uni, s=0.3, 
                linewidths = 0.5, 
                alpha=alpha_scatter,
                color = group2_color)
    plt.xticks(fontsize=ticks_fontsize)
    plt.yticks(fontsize=ticks_fontsize)
    plt.xlabel('$X_0$',fontsize=label_fontsize)
    plt.ylabel('$X_4$',fontsize=label_fontsize)
    plt.title(r'$X_0 \ vs \ X_4$', fontsize = title_fontsize)
    plt.tight_layout()
    plt.savefig(fname = res_folder_name+'/data_uniform.png')
    plt.close()

    # plot X5 and X6
    plt.figure(figsize=figsize)
    plt.scatter(x5_circ, x6_circ, s=0.3, 
                linewidths = 0.5, 
                alpha=alpha_scatter,
                color=group3_color)
    plt.xticks(fontsize=ticks_fontsize)
    plt.yticks(fontsize=ticks_fontsize)
    plt.xlabel('$X_5$',fontsize=label_fontsize)
    plt.ylabel('$X_6$',fontsize=label_fontsize)
    plt.title(r'$X_5 \ vs \ X_6$', fontsize = title_fontsize )
    plt.tight_layout()
    plt.savefig(fname = res_folder_name+'/data_circle.png')
    plt.close()

   # Create dendrogram based on MIC and Correlation Matrix

    print("computing linkage based on correlations...")
    linkage_corr = linkage( data, metric = abs_corr, method='average')
    print("computing linkage based on mic_e...")
    linkage_mic  = linkage(data, metric=mic_dist, method='average')
    
    # save linkage matrices
    pd.DataFrame(linkage_corr).to_csv(res_folder_name + \
                                      '/linkage_corr.csv',index=False,header=None)
    pd.DataFrame(linkage_mic).to_csv(res_folder_name \
                                     +'/linkage_mic.csv',index=False,header=None)

    print("plotting dendrograms...")

    dendrogram_labels = ['$X_0$' , 
                            r'$X_1=X_0^2$',
                            r'$X_2=\sin(X_0)$',
                            r'$X_3=\frac{1}{2} \cdot X_0$',
                            r'$X_3=Unif(0,10)$',
                            r'$X_5=2\cdot\cos(\theta)$',
                            r'$X_6=2\cdot\sin(\theta)$']

    # Plot dendrograms
    plt.figure(figsize=(10,6))
    plt.title('Correlation Matrix Hierarchical Clustering')
    dendrogram(linkage_corr, 
               labels=dendrogram_labels, 
               orientation='right')
    plt.xlabel('$1-|\\rho|$')
    plt.ylabel('features')
    plt.xlim([0.0,1])
    plt.plot( [0.7,0.7], [0,70],
             linestyle="--", 
             color="black", 
             marker='o', 
             markeredgecolor="black", 
             alpha=0.5)
    plt.tight_layout()
    plt.savefig(fname = res_folder_name+'/hierclus_corr_average.png')
    plt.close()

    plt.figure(figsize=(10,6))
    dendrogram(linkage_mic, 
               labels=dendrogram_labels,
               orientation='right')
    plt.xlabel('$1 - MIC_e$')
    plt.ylabel('features')    
    plt.xlim([0.0,1])
    plt.plot( [0.7,0.7], [0,70], 
             linestyle="--", 
             color="black", 
             marker='o', markeredgecolor="black",
             alpha=0.5)
    plt.title('$MIC_e$ Hierarchical Clustering')
    plt.plot()
    plt.tight_layout()
    plt.savefig(fname = res_folder_name+'/hierclus_MIC_average.png')
    plt.close()

if __name__=="__main__":
    main()