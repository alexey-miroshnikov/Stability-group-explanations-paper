############################################## 
# Example of (pseudo)-instabilities
# in models with linear dependencies
############################################## 
# Code authors:
#   Alexey Miroshnikov
#   Konstandinos Kotsiopoulos
# Consultant:
#   Khashayar Filom
############################################## 
# Version 1: May 17 2025
############################################## 

import numpy as np
import matplotlib.pyplot as plt; plt.style.use('ggplot')
import sys
import os
from argparse import ArgumentParser

script_dirname = os.path.dirname(os.path.abspath(__file__))

sys.path.append(script_dirname)

class datagenerator:
     
    def __init__(self, delta_dep=None,delta_noise=None):
        
        if delta_dep is None:

            self._delta_dep = 0.1 # dependencies
        
        else:

            self._delta_dep = delta_dep

        if delta_noise is None:
            
            self._delta_noise = 0.0 # response
        
        else:

            self._delta_noise = delta_noise


    def data_sampler(self, size, seed=None):

        if seed is not None:

            np.random.seed( seed = seed )
        
        # sample latent variables:
        
        eps0 = np.random.normal(loc=0.0,scale=self._delta_dep,size=size)
        
        eps1 = np.random.normal(loc=0.0,scale=self._delta_dep,size=size)        
        
        eps_noise = np.random.normal(loc=0.0,scale=self._delta_noise,size=size)
        
        Z = np.random.normal(loc=0.0,scale=1.0,size=size) # latent

        X = np.zeros(shape=(size,3))

        X[:,0] = Z + eps0
        
        X[:,1] = Z + eps1
        
        X[:,2] = np.random.normal(loc=0.0,scale=1.0,size=size)

        Y = linear_model()(X,0.0) + eps_noise

        eps = np.stack([eps0,eps1,eps_noise],axis=1)

        return X,Y,eps

    def update(self,**argdict):
         
        delta_dep = argdict.get("delta_dep", None )

        delta_noise = argdict.get("delta_noise", None )

        if delta_dep is not None:

            self._delta_dep = delta_dep

        if delta_noise is not None:         

            self._delta_noise = delta_noise

    @property
    def delta_dep(self):
        return self._delta_dep

    @property
    def delta_noise(self):
        return self._delta_noise


class linear_model():

    def __call__ (self,X,alpha):
        
        assert isinstance(X,np.ndarray)

        assert len(X.shape)==2

        assert X.shape[1]==3

        return (1+alpha) * X[:,0] + (1-alpha) * X[:,1] + X[:,2] * 0.5 # X2':=X2*0.5 ~ N(0,s^2=1/4)


class cond_shapley():

    def __call__(self,X,eps0,eps1,delta,alpha):

        shap_ce = np.zeros(shape=X.shape)

        shap_ce[:,0] = X[:,0] + (eps0-eps1)/(1+delta**2) # phi_ce_0(Y)

        shap_ce[:,0] += (alpha/2) * (eps0-eps1+(X[:,0]+X[:,1])*(delta**2)/(1+delta**2))
        
        shap_ce[:,1] = X[:,1] + (eps1-eps0)/(1+delta**2) # phi_ce_1(Y)

        shap_ce[:,1] += (alpha/2) * (eps0-eps1-(X[:,0]+X[:,1])*(delta**2)/(1+delta**2))

        shap_ce[:,2] = X[:,2]
    
        return shap_ce


class marg_shapley():

    def __call__(self,X,alpha):

        shap_me = np.zeros(shape=X.shape)

        shap_me[:,0] = (1+alpha) * X[:,0] 
        
        shap_me[:,1] = (1-alpha) * X[:,1] 

        shap_me[:,2] = X[:,2]
    
        return shap_me


class quot_marg_shapley(): # equal to conditional ones

    def __call__(self,X,alpha):

        shap_quot_me = np.zeros(shape=(X.shape[0],2))

        shap_quot_me[:,0] = (1+alpha) * X[:,0] + (1-alpha) * X[:,1]                

        shap_quot_me[:,1] = X[:,2]
    
        return shap_quot_me


def main( ):

    parser = ArgumentParser( description = 'Models in a Rashomon ball')    

    n_samples = 100

    delta_dep = 0.1

    delta_noise = 0.0

    pics_folder_name = script_dirname + "/pics"

    # creating folders for pictures:

    if not os.path.exists(pics_folder_name):
            
            os.mkdir(pics_folder_name)

    dt = datagenerator( delta_dep=delta_dep, delta_noise=delta_noise )

    X, Y, eps = dt.data_sampler( size = n_samples, seed=123)

    range_data = (np.min(Y),np.max(Y))

    r_data = np.max(range_data)*1.2

    dim = X.shape[1]

    model = linear_model()

    shap_ce = cond_shapley()

    shap_me = marg_shapley()

    shap_quot_me = quot_marg_shapley()

    n_models = 11

    alpha_list = np.linspace(-3.0,3.0,n_models)

    true_model_index = int(n_models/2)

    n_models = len(alpha_list)

    # model predictions:
    model_vals = [ model(X,alpha) for alpha in alpha_list ]
    
    # conditional shapley:
    shaps_ce_vals = [ shap_ce(X,eps[:,0],eps[:,1],dt.delta_dep,alpha) for alpha in alpha_list ]

    # marginal shapley:
    shaps_me_vals = [ shap_me(X,alpha) for alpha in alpha_list ]
   
    # quotient marginal shapley:
    shaps_quot_me_vals = [ shap_quot_me(X,alpha) for alpha in alpha_list ]


    # Plotting:

    resp_color = "grey"

    cmap_shap   = plt.get_cmap("jet")  # cmap for explanations
    shap_colors = cmap_shap(np.linspace(0.0,1,n_models))


    alpha = 0.5
    figsize = (10,8)
    
    markersize_shap = 6    
    markersize_pred = 6
    markersize_data = 10
    markeredgewidth = 0.5    
    legend_fontsize = 18
    label_fontsize  = 18
    
    model_labels = [ r'''$f_{''' +r'''{0:.2f}'''.format(alpha) + r"}$" for alpha in alpha_list]

    # plot predictions of each model:    

    for i in range(dim):

        fig, ax = plt.subplots( figsize = figsize )			

        for k in range(n_models):

            plt.plot( X[:,i], model_vals[k],
                        marker='o',
                        markersize=markersize_pred,
                        linestyle='',
                        color = shap_colors[k],
                        label = model_labels[k],
                        markeredgecolor = 'black',
                        markeredgewidth = markeredgewidth,
                        alpha=alpha,zorder=1)

        plt.xlabel( "$X_{0}$".format(i+1), fontsize=14 )
        plt.ylabel( "Explanations", fontsize=14 )
        plt.legend( fontsize=legend_fontsize, loc='center left', bbox_to_anchor=(1, 0.5))            
        plt.ylim((-r_data,r_data))
        plt.tight_layout()
        plt.savefig( fname = pics_folder_name + '/pred_versus_X{0}.png'.format(i+1) )        
        plt.close()


    # plot differences of responses between model 0 and true model:
    for i in range(dim):				
        fig, ax = plt.subplots( figsize = figsize )			
        for k in range(n_models): 
            plt.plot( X[:,i], Y - model_vals[k],
                        marker='o',
                        markersize=markersize_pred-1,
                        linestyle='',
                        color = shap_colors[k],
                        label = "$Y-$" + model_labels[k],
                        markeredgecolor='black',
                        markeredgewidth=markeredgewidth,
                        alpha=1,zorder=n_models-k)
        plt.ylim((-5,5))
        plt.xlabel("$X_{0}$".format(i+1), fontsize=label_fontsize)
        plt.ylabel("Prediction error", fontsize=label_fontsize)
        plt.legend( fontsize=legend_fontsize, loc='center left', bbox_to_anchor=(1,0.5))
        plt.tight_layout()				
        plt.savefig( fname = pics_folder_name + '/diff_model_resp_versus_X{1}.png'.format(k+1,i+1) )
        plt.close()


    # plot conditional explanations of each model:    
        
    for i in range(dim):

        fig, ax = plt.subplots( figsize = figsize )	

        for k in range(n_models):

            if k == true_model_index:
            
                label=r'''$\phi^{CE}''' + r'''_{0}'''.format(i+1)+"(Y)$",
            
            else:
            
                label = r'''$\bar{\phi}^{CE}''' + r'''_{0}'''.format(i+1)+"($" + model_labels[k] + "$)$"

            plt.plot( X[:,i], shaps_ce_vals[k][:,i],
                        marker='o',
                        markersize=markersize_shap,
                        linestyle='',
                        color = shap_colors[k],
                        markeredgecolor='black',
                        markeredgewidth=markeredgewidth,
                        label=label, 
                        alpha=alpha, zorder=k+2)
            
        plt.plot( X[:,i], Y,
                    marker='o',
                    markersize=markersize_data,
                    linestyle='',
                    color = resp_color,
                    markeredgecolor='black',
                    markeredgewidth=markeredgewidth,
                    label="Y",
                    alpha=0.75, zorder=0)

        plt.xlabel("$X_{0}$".format(i+1), fontsize=label_fontsize)
        plt.ylabel("Explanations", fontsize=label_fontsize)
        plt.legend( fontsize=legend_fontsize, loc='center left', bbox_to_anchor=(1,0.5))
        plt.ylim(-r_data,r_data)
        plt.tight_layout()
        plt.savefig( fname = pics_folder_name + '/shap_ce_versus_X{0}.png'.format(i+1))			
        plt.close()


    # plot marginal explanations of each model:    
    
    for i in range(dim):

        fig, ax = plt.subplots( figsize = figsize )	

        for k in range(n_models):

            if k == true_model_index:
            
                label=r'''$\bar{\phi}^{ME}''' + r'''_{0}'''.format(i+1)+"(f_{true})$",
            
            else:
            
                label = r'''$\bar{\phi}^{ME}''' + r'''_{0}'''.format(i+1)+"($" + model_labels[k] + "$)$"

            plt.plot( X[:,i], shaps_me_vals[k][:,i],
                        marker='o',
                        markersize=markersize_shap,
                        linestyle='',
                        color = shap_colors[k],
                        markeredgecolor='black',
                        markeredgewidth=markeredgewidth,
                        label=label,
                        alpha=alpha, zorder=k+2)
            
        plt.plot( X[:,i], Y,
                    marker='o',
                    markersize=markersize_data,
                    linestyle='',
                    color = resp_color,
                    markeredgecolor='black',
                    markeredgewidth=markeredgewidth,
                    label="Y",
                    alpha=0.75, zorder=n_models+1)

        plt.xlabel("$X_{0}$".format(i+1), fontsize=label_fontsize)
        plt.ylabel("Explanations", fontsize=label_fontsize)
        plt.legend( fontsize=legend_fontsize, loc='center left', bbox_to_anchor=(1,0.5))            
        plt.tight_layout()
        plt.savefig( fname = pics_folder_name + '/shap_me_versus_X{0}.png'.format(i+1))			
        plt.close()


    P = [[0,1],[2]]

    for j in range(len(P)):
        
        Sj=P[j]

        for i in P[j]:

            fig, ax = plt.subplots( figsize = figsize )	

            for k in range(n_models):

                if k == true_model_index:
                    
                    label=r'''$\bar{\phi}^{ME,\!\mathcal{P}}''' + r'''_{0}'''.format(i+1)+"(f_{true})$",
                
                else:
                
                    label = r'''$\bar{\phi}^{ME,\!\mathcal{P}}''' + r'''_{0}'''.format(i+1)+"($" + model_labels[k] + "$)$"

                plt.plot( X[:,i], shaps_quot_me_vals[k][:,j],
                            marker='o',
                            markersize=markersize_shap,
                            linestyle='',
                            color = shap_colors[k],
                            markeredgecolor='black',
                            markeredgewidth=markeredgewidth,
                            label=label,
                            alpha=alpha, zorder=k+2)
            
            plt.plot( X[:,i], Y,
                        marker='o',
                        markersize=markersize_data,
                        linestyle='',
                        color = resp_color,
                        markeredgecolor='black',
                        markeredgewidth=markeredgewidth,
                        label="Y",
                        alpha=0.75, zorder=n_models+1)

            plt.xlabel("$X_{0}$".format(i+1), fontsize=label_fontsize)
            plt.ylabel("Explanations", fontsize=label_fontsize)
            plt.legend( fontsize=legend_fontsize, loc='center left', bbox_to_anchor=(1,0.5))
            plt.ylim(-r_data,r_data)
            plt.tight_layout()
            plt.savefig( fname = pics_folder_name + '/shap_quot_me_versus_X{0}.png'.format(i+1))			
            plt.close()

    # plot difference between marginal and conditinal explanations of each model:
    
    for i in range(dim):

        fig, ax = plt.subplots( figsize = figsize )	

        for k in range(n_models):

            if k == true_model_index:
            
                label=r'''$(\bar{\phi}^{ME}'''+ r'''_{0}'''.format(i+1) + r'''-\bar{\phi}^{CE}''' + r'''_{0})'''.format(i+1)+"(f_{true})$",
            
            else:
            
                label = r'''$(\bar{\phi}^{ME}''' + r'''_{0}'''.format(i+1) + r'''-\bar{\phi}^{CE}''' + r'''_{0})'''.format(i+1)+"($" + model_labels[k] + "$)$"

            plt.plot( X[:,i], shaps_me_vals[k][:,i]-shaps_ce_vals[k][:,i],
                        marker='o',
                        markersize=markersize_shap,
                        linestyle='',
                        color = shap_colors[k],
                        markeredgecolor='black',
                        markeredgewidth=markeredgewidth,
                        label=label,
                        alpha=alpha, zorder=k+2)
                        
        plt.xlabel("$X_{0}$".format(i+1), fontsize=label_fontsize)
        plt.ylabel("Difference of explanations", fontsize=label_fontsize)
        plt.legend( fontsize=legend_fontsize, loc='center left', bbox_to_anchor=(1,0.5))            
        plt.tight_layout()
        plt.savefig( fname = pics_folder_name + '/shap_diff_me_ce_versus_X{0}.png'.format(i+1))			
        plt.close()


if __name__=="__main__":

    main()