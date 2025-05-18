import numpy as np
from copy import copy,deepcopy
import numbers

##################################################################### 
# generate hyper parameters
def generate_hp_states( bounds_dict, size=None, seed=None ):

    if seed is not None:
        np.random.seed(seed)
        
    if size is None:
        return_a_state = True
        size=1
    else:
        assert isinstance(size,numbers.Integral)
        assert size>=1
        return_a_state = False

    hp_states = [None] * size

    template_state = deepcopy(bounds_dict)

    for key in template_state.keys():
        template_state[key] = None

    for j in range(len(hp_states)):

        hp_states[j] = copy(template_state)

        for key in hp_states[j].keys():

            low  = bounds_dict[key][0][0]
            high = bounds_dict[key][0][1]

            if bounds_dict[key][1]=="int":
                value = np.random.randint(low=low,high=high+1)
            elif bounds_dict[key][1]=="float":
                value = np.random.uniform(low=low,high=high)
            elif bounds_dict[key][1]=="list":
                size = bounds_dict[key][0][2]
                if size>0:
                    value = np.random.randint(low=low,high=high,size=size)
                else:
                    value = []

            else:
                raise ValueError
            
            hp_states[j][key]=copy(value)
    
    if return_a_state:
        return hp_states[0]
    else:
        return hp_states

def logit(p, eps=0):
    return np.log((p+eps)/(1-p+eps))

def logistic(X):
    return (np.exp(X)/(1+np.exp(X)))



##################################################################### 
# L2-norm of a random variable
def L2_norm_rv(X,*,axis=None):

    assert isinstance(X,np.ndarray)
    assert len(X.shape)<=2

    if len(X.shape)==1:
        return np.sqrt(np.mean(np.power(X,2)))

    if len(X.shape)==2:		
        if axis is not None:
            assert axis>=0
            assert axis<=1
            return np.sqrt(np.mean(np.power(X,2),axis=axis))
        else:
            return np.sqrt(np.mean(np.power(X,2),axis=axis))

##################################################################### 
# L2-norm of a vector:
def L2_norm_vec(X,*,axis=None):

    assert isinstance(X,np.ndarray)
    assert len(X.shape)<=2

    if len(X.shape)==1:
        return np.sqrt(np.sum(np.power(X,2)))

    if len(X.shape)==2:		
        if axis is not None:
            assert axis>=0
            assert axis<=1
            return np.sqrt(np.sum(np.power(X,2),axis=axis))
        else:
            return np.sqrt(np.sum(np.power(X,2),axis=axis))



