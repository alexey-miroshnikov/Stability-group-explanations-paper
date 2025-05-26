import numpy as np
import numbers
from copy import copy,deepcopy
from tqdm import tqdm
import itertools
from libgame import GameBase
from libgame import ProbGameBase

####################################################################################
# get combinations using itertools
####################################################################################

def get_combinations( S, include_list = False ):
    assert isinstance( S, ( tuple, list, np.ndarray ) )
    combinations=[]
    for m in range(len(S)): 
        combinations += list(map( list, itertools.combinations(S,m) ))
    if include_list:
        combinations.append(S)
    return combinations

####################################################################################
# DirectGroupShapley object that can be sued with any game v=v(X,S), for example PDP
####################################################################################

# For large values n can use Stirling's approximation or log=True which allows to wrk via logs to avoid product overfloat
# Note: we use the formula for Shapley where S contains i: w(n,S)* ( v(S)-v(S\i) ) and so |S|>=1
# for this reason w(s,n)=0 when s=0
def ShapleyCoefficients( n, log=False ):

    assert(n>0), "number of players must be positive"
    assert ( isinstance(n,int) or isinstance(n,np.int32) ), "[Error] Input must be an integer number!"

    idx = np.array( [float(i) for i in range(n+1)] )
    idx[0]=1

    if log:
        log_factorials   = np.cumsum(np.log(idx))
        log_coefficients = np.zeros(shape=(n+1,))
        for j in range(1,n+1):
            log_coefficients[j] = log_factorials[j-1] + log_factorials[n-j] - log_factorials[n] 
        coefficients = np.exp( log_coefficients )
        coefficients[0]=0
        return coefficients
    else:
        factorials   = np.cumprod( idx )
        coefficients = np.zeros(shape=(n+1,))
        for j in range(1,n+1):
            coefficients[j]=factorials[j-1]*factorials[n-j]/factorials[n]
        return coefficients

#################################################################################### 

class DirectQuotShapley:

#   game: could be either GameBase, ProbGameBase, or a callable object v(index_set,values|dataset)
#   if v(index_set,values|dataset) is a generic callable object then:
# 		pred_dim must be specified and it is the dimension of the input dataset "values"
# 		if n_players is not specified, then n_players=pred_dim (we play on input predictors)
#       carrier=[0,...n_players-1]      
#   else: we assume that input dataset "values" and the background dataset have the same dimension
# 		  and read off the dimension from the shape of the background dataset.
#	the number of players also reads of from the carrier.

    def __init__( self, 
                  game, # game(X,S) and pred_dim is the dimensionality of X
                  pred_dim  = None, 
                  n_players = None,
                  index_partition = None, verbal=False ):

        #declaration of the variables of the object:
        self._group_comb_list = None		
        self._group_to_single = None
        self._index_partition = None
        self._pred_dim  = None			
        self._n_players = None			
        self._game = None
        self._shap_coef = None
        self.update( game=game, 
                     pred_dim=pred_dim, 
                     n_players=n_players,
                     index_partition=index_partition, 
                     verbal=verbal )				

    def _index_partition_check(self,carrier,index_partition):
        # carrier must be a set
        P=index_partition # nickname for index_partition
        assert isinstance(carrier,set), '[Error] carrier must be a set.'
        assert isinstance(P,list), '[Error] index_partition must be list of lists.'
        assert len(P)>=1, '[Error] index_partition must be nonempty.'
        for S in P:
            assert isinstance(S,(list,tuple,set)), "[Error] Each elements of the partition must be a list, tuple, or set."
        all_indexes = sum(P,[]) # combine all lists together
        assert len(all_indexes) == len(set(all_indexes)), '[Error] index_partition must have distinct elements.'
        assert set(all_indexes) == carrier, '[Error] index_partition must contain all indexes of the carrier {0}'.format(carrier)
  
        return True


    def update(self, game, pred_dim = None, n_players = None, index_partition = None, verbal=False):		
        
        # read of the information from the game about the carrier:
        if isinstance(game,GameBase):
            if pred_dim is not None or n_players is not None:
                raise ValueError("[Error] pred_dim and n_players must not be specified. They are derived from the game.")
            assert game.carrier==set(range(game.n_players)), "[Error] games with non-standard carriers are not implemented."
            n_players = game.n_players
            if isinstance(game,ProbGameBase):
                pred_dim = game.data_dim # the assumption that background dataset and input datase have the same dimansion.
            else:
                pred_dim = None
        else: # otherwise pred_dim must be specified and by default n_players = pred_dim, unless specified
            assert pred_dim is not None, "pred_dim (dimension of the input dataset) must be specified for a generic game."
            assert isinstance(pred_dim, (int,np.int32)), "[Error] Input must be an integer number."
            assert pred_dim>=1, "[Error] Predictor dimension must be positive."

            if n_players is None:
                n_players = pred_dim
            else:
                assert isinstance(n_players, (int,np.int32)), "[Error] Input must be an integer number!"
                assert n_players>0, "[Error] n_players must be positive."

        standard_carrier = {i for i in range(n_players)} # the standard carrier {0,1,....,n_players-1}

        if index_partition is None:
            index_partition = [ [i] for i in range(n_players)]
        else:
            self._index_partition_check(carrier = standard_carrier, index_partition=index_partition)

        assert callable(game), "[Error] game must be callable."

        n_groups = len(index_partition)
        # construct all index combinations for the number of elements in the partition:
        group_idx_range = [ j for j in range(n_groups) ]
        group_comb_list = get_combinations( group_idx_range, include_list = True )
        group_comb_list = [ tuple( comb ) for comb in group_comb_list ]
        group_to_single_map = {}
        
        # v_bar(U) is induced by the game v( uniong of S_j j in U). Let us create a map from U to S.
        for comb in group_comb_list:
            group_to_single_map[comb]=[]
            for j in comb:
                group_to_single_map[comb].extend(index_partition[j])					

        self._group_comb_list = deepcopy(group_comb_list)
        self._group_to_single = deepcopy(group_to_single_map)
        self._index_partition = deepcopy(index_partition)
        self._pred_dim  = pred_dim
        self._n_players = n_players
        self._game = game # stays as a pointer
        self._shap_coef = ShapleyCoefficients(n_groups,log=False)
        
        if verbal:
            print("\nPredictor dimension: ")
            print(self._pred_dim)
            print("\nn_players: ")
            print(self._n_players)
            print("\nPartition: ")
            print(self._index_partition)
            print("\nGroup_game_idx_map:")
            [ print(key,": ",group_to_single_map[key]) for key in group_to_single_map ]				
            print("\nGroup subsets: ")
            print(self._group_comb_list)
            print("\nShapley coefficients: ")
            print(self._shap_coef)


    def _input_dataset_check(self,pred):

        if self._pred_dim is None: # this can happen only if game is GameBase offspring
            assert pred is None, "game is deterministic. it must not depend on the input dataset."
        else:
            assert isinstance(pred, np.ndarray), "input dataset pred must be np.ndarray."
            assert len(pred.shape)==2,  "[Error]: pred must be np.array of shape (N,dim)."
            assert pred.shape[0]>=1,    "[Error]: pred must be np.array of shape (N,dim)."
            assert pred.shape[1]>=1,    "[Error]: pred must be np.array of shape (N,dim)."
            assert pred.shape[1]==self._pred_dim, "[Error]: pred dimension must be {0}".format(self._pred_dim)

        return True

    # by default output format is "np.array" and another is a "list":
    def __call__(self, pred = None, *, group_idx=None, progress=False, progress_game=False, out_format="np.array"):

        # check input dataset pred:
        self._input_dataset_check(pred)
        
        # auxilary _game_call function that takes into account different types of games:
        def _game_call(index_set,values,**argdict):
            if isinstance(self._game, ProbGameBase):
                if self._game.isdeterministic:
                    return self._game._game_call(index_set=index_set,**argdict)
                else:
                    return self._game._game_call(index_set=index_set,values=values,**argdict)
            elif isinstance(self._game,GameBase):
                return self._game._game_call(index_set=index_set, **argdict)
            # elif isinstance(self._game, MarginalMeanFunc):
            # 	return self._game(index_set=index_set,values=values) 
                # order is flipped to accomodate backward compatibility including Marginal Expectation
            else:
                return self._game(values,index_set) 
               # order is flipped to accomodate backward compatibility including Marginal Expectation
        
        group_to_single_map = self._group_to_single
        game = self._game				
        n_groups = len(self._index_partition)
        shap_coef = self._shap_coef

        if group_idx is None:
            group_idx = [ j for j in range(n_groups) ]

        if isinstance( group_idx, numbers.Number):
            group_idx_is_number=True
            group_idx=[group_idx]
        else:
            group_idx_is_number=False
            assert isinstance( group_idx, (list,tuple)), 'group_idx must be a list or integer'

        for k in range(len(group_idx)):
            assert group_idx[k] < n_groups, 'index must range in 0...{0}'.format(n_groups)

        # precompute group-game at all group subsets for all X values:
        group_game_dict = {}
        for U in tqdm(group_to_single_map,disable = not progress):
            S = group_to_single_map[U] 
            group_game_dict[U] = _game_call(S,pred,progress = progress_game) 

        shap=[]
        for j in tqdm(group_idx, disable = True ):						
            shap_j = 0
            for U in group_to_single_map:					
                if j in U: # note in this case |U|>=1
                    U_j = list(U)
                    U_j.remove(j)
                    U_j = tuple(U_j)
                    u = len(U)
                    shap_j += shap_coef[u] * ( group_game_dict[U] - group_game_dict[U_j] )
            shap.append(copy(shap_j))

        if group_idx_is_number:
            return deepcopy(shap[0]) 
        else:
            # TODO: [mistake] format should respect the game output format
            if out_format=="list": return deepcopy(shap)
            if out_format=="np.array":
                temp = np.array(shap)    # turning into np.array
                if len(temp.shape)==1:   # means shap is a list of scalars
                    return deepcopy(temp)
                elif len(temp.shape)==2: # means shap is a list of lists
                    return deepcopy(temp.T) # transpose to have the shape (N,n_players)
                else: raise ValueError("game output format must be either scalars or columns (N,) where N is pred.shape[0]")

    @property
    def index_partition(self):
        return copy(self._index_partition)

    @property
    def pred_dim(self):
        return copy(self._pred_dim)

    @property
    def n_players(self):
        return copy(self._n_players)

    @property
    def shap_coef(self):
        return copy(self._shap_coef)