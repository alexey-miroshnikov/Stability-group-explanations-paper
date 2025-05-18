import numpy as np
from copy import copy,deepcopy
from libgame import GameBase
from libgame import ProbGameBase
from libgame import MappedProbGame
from libgame import QuotientProbGame
from libshapvalue import DirectQuotShapley


####################################################################################
####################################################################################

class OwenSubGame(ProbGameBase):

    def __init__( self, game, index_partition, subset_index ):
        assert isinstance(game,ProbGameBase), "[Error] game must be of ProbBaseGame offspring."
        self._index_partition_check(game.carrier,index_partition)
        self._game = game                       # keep only pointer as this is a wrapper class
        self._index_partition = index_partition # keep only pointer as this is a wrapper class

        assert isinstance(subset_index,int), \
             "[Error] subset_index must be non-negative integer."
        assert subset_index>=0, \
             "[Error] subset_index must be non-negative integer."
        assert subset_index <= len(self._index_partition), \
            "[Error] subset_index must be <= {0}".format(len(self._index_partition)-1)		
        self._subset_index = subset_index

        super().__init__(data_set=game.data_set, carrier=set(self._index_partition[subset_index]))

        self._interm_game = QuotientProbGame(game=game,index_partition=self._index_partition)

    def _index_partition_check(self,carrier,index_partition):
        P=index_partition # nickname for index_partition
        assert isinstance(carrier,set), '[Error] carrier must be a set.'
        assert isinstance(P,list), '[Error] index_partition must be list of lists.'
        assert len(P)>=1, '[Error] index_partition must be nonempty.'
        for S in P:
            assert isinstance(S,(list,tuple,set)), \
                "[Error] Each elements of the partition must be a list, tuple, or set."
        all_indexes = sum(P,[]) # combine all lists together
        assert len(all_indexes) == len(set(all_indexes)), \
            '[Error] index_partition must have distinct elements.'
        assert set(all_indexes) == carrier, \
            '[Error] index_partition must contain all indexes of the carrier {0}'.format(carrier)
        return True

    # call is identical to the parent
    def __call__(self, index_set, values=None, **argdict):
        self._index_set_check(index_set)
        self._values_check(values)
        return self._game_call(index_set=index_set,values=values,**argdict)

    def _game_call(self, index_set, values=None, **argdict):		
        vT = self._interm_game
        T  = index_set
        vT.update_subset_replacement(subset_replacement=T,subset_index=self._subset_index)
        shap_vT = DirectQuotShapley(game=vT)				
        return shap_vT(values,group_idx=self._subset_index)

    def update_subset_index(self,subset_index):
        assert isinstance(subset_index,int), \
             "[Error] subset_index must be non-negative integer."
        assert subset_index>=0, \
             "[Error] subset_index must be non-negative integer."
        assert subset_index <= len(self._index_partition), \
            "[Error] subset_index must be <= {0}".format(len(self._index_partition)-1)		
        self._subset_index = subset_index
        self._carrier=set(self._index_partition[subset_index])						
        
    @property
    def subset_index(self):
        return deepcopy(self._subset_index)

    @property
    def index_partition(self):
        return deepcopy(self._index_partition)

    @property
    def input_carrier(self):
        return copy(self._game.carrier)

    @property
    def iswrapper(self):
        return True

    @property
    def input_game(self):
        return self._game

###############################################################################
###############################################################################

class DirectOwen(DirectQuotShapley):

    def __init__( self, 
                  game,
                  index_partition = None, 
                  verbal=False ):

        # NOTE: only standard carriers {0,...,n_players-1} are allowed.
        assert isinstance(game,GameBase),\
            "[Error] game must be of class GameBase or its offspring."

        assert game.carrier==set(range(game.n_players)), \
            "[Error] Only standard carriers are allowed."
        
        # game is deterministic means it has a scalar output 
        # with no background and no input datasets
        self._game_isdeterministic = game.isdeterministic 

        super().__init__(game=game,
                         index_partition=index_partition,
                         verbal=verbal)
        
        self._owen_sub_game = OwenSubGame(game,
                                          self._index_partition,
                                          subset_index=0)
    

    def __call__(self, pred = None, *, 
                       progress = False, 
                       progress_game = False, 
                       out_format = "np.array" ):

        self._input_dataset_check(pred)

        # nicknames:
        P        = self._index_partition

        game     = self._game

        owengame = self._owen_sub_game

        owen_values_list = [None]*game.n_players

        for j in range(len(P)):

            S_j = P[j]

            owengame.update_subset_index(j) # v^{(j)}

            pi_map_j = dict(zip(S_j,range(len(S_j))))

            pi_owengame_j = MappedProbGame(game=owengame, idx_map=pi_map_j )

            shap_owengame_pi_j = DirectQuotShapley(pi_owengame_j)
        
            owen_values_pi_j = shap_owengame_pi_j(pred, 
                                                  progress=progress, 
                                                  progress_game=progress_game, 
                                                  out_format="list") 

            for player_j,pi_player_j in pi_map_j.items():

                owen_values_list[player_j] = deepcopy(owen_values_pi_j[pi_player_j])

        if out_format=="list": 
            
            return deepcopy(owen_values_list)

        if out_format=="np.array":
            
            temp = np.array(owen_values_list)    # turning into np.array
            
            if len(temp.shape)==1:               # means shap is a list of scalars
            
                return deepcopy(temp) # no need to deepcopy
            
            elif len(temp.shape)==2:  # means shap is a list of lists
            
                return deepcopy(temp.T) # transpose to have the shape (N,n_players)
            
            else: raise ValueError("game output format must be either scalars or columns \
                        (N,) where N is pred.shape[0]")