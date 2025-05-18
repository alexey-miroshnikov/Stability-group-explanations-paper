from copy import copy, deepcopy
import numpy as np
from tqdm import tqdm

####################################################################################

class GameBase:
    
    def __init__( self, carrier = None, n_players = None ):

        """ carrier (or support) is a set of non-negative integers 
            where the game is played.
            n_players = number of players. 
            NOTE: typically the carrier is a subset of the universe of players,
            but in this implementation they coincide.
            NOTE: n_players not required if carrier is provided.
            if carrier is not provided, n_players gives rise to the 
            default carrier {0,1,2,...,n_players-1}.
        """
        
        # create placeholders:
        self._carrier     = None
        self._n_players   = None
                    
        if carrier is not None and n_players is not None:
            raise ValueError("Both inputs cannot be provided.\
                Input either carrier or n_players.")

        if carrier is None and n_players is None:
            self._carrier   = {0}
        else:	
            if carrier is None: # then we read of the carrier from n_players				
                assert isinstance( n_players, (int,np.int32) ), \
                    "[Error] n_players must be an integer number."
                assert n_players>=1, "[Error] n_players must be positive."
                self._carrier = set( [i for i in range(n_players)] )			
            else:
                assert isinstance(carrier,set), "[Error] carrier must be a set."
                self._carrier = copy(carrier)

        assert len(np.where(np.array(list(self._carrier))<0)[0])==0,\
            "carrier must have non-negative integers"

        self._n_players = len(self._carrier) 

    def _index_set_check( self, index_set ):
        assert isinstance(index_set,(tuple,set,list)), \
            "[Error] index_set must be a set, tuple, or list."
        assert set(index_set).issubset(self._carrier), \
            "[Error] index_set must be a subset of carrier={0}".format(self._carrier)		
        
    def _game_call( self, index_set=None, **argdict ):
        raise NotImplementedError("[Error] do NOT use an abstract method of the base class.")

    def __call__(self, index_set=None, **argdict):
        self._index_set_check(index_set)
        return self._game_call(index_set=index_set,**argdict)

    def update_carrier(self, carrier ):
        assert isinstance(carrier,set), "[Error] carrier must be a set."
        self._carrier   = copy(carrier)
        self._n_players = len(self._carrier)

    @property
    def n_players(self):
        return copy(self._n_players)

    @property
    def carrier(self):
        return copy(self._carrier)

    # by default a game with no dataset becomes deterministic
    # the offspring ProbGameBase has this property too which 
    # is set to False if the background dataset is not None.
    @property
    def isdeterministic(self):
        return True
    

####################################################################################

class ProbGameBase(GameBase):

    def __init__( self, data_set = None, n_players = None, carrier = None ):

        """
        carrier is a set of non-negative integers. 
        n_players is the size of the carrier
        data_set is np.ndarray dataset which can be None
        """

        super().__init__(n_players = n_players, carrier = carrier)
        self._data_set    = None

        if data_set is not None:
            assert isinstance(data_set,np.ndarray),\
            "[Error] data_set must be np.ndarray."
            assert len(data_set.shape)==2,\
            "[Error] data_set shape must be (N,d)."
            self._data_set = data_set # [Question]: should we keep a pointer or copy?

    def _values_check( self, values ):

        if self._data_set is None:
            assert values is None, \
            "[Error] values must be None to agree with the background data_set."
        else:
            assert isinstance(values,np.ndarray), \
            "[Error] values must be np.ndarray."
            assert len(values.shape)==2, \
            "[Error] values shape must be (N,d)."
            assert self._data_set.shape[1] == values.shape[1], \
            "[Error] values.shape[1] must be the same as data_set.shape[1]={0}.".format(self._data_set[1])
        
    def _game_call( self, index_set, values, **argdict ): # abstract raw game call:
        raise NotImplementedError( "[Error] not implemented (abstract method)." )

    # Assumption: any offspring with data_set=None is considered to be deterministic.
    # a call for a deterministic game must be reimplemented and must require only index_set: game(S) 

    def __call__(self, index_set = None, values = None, **argdict):
        self._index_set_check(index_set)
        self._values_check(values)
        return self._game_call(index_set=index_set,values=values,**argdict)

    @property
    def isdeterministic(self):		
        if self._data_set is None:			
            return True
        else:
            return False

    @property
    def data_set(self):
        if self._data_set is None:
            return None
        else:
            return copy(self._data_set)

    @property
    def data_shape(self):
        if self._data_set is None:
            return None
        else:
            return copy(self._data_set.shape)

    @property
    def data_dim(self):
        if self._data_set is None:
            return None
        else:
            return copy(self._data_set.shape[1])

    
####################################################################################

class QuotientProbGame(ProbGameBase):

        def __init__( self, game, index_partition, subset_replacement=None, subset_index = None ):
            assert isinstance(game,ProbGameBase), "[Error] game must be of ProbBaseGame offspring."
            self._index_partition_check(game.carrier,index_partition)
            self._game = game                       # keep only pointer as this is a wrapper class
            self._index_partition = index_partition # keep only pointer as this is a wrapper class
            quotient_carrier = {i for i in range(len(index_partition))}
            super().__init__(data_set=game.data_set, carrier = quotient_carrier)
            
            if subset_replacement is None: # is subset_replacement is not given then game is just quotient game
                assert subset_index is None, \
                    "[Error] subset_index must be None when subset_replacement is None."
                self._T = None
                self._index_T = None
                self._adj_index_partition = None
            else:
                assert subset_index is not None, \
                    "[Error] subset_index must not be None when subset_replacement is not None."
                assert isinstance(subset_index,(int,np.int,np.int32)), \
                    "[Error] subset_index must be nonnegative integer."					
                assert subset_index>=0,\
                    "[Error] subset_index must be nonnegative integer."
                assert subset_index <= len(self._index_partition), \
                    "[Error] subset_index must be less than {0}".format(len(self._index_partition))				
                assert set(subset_replacement).issubset(set(self._index_partition[subset_index])), \
                    "[Error] subset_replacement must be a subset of {0}".format(self._index_partition[subset_index])
                self._T = deepcopy(subset_replacement)
                self._index_T = subset_index
                self._adj_index_partition = deepcopy(self._index_partition)
                self._adj_index_partition[self._index_T] = deepcopy(self._T) # no need to deepcopy?

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

        # no need really to re-implement a call:
        def __call__(self, index_set, values=None, **argdict):
            self._index_set_check(index_set)
            self._values_check(values)
            return self._game_call(index_set=index_set,values=values,**argdict)

        # raw call
        def _game_call(self, index_set, values=None, **argdict):

            U=[] # union

            if len(index_set)!=0:
                if self.isquotient:				
                    for j in index_set:
                        U.extend(self._index_partition[j])
                else:
                    for j in index_set:
                        U.extend(self._adj_index_partition[j])

            if self._game.isdeterministic: # means it does not depend on the dataset
                return self._game._game_call(index_set=U, **argdict)
            else:
                return self._game._game_call(index_set=U,values=values,**argdict)	

        @property
        def isquotient(self):
            if self._T is None:
                return True
            else:
                return False

        # TODO: test this function.
        def update_subset_replacement(self, subset_replacement=None, subset_index=None ):

            if subset_replacement is None: # if subset_replacement is not given then game is just quotient game
                assert subset_index is None, \
                    "[Error] subset_index must be None when subset_replacement is None."
                self._T = None
                self._index_T = None
                self._adj_index_partition = None
            else:
                assert subset_index is not None, \
                    "[Error] subset_index must not be None when subset_replacement is not None."
                assert isinstance(subset_index,int), \
                    "[Error] subset_index must be nonnegative integer."
                assert subset_index>=0,\
                    "[Error] subset_index must be nonnegative integer."
                assert subset_index <= len(self._index_partition), \
                    "[Error] subset_index must be less than {0}".format(len(self._index_partition))
                assert set(subset_replacement).issubset(set(self._index_partition[subset_index])), \
                    "[Error] subset_replacement must be a subset of {0}".format(self._index_partition[subset_index])
                self._T = deepcopy(subset_replacement)
                self._index_T = subset_index
                self._adj_index_partition = deepcopy(self._index_partition)
                self._adj_index_partition[self._index_T] = deepcopy(self._T)

        @property
        def subset_replacement(self):
            return deepcopy(self._T)

        @property
        def subset_replacement_index(self):
            return deepcopy(self._index_T)

        @property
        def index_partition(self):
            return deepcopy(self._index_partition)

        @property
        def adj_index_partition(self):
            return deepcopy(self._adj_index_partition)

        @property
        def input_carrier(self):
            return copy(self._game.carrier)

        @property
        def iswrapper(self):
            return True

        @property
        def input_game(self):
            return self._game # copy?


####################################################################################
####################################################################################

# Wrapper around ProbGameBase which allows to play a game on a mapped set of indexes
# Given a carrier T, index map it construct a game played on {0,1,...,|T|-1}.
class MappedProbGame(ProbGameBase):

    def __init__( self, game, idx_map ):

        assert isinstance(game,ProbGameBase), "game must be of ProbBaseGame offspring"
        self._game = game
        
        assert isinstance(idx_map,dict), "idx_map must be a dictionary"

        assert set(idx_map.keys())==game.carrier,\
            "the set of keys in idx_map must be equal to game.carrier"
        
        assert len(set(idx_map.values()))==len(idx_map.values()),\
            "values in idx_map must be distinct"
        
        self._pi     = copy(idx_map)

        self._pi_inv = {v:k for k,v in idx_map.items()}

        self._pi_vec     = np.vectorize(lambda idx: self._pi[idx])

        self._pi_inv_vec = np.vectorize(lambda idx: self._pi_inv[idx])

        pi_carrier = set(self._pi.values())

        super().__init__(data_set = game.data_set, carrier = pi_carrier )

    # raw call: "values" is optional to handle deterministic & non-deterministic games.
    def _game_call( self,index_set,values=None,**argdict ): 

        if len(index_set)==0:			
            pi_inv_index_set=[]
        else:
            pi_inv_index_set = list(self._pi_inv_vec(list(index_set)))

        if self._game.isdeterministic:

            return self._game._game_call(index_set=pi_inv_index_set, **argdict)
        else:
            return self._game._game_call(index_set=pi_inv_index_set,values=values,**argdict)

    @property
    def input_game(self):
        return self._game

    @property
    def iswrapper(self):
        return True

    @property
    def pi(self):
        return self._pi

    @property
    def pi_inv(self):
        return self._pi_inv

####################################################################################
####################################################################################
# The marginal game is defined as the marginal expectation of the model f(X)
# with respect to features X_S for S in N:={0,1,2,...,d-1}.
# The default carrier is set to N, but can be restricted to any subset of N.

class MarginalProbGame(ProbGameBase):

    def __init__( self, func, pred, carrier = None ):

        assert isinstance(pred,np.ndarray)
        assert len(pred.shape)==2
        assert pred.shape[0]>=1
        assert pred.shape[1]>=1
        assert callable(func)

        
        sup_carrier = set( [i for i in range(pred.shape[1])] )

        # construction the parent:
        if carrier is None:
            super().__init__( data_set = pred, carrier=sup_carrier )
        else:
            assert isinstance(carrier,set), "[Error] carrier must be a set."
            assert carrier.issubset(sup_carrier), \
            "[Error] carrier={0} must be a subset of {1}".format(carrier,sup_carrier)
            super().__init__( data_set = pred, carrier=carrier )

        # adding new attributes:
        self._func = func
        self._pred = self._data_set # predictors
        self._pred_copy = copy(self._pred) # a copy of predictors

    # Overloaded game call with no checks (checks done in the parent)
    def _game_call(self, index_set, values, **argdict):
        # when the game is evaluated at index_set, the parent class checks it is in the carrier.
        return self._marginal_mean( x=values, index_set=index_set, **argdict)

    # auxilary method that computes marginal function 
    def _marginal_mean( self, x=None, index_set=None, **argdict ):	

        progress = argdict.get("progress",False)

        S = index_set  # the parent class ensures S belongs to the carrier        

        X = self._pred_copy # copy of the background dataset of features

        assert x is not None, 'x values array must be provided.'

        assert isinstance(x,np.ndarray)

        assert len(x.shape)== 2, 'x must be 2d np.array'		        
        
        assert x.shape[1]==X.shape[1], 'x must have {0} columns'.format(X.shape[1])

        if len(S) == x.shape[1]: # S=N

            return self._func(x)
        
        elif len(S)==0: # S=empty set

            return np.mean(self._func(X)) * np.ones(shape=(x.shape[0],))
        
        else: # |N|>|S|>=1

            x_s = x[:,S]
                        
            vals = np.zeros( shape = (x_s.shape[0],) )

            for i in tqdm(range(x_s.shape[0]),disable=not progress):

                X[:,S] = x_s[i,:] # impute the number x_s in s

                vals[i] = np.mean(self._func(X))
            
            X[:,S] = self._pred[:,S] # restore the values in X

            return vals

    def update_carrier(self,carrier):
        assert isinstance(carrier,set), "[Error] carrier must be a set"
        sup_carrier = set( [i for i in range(self.data_dim)] )
        assert carrier.issubset(sup_carrier),\
            "[Error] carrier={0} must be a subset of {1}".format(carrier,sup_carrier)
        self._carrier   = carrier
        self._n_players = len(carrier)
