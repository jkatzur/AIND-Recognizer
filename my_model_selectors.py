import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN

    where L is the likelihood of the fitted model
    p is the number of parameters = transistion probs (states**2) + means(states*data) + covars(states*data)
        where states = is the number of model states and data = number data points
        this becomes states**2 + 2 * states * N - 1
    N is the number of data points

    From source...
    The term −2 log L decreases with increasing model complexity (more parameters)
    whereas the penalties 2p or p log N increase with increasing complexity
    The BIC applies a larger penalty whenN>e2 =7.4.

    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        try:
            min_max = range(self.min_n_components, self.max_n_components+1)
            # This is how I track the BIC for each num components
            values = {i:0 for i in min_max}
            # For each potential number of components do the following...
            for i in min_max:
                hmm_model = self.base_model(i)
                # log Likelihood
                L = hmm_model.score(self.X,self.lengths)
                # number of data points
                N = sum(self.lengths)
                # parameter calculation from
                p = (i ** 2) + (2 * i * N) - 1
                values[i] = (-2 * L) + (p * np.log(N))
                print("For I = {} states; LL = {}, BIC = {}".format(i, L, values[i]))
            # This section determines which number of components has the highest average log Likelihood
            best_num_components = self.min_n_components
            is_max = values[self.min_n_components]
            for i in min_max:
                if values[i] > is_max:
                    is_max = values[i]
                    best_num_components = i
            # Return the best model
            return self.base_model(best_num_components)
            # Error handle in case model does not return a potential match
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        raise NotImplementedError

class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds
    This function is aiming to determine the ideal number of components in the HMM
    To do this, for each number of components (with large enough sample size)
    We do a cross-validation fold to split Training and Testing data and calculate an average model score
    We then review the log Likelihood score for each and return the best model
    '''
    def select(self):
        # This is wrapped in a try catch in case the model fails to determine a potential match
        try:
            split_method = KFold()
            min_max = range(self.min_n_components, self.max_n_components+1)
            # This is how I track the average log Likelihood for each num components
            values = {i:0 for i in min_max}
            # For each potential number of components do the following...
            for i in min_max:
                tot = 0
                count = 0
                # Only will do cross-validation if there are enough samples
                if len(self.sequences) > 2:
                    # Run the cross-validation for each k-fold split
                    for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                        self.X, self.lengths = combine_sequences(cv_train_idx,self.sequences)
                        test_X, test_lengths = combine_sequences(cv_test_idx,self.sequences)
                        hmm_model = self.base_model(i)
                        tot += hmm_model.score(test_X, test_lengths)
                        count += 1
                    # calculates log Likelihood given the model scores for each k fold at this number of components
                    values[i] = float(tot) / float(count)
                # If not enough samples for cross-validation just run the score on the samples
                else:
                    hmm_model = self.base_model(i)
                    values[i] = hmm_model.score(self.X, self.lengths)
                #print("For I = {} states; Score = {}".format(i, values[i]))
        # Error handle in case model does not return a potential match
        except:
            pass

        # This section determines which number of components has the highest average log Likelihood
        best_num_components = self.min_n_components
        is_max = values[self.min_n_components]
        for i in min_max:
            if values[i] > is_max:
                is_max = values[i]
                best_num_components = i
        # Return the best model
        return self.base_model(best_num_components)
