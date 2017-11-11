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
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", message="divide by zero encountered in log")

        # TODO implement model selection based on BIC scores
        best_bic = None
        best_state = self.n_constant

        for n_state in range(self.min_n_components, self.max_n_components + 1):
            try:
                hmm_model = self.base_model(n_state)
                d = len(self.X[0])
                p = n_state ** 2 + (2 * d * n_state) - 1
                logL = hmm_model.score(self.X, self.lengths)
                bic = -2 * logL + p * np.log(n_state)
                if best_bic is None or best_bic > bic:
                    best_bic = bic
                    best_state = n_state
            except:

                bic = float('-inf')

        return self.base_model(best_state)



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
        warnings.filterwarnings("ignore", message="divide by zero encountered in log")

        # TODO implement model selection based on DIC scores
        best_dic = None
        best_state = self.n_constant

        other_words = list(self.words)
        other_words.remove(self.this_word)

        for n_state in range(self.min_n_components, self.max_n_components + 1):
            try:
                hmm_model = self.base_model(n_state)
                logL_word = hmm_model.score(self.X, self.lengths)
                other_scores = 0.0
                for word in other_words:
                    X, lengths = self.hwords[word]
                    other_scores += hmm_model.score(X, lengths)
                    m = len(self.words) - 1
                dic = logL_word - (other_scores / m)
                if best_dic is None or best_dic < dic:
                    best_dic = dic
                    best_state = n_state
            except:
                pass

        return self.base_model(best_state)


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", message="divide by zero encountered in log")

        # TODO implement model selection using CV
        best_score = None
        best_state = self.n_constant

        for n_state in range(self.min_n_components, self.max_n_components + 1):
            try:
                scores = []
                score = 0.0

                if (len(self.sequences) > 1):

                    folds = min(len(self.sequences), 3)
                    kf = KFold(shuffle=True, n_splits=folds)

                    for cv_train_idx, cv_test_idx in kf.split(self.sequences):
                        test_scores = []
                        self.X, self.lengths = combine_sequences(cv_train_idx, self.sequences)
                        X_test, lengths_test = combine_sequences(cv_test_idx, self.sequences)

                        model = GaussianHMM(n_components=n_state,
                                            covariance_type="diag",
                                            n_iter=1000,
                                            random_state=self.random_state,
                                            verbose=False).fit(X_train, lengths_train)

                        test_scores.append(model.score(X_test, lengths_test))

                    score = scores.append(np.mean(test_scores))

                else:
                    score = scores.append(np.mean(model.score(self.X, self.lengths)))

                if best_score is None or best_score < score:
                    best_score = score
                    best_state = n_state

            except:
                pass

        return self.base_model(best_state)
