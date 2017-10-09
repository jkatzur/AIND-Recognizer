import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
    """
    probabilities = []
    guesses = []
    for word, (testX, testLengths) in test_set.get_all_Xlengths().items():
        is_best = float("-inf")
        prob_dict = {}
        for (trained, model) in models.items():
            # Set this up due to -> https://discussions.udacity.com/t/failure-in-recognizer-unit-tests/240082/5?u=cleyton_messias
            try:
                test_prof = model.score(testX,testLengths)
                prob_dict[trained] = test_prof
            except:
                prob_dict[trained] = float("-inf")
            if test_prof > is_best:
                is_best = test_prof
                best_word = trained
        probabilities.append(prob_dict)
        guesses.append(best_word)
    return probabilities, guesses
