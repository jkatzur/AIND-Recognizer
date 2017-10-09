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

    Similar to DIC model - the model is trained elsewhere and this is just running it on test data
    The structure here is to run through each test word and compare the results for each of the word-models
    Then, we identify which word-model scores the best, and make our guess that it's that word
    """
    probabilities = []
    guesses = []
    # Copied this structure from my DIC implementation - it's a similar paradigm
    # Used the same for structure I got from https://discussions.udacity.com/t/dic-always-giving-me-15-states/378722
    for word, (testX, testLengths) in test_set.get_all_Xlengths().items():
        is_best = float("-inf")
        prob_dict = {}
        # This section runs through all the trained words and model pairs to see which word matches the test word the best
        for (trained, model) in models.items():
            # This was tricky. I only figured out that I needed the try except from the forum here -> https://discussions.udacity.com/t/failure-in-recognizer-unit-tests/240082/5?u=cleyton_messias
            try:
                test_prof = model.score(testX,testLengths)
                prob_dict[trained] = test_prof
            except:
                prob_dict[trained] = float("-inf")
            # this structure is just a normal pick the max value calculation
            if test_prof > is_best:
                is_best = test_prof
                best_word = trained
        probabilities.append(prob_dict)
        guesses.append(best_word)
    return probabilities, guesses
