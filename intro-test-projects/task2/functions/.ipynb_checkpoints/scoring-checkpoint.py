#### SCORER ###

def scorer(data: pd.DataFrame, model, splitter: float) -> tuple:
    """Fits a given model to a given dataset and 
    returns the score on both training and testing as a tuple."""

    # Find the row to split dataset in training and testing set
    split = int(data.shape[0] * splitter)

    # Sclice the dataframe accordingly
    features_train = data.iloc[1:split,:-1]
    labels_train = data.iloc[1:split,-1]

    features_test = data.iloc[split:,:-1]
    labels_test = data.iloc[split:,-1]

    # Fit the model onto training data
    model.fit(features_train, labels_train)

    # Compute score for both training and testing
    training_score = model.score(features_train, labels_train)
    test_score = model.score(features_test, labels_test)

    return (training_score, test_score)




### SCOREBOARD ###

def scoreboard(models: list, data: pd.DataFrame, splitter: float) -> dict:
    """Takes a list of several models and runs them on the same dataset.
    Training and Test score are written in a dictionary."""

    # Initializing output dict
    scores = dict()

    # Scoring of the models
    for m in models:
        scores[m] = scorer(data, m, splitter)

    return scores