# classify and boosting

from sklearn import svm
from sklearn import tree

# Boosting imports
from sklearn.ensemble import AdaBoostClassifier


def get_model(training_data: list, model_name: str):
    """
    Takes training data as input, and returns a trained sklearn model
    Args:
        training_data: list of lists, [0] is feature vectors and [1] is gold labels
        model_name: string name of model type
    Returns:
        trained model
    """
    if model_name == 'svm':
        clf = svm.SVC(random_state=0)
        clf.fit(training_data[0], training_data[1])
        return clf

    if model_name == 'svmboost':
        X = training_data[0]
        Y = training_data[1]

        AdaBoost = AdaBoostClassifier(svm.SVC(probability=True, kernel='linear'), n_estimators=50, learning_rate=1.0, algorithm='SAMME')
        clf = AdaBoost.fit(X, Y)
        return clf


    if model_name == 'dt':
        clf = tree.DecisionTreeClassifier(random_state=0)
        clf = clf.fit(training_data[0], training_data[1])
        return clf

    if model_name == 'dtboost':
        X = training_data[0]
        Y = training_data[1]

        model = tree.DecisionTreeClassifier(random_state=0, criterion='entropy', max_depth=1)
        AdaBoost = AdaBoostClassifier(base_estimator=model, n_estimators=200, learning_rate=1)
        clf = AdaBoost.fit(X, Y)
        return clf

def train_and_classify(formatted_data: list, model_name: str):
    """
    Takes datasets and model name as input, returns predictions on the development data
    Args:
        datasets: list of lists of feature vectors and labels
        model_name: string name of classifier model type
    Returns:
        predictions
    """
    model = get_model(formatted_data[0], model_name)
    predictions = model.predict(formatted_data[1][0])
    return predictions

