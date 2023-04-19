from sklearn import svm
from sklearn import tree


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
        clf = svm.SVC()
        clf.fit(training_data[0], training_data[1])
        return clf

    if model_name == 'dt':
        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(training_data[0], training_data[1])
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

