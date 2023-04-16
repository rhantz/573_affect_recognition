from sklearn import svm


def get_model(x_list, y_list):
    clf = svm.SVC()
    clf.fit(x_list, y_list)
    return clf


def get_predictions(model, dev_data):
    predictions = model.predict(dev_data)
    return predictions
