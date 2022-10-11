import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
import time

#---------------------------------------------LEARNING CURVES AND TUNNING---------------------------------------------#

"""Auxiliaries"""

def calculating_error(dict, key_error):
    error_array = -dict[str(key_error)]
    error = np.mean(error_array)
    return error

def get_errors(grid_results):
    results = pd.DataFrame.from_dict(grid_results)
    results = results.sort_values(by=["mean_test_score"], ascending=False)
    results = results.head().reset_index()
    error_cv = -results.loc[0, "mean_test_score"]
    error_tr = -results.loc[0, "mean_train_score"]
    print(f"Training Log Loss {error_tr:0.2f}, CV Log Loss {error_cv:0.2f}")
    print('Train/Validation: {}'.format(round(error_cv/error_tr, 1)))

def grid_search(estimator, parameters, cv, scoring, X_train, y_train):
    grid = GridSearchCV(estimator, param_grid=parameters, cv=cv, scoring=str(scoring), return_train_score=True)
    inicio = time.time()
    grid.fit(X_train, y_train)
    fin = time.time()
    print("The time it takes to fit the model is",round(fin-inicio),"seconds.")
    print("Best params: "+str(grid.best_params_))
    return grid

#---------------------------------------------MODEL TEST PERFORMANCES---------------------------------------------#


def make_predictions(classifier, X_test, threshold):
    prob = classifier.predict_proba(X_test)
    pos_prob = prob[:, 1]
    pos_prob.tolist()
    predictions = []
    for i in range(len(pos_prob)):
        if pos_prob[i] >= threshold:
            predictions.append(1)
        else:
            predictions.append(0)
    
    return predictions


def get_performance(predictions, y_test, classifier):
    accuracy = metrics.accuracy_score(y_test, predictions)
    precision = metrics.precision_score(y_test, predictions)
    recall = metrics.recall_score(y_test, predictions)
    f1_score = metrics.f1_score(y_test, predictions)
    
    report = metrics.classification_report(y_test, predictions)
    
    cm = metrics.confusion_matrix(y_test, predictions)
    cm_as_dataframe = pd.DataFrame(data=cm)
    
    print("Performance metrics:")
    print('-'*30)
    print('Accuracy:', accuracy)
    print('Precision:', precision)
    print('Recall:', recall)
    print('F1 Score:', f1_score)
    print('\nModel Classification report:')
    print('-'*30)
    print(report)
    print('\nPrediction Confusion Matrix:')
    print('-'*30)
    print(cm_as_dataframe)
    


def plot_roc(model, y_test, features):

    prob = model.predict_proba(features)
    y_score = prob[:, prob.shape[1]-1] 
    fpr, tpr, _ = metrics.roc_curve(y_test, y_score) 
    roc_auc = metrics.auc(fpr, tpr)

    plt.figure(figsize=(10, 5))
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc})', linewidth=2.5)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

    return roc_auc


def compare_models(classifier, X_train, y_train, X_test, y_test, threshold):
    classifier.fit(X_train, y_train)
    predictions = make_predictions(classifier, X_test, threshold)
    get_performance(predictions, y_test, classifier)
    plot_roc(classifier, y_test, X_test)
        