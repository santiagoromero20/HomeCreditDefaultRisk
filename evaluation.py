import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics


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
    
    print("The",classifier,"performance metrics:")
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
        