from math import sqrt

from sklearn.metrics import mean_squared_error


def rmse(y_true, y_pred):
    return sqrt(mean_squared_error(y_true, y_pred))


def update_best_score(dataset_type, current_score, best_score):
    '''
    Compare the best score and current score
    Regression: rmse the lower the better
    Classification: auc-roc the higher the better

    return: better score
    '''
    is_better = (dataset_type == 'regression' and current_score < best_score) or \
                (dataset_type == 'classification' and current_score > best_score)
    if is_better:
        return current_score
    return best_score