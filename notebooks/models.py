import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.model_selection import RepeatedStratifiedKFold, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier


def encode(data, method="ordinal"):
    """Encode categorical columns in the dataset.

    Parameters
    ----------
    data: dataframe
        The dataset to be encoded.

    method: str, {'onehot', 'ordinal'}, default='ordinal'
        The encoding method.

    Returns
    -------
    data:
        The encoded dataset.
    """
    assert method in (['ordinal', 'onehot'])
    if method == 'onehot':
        var_dummies = pd.get_dummies(data['XC'], prefix = 'X')
        data = pd.concat([data, var_dummies], axis=1)
        data.drop(columns=['XC'], inplace=True)
    if method == 'ordinal':
        data['XC'] = pd.factorize(data['XC'], sort=True)[0]
    
    return data


def model_init(model_type, **params):
    """ Initialize a model according to the model name.

    Parameters
    ----------
    model_type: {'LR', 'LDA', 'KNN', 'CART', 'NB', 'SVM', 'LGBM', \
        'XGBoost', 'CatBoost'}
        The type of the model to initialize.

    params: dictionary
        A dictionary of parameters for the specified model.

    Returns
    -------
    model:
        An initialized model.
    """

    model_candidates = ['LR', 'LDA', 'KNN', 'CART', 'NB', 'SVM', 'LGBM', \
        'XGBoost', 'CatBoost']
    if model_type not in model_candidates:
        raise ValueError(f"Model should be one of {model_candidates}.\n")

    if model_type == 'LR':
        model = LogisticRegression(**params)
    elif model_type == 'LDA':
        model = LinearDiscriminantAnalysis(**params)
    elif model_type == 'KNN': 
        model = KNeighborsClassifier(**params)
    elif model_type == 'CART': 
        model = DecisionTreeClassifier(**params)
    elif model_type == 'NB': 
        model = GaussianNB(**params)
    elif model_type == 'SVM':
        model = LinearSVC(**params)
    elif model_type == 'LGBM':
        model = LGBMClassifier(**params)
    elif model_type == 'XGBoost':
        model = XGBClassifier(**params)
    elif model_type == 'CatBoost':
        model = CatBoostClassifier(**params)
    
    return model


def model_eval(X, y, model_type, scoring, seed, **params):
    """ Evaluate a single model using cross-validation.

    The cross validation used here is RepeatedStratifiedKFold. Refer to \
        https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RepeatedStratifiedKFold.html \
        for more details.

    Parameters
    ----------
    X: ndarray of shape (n, dx)
        The predicting feature matrix.
    
    y: ndarray of shape (n, )
        The response.

    model_type: {'LR', 'LDA', 'KNN', 'CART', 'NB', 'SVM', 'LGBM', \
        'XGBoost', 'CatBoost'}
        The type of the model to evaluate.
    
    scoring: {'accuracy', 'f1', 'f1_weighted', 'precision', 'recall', \
        'roc_auc', 'neg_log_loss', 'neg_brier_score'}
        The scoring metric to evaluate in cv.
    
    seed: int
        The seed to assure the same cross validation fold splitting.
    
    params: dictionary
        The parameter specification for the model.

    Returns
    -------
    No returns. Print out the mean and stanndard error of the cv results.

    References
    ----------
    https://scikit-learn.org/stable/modules/model_evaluation.html
    """
    model = model_init(model_type, **params)
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=seed)
    cv_results = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    msg = "%s: %f (%f)" % (model_type, cv_results.mean(), cv_results.std())
    print(msg)


def model_compare(X, y, model_types, model_params, model_names, scoring, \
    image_dir='../images', display=True, seed=99):
    """ Compare multiple models in terms of scoring, using cross validation.
    
    The cross validation used here is RepeatedStratifiedKFold. 
    
    Parameters
    ----------
    X: ndarray of shape (n, dx)
        The predicting feature matrix.
    
    y: ndarray of shape (n, )
        The response.
        
    model_types: list of str
        A list of model types. Refer to model_init() to see candidate models.
    
    model_params: list of dictionary
        A list of model parameters, and model parameters are specified by dictionaries.
        Should be of the same length as model_types.

    model_names: list of str
        A list of model names, corresponding to the models in model_types.
        Should be of the same length as model_types and model_params.

    scoring: a dictionary of multiple scoring metrics.
        Example: scoring = {'auc': 'roc_auc', 'accuracy': 'accuracy'}
        The scoring metrics to evaluate in cv.
    
    image_dir: a str
        The path to save the generated images.

    display: bool
        Whether or not to display the figures.

    seed: int
        The seed to assure the same cross validation fold splitting.
        
    Returns
    -------
    results: a list of dictionary
        Each element is a dictionary containing the cross validation results.
    
    compare_df: a dataframe
        Columns represent the metrics and indices correspond to model names.
        Each entry is a string containing the mean and standard deviation of the cv results.
    
    compare_plot_df: a dataframe
        Columns represent model names and indices are the metrics.
        Each entry is the mean of the corresponding cv results.
        This dataframe is used for generating comparison plots.

    A model comparison diagram of model comparison would be displayed and saved to \
        the image folder.
    With respect to each scoring metric, one boxplot comparison diagram would be \
        displayed and saved to the image folder.
    """
    results = []
    compare_df = pd.DataFrame(columns = list(scoring.keys()), \
        index = model_names)
    # create compare_plot_df for plotting
    compare_plot_df = pd.DataFrame(columns = model_names, \
        index = list(scoring.keys()))

    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=seed)
  
    for model_type, params, model_name in zip(model_types, model_params, model_names):
        print(f"Running {model_name}...", end='\t')
        model = model_init(model_type, **params)
        cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring)
        results.append(cv_results)
        cv_metrics = [f"{cv_results[key].mean(): .5f} ({cv_results[key].std(): .5f})"\
            for key in cv_results.keys() if key.startswith('test')]
        compare_df.loc[model_name] = cv_metrics

        compare_plot_df[model_name] = [cv_results[key].mean()\
            for key in cv_results.keys() if key.startswith('test')]
        print(f"{model_name} done!")
    
    # plot the model comparison results using mean metrics
    plt.rcParams["figure.figsize"] = (10,8)
    plt.figure()
    compare_plot_df.plot()
    plt.legend(loc='best')
    plt.savefig(f"{image_dir}/model_compare.pdf")
    print(f"Model Comparison Diagram")
    print(f"It has been saved to '{image_dir}/model_compare.pdf'.")
    if display:
        plt.show()
    else:
        plt.close()
    
    # plot boxplot comparison diagrams with respect to each metric
    for key in scoring.keys():
        fig = plt.figure()
        fig.suptitle(f'Model Comparison w.r.t. {key}')
        ax = fig.add_subplot(111)
        plt.boxplot([results[i][f"test_{key}"] for i in range(len(results))])
        ax.set_xticklabels(model_names)
        plt.savefig(f"{image_dir}/model_compare_{key}.pdf")
        print(f"Boxplot for algorithm comparison in terms of {key}.")
        print(f"It has been saved to '{image_dir}/model_compare_{key}.pdf'.")
        if display:
            plt.show()
        else:
            plt.close()
            
    return results, compare_df, compare_plot_df

