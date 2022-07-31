import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import itertools
from sklearn.model_selection import GridSearchCV
import joblib

def purposeCleaning(dataframe):
    df = dataframe.loc[dataframe['purpose'].isnull() == False]

    counts = df['purpose'].value_counts()
    keep_list = counts[counts > 15000].index
    df = df[df['purpose'].isin(keep_list)]

    to_replace = {
    'Debt consolidation': 'debt_consolidation',
    'Home improvement': 'home_improvement',
    'Credit card refinancing': 'credit_card',
    'Other': 'other',
    'Vacation': 'vacation',
    'Medical expenses': 'medical',
    'Car financing': 'car',
    'Major purchase': 'major_purchase',
    'Moving and relocation': 'moving',
    'Home buying': 'house'
    }

    df['purpose'] = df['purpose'].replace(to_replace)

    return df

def balanceData(dataframe):
    ones = dataframe[dataframe['charged_off'] == 1]

    zeroes = dataframe[dataframe['charged_off'] == 0]

    if zeroes.shape[0] > ones.shape[0]:
        keep_0s = zeroes.sample(frac=ones.shape[0]/zeroes.shape[0], random_state = 1)
        dataframe = pd.concat([keep_0s,ones],axis=0)
    else:
        keep_1s = ones.sample(frac=zeroes.shape[0]/ones.shape[0], random_state = 1)
        dataframe = pd.concat([keep_1s,zeroes],axis=0)

    return dataframe
    
# This function prints and plots the confusion matrix.
# Normalization can be applied by setting `normalize=True`.    

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def RandomForestTuning(model, X_train, y_train, filename, cv=10) :

    params_rf ={
            'n_estimators': [10, 50, 100, 200],
            'criterion': ["gini", "entropy"],
            'max_depth': [5,10,12,15,20],
            'max_features': ["log2", "sqrt"],
            'bootstrap':[True, False]
    }

    grid_rf = GridSearchCV(estimator=model, param_grid=params_rf ,cv = cv, scoring="neg_mean_squared_error", verbose = 1, n_jobs=-1)
    grid_rf.fit(X_train, y_train)

    best_hyperparams = grid_rf.best_params_
    print("Best hyperparameters: \n", best_hyperparams )
    best_rf = grid_rf.best_estimator_

    # Saving the model

    joblib.dump(best_rf, filename)
    