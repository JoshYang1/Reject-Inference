import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import itertools
from pytest import param
from sklearn.metrics import confusion_matrix, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import joblib
from sklearn.preprocessing import StandardScaler

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
            'max_features': ["log2", "sqrt"]
    }

    grid_rf = RandomizedSearchCV(estimator=model, param_distributions=params_rf ,cv = cv, scoring="neg_mean_squared_error", verbose = 1, n_jobs=-1)
    grid_rf.fit(X_train, y_train)

    best_hyperparams = grid_rf.best_params_
    best_rf = grid_rf.best_estimator_

    # Saving the model
    joblib.dump(best_rf, filename)
    
    return best_hyperparams, best_rf


# def AcceptedLoansPreprocessing(train_df, numerical_cols, test_df):
#     scaler = StandardScaler(copy=False)

#     # dataset['issue_d'] = pd.to_datetime(dataset['issue_d'])

#     # # Splitting dataset into train and test with newer instances 

#     # train_df = dataset.loc[dataset['issue_d'] < dataset['issue_d'].quantile(0.8)]
#     # test_df = dataset.loc[dataset['issue_d'] >= dataset['issue_d'].quantile(0.8)]

#     # train_df = train_df.drop('issue_d', axis=1)
#     # test_df = test_df.drop('issue_d', axis=1)

#     train_df[numerical_cols] = train_df[numerical_cols].fillna(train_df[numerical_cols].mean())
#     test_df[numerical_cols] = test_df[numerical_cols].fillna(test_df[numerical_cols].mean())

#     train_df[numerical_cols] = scaler.fit_transform(train_df[numerical_cols], train_df['charged_off'])
#     test_df[numerical_cols] = scaler.transform(test_df[numerical_cols])

#     return train_df, test_df


def AcceptedLoansSplit(train, test):
    y_train = train['charged_off']
    y_test = test['charged_off']

    X_train = train.drop('charged_off', axis=1)
    X_test = test.drop('charged_off', axis=1)

    return X_train, y_train, X_test, y_test

def AppendResults(df,title, AUC_scores, y_test, y_score) :

    # y_score_flag = [int(round(i)) for i in y_score]

    temp_df = pd.DataFrame()

    temp_df["Model"] = [title]
    temp_df["Train AUC Scores"] = [AUC_scores]
    temp_df["Test AUC"] = roc_auc_score(y_test, y_score)
    temp_df["Test Recall (1)"] = recall_score(y_test, y_score, pos_label=1)
    temp_df["Test Recall (0)"] = recall_score(y_test, y_score, pos_label=0)
    temp_df["Confusion Matrix"] = [confusion_matrix(y_test, y_score)]

    return pd.concat([df,temp_df])

def splitCount(column):

    neg, pos = np.bincount(column)
    total = neg + pos
    rate = 100 * pos / total

    return total, pos, rate

def RejectPreprocessing(df, num_cols, cat_cols, prefix, scaler) :
    df = pd.get_dummies(df, prefix=prefix, columns=cat_cols, drop_first=False)
    df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
    df[num_cols] = scaler.transform(df[num_cols])
    
    return df
