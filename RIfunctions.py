import pandas as pd


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
