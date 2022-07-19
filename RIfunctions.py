def purposeCleaning(dataframe):
    counts = dataframe['purpose'].value_counts()
    keep_list = counts[counts > 15000].index
    df = dataframe[dataframe['purpose'].isin(keep_list)]

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