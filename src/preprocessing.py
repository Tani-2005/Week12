import pandas as pd

def handle_missing(df):

    for col in df.columns:

        if df[col].dtype == "object":
            df[col].fillna(df[col].mode()[0], inplace=True)

        else:
            df[col].fillna(df[col].median(), inplace=True)

    return df


def encode_categorical(df):

    categorical_cols = df.select_dtypes(include="object").columns

    return pd.get_dummies(
        df,
        columns=categorical_cols,
        drop_first=True
    )

def align_features(input_df, feature_columns):
    """
    Align prediction dataframe with training feature columns.
    Missing columns added as 0.
    Extra columns removed.
    """

    for col in feature_columns:
        if col not in input_df:
            input_df[col] = 0

    input_df = input_df[feature_columns]

    return input_df

def align_features(input_df, feature_columns):
    """
    Align prediction dataframe columns with training columns.
    Missing columns → added as 0
    Extra columns → removed
    Column order → preserved
    """

    for col in feature_columns:
        if col not in input_df:
            input_df[col] = 0

    return input_df[feature_columns]