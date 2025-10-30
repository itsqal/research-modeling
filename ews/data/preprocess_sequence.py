import pandas as pd

def preprocess_sequence(df: pd.DataFrame, sequence_column: str, sep=',') -> list:
    """
    Preprocesses the input DataFrame to extract sequences for pattern mining.

    Args:
        df (pd.DataFrame): Input DataFrame containing sequences.
        sequence_column (str): Name of the column containing sequences.

    Returns:
        list: A list of sequences, where each sequence is represented as a list of items.
    """

    df[sequence_column] = (
        df[sequence_column]
        .astype(str)
        .str.replace(" ", "", regex=False)
        .str.split(sep)
    )

    sequences = df[sequence_column].tolist()
    return sequences