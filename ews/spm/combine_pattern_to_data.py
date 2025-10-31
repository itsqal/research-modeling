import pandas as pd
from ews.utils.helper import contains_pattern

def combine_pattern_to_data(df: pd.DataFrame, patterns: pd.DataFrame, sequence_column: str, pattern_column: str, sep: str =',') -> pd.DataFrame:
    candidate_patterns = patterns[pattern_column].tolist()

    for pat in candidate_patterns:
        pat_tokens = pat.split(sep)
        df[f"has_{pat.replace(sep,'_')}"] = df[sequence_column].apply(lambda seq: contains_pattern(seq, pat_tokens)).astype(int)

    return df