import pandas as pd

def filter_patterns(patterns: pd.DataFrame, min_average_prob: float, prob_column: str = 'mean_prob'):
    filtered_patterns = patterns[patterns['probability'] >= min_prob].reset_index(drop=True)
    return filtered_patterns