from prefixspan import PrefixSpan
import pandas as pd

from ews.utils.helper import contains_pattern

class SequenceMiner:
    def __init__(self, min_support: int =2, min_len: int =2, sequences=None):
        self.min_support = min_support
        self.min_len = min_len
        self.sequences = sequences if sequences is not None else []

    def fit(self) -> 'pd.DataFrame':
        """
        Apllies the PrefixSpan algorithm to mine frequent sequential patterns from the input sequences.
        """
        ps = PrefixSpan(self.sequences)
        ps.minlen = self.min_len

        frequent_patterns = ps.frequent(self.min_support)

        patterns_data = [
            { 'support': sup, 'pattern': ','.join(seq)} for sup, seq, in frequent_patterns
        ]

        df_patterns = pd.DataFrame(patterns_data)
        df_patterns["support_ratio"] = df_patterns["support"] / len(self.sequences)

        return df_patterns

    def set_min_support(self, min_support_percentage):
        total_sequences = len(self.sequences)
        self.min_support = int(min_support_percentage * total_sequences)

    def set_min_len(self, min_len):
        self.min_len = min_len

    def apply_mean_prob_column(self, df: pd.DataFrame, df_pattern: pd.DataFrame, prob_column: str = 'probability', sequence_column: str = 'sequence', sep: str = ','):
        pattern_stats = []

        for _, row in df_pattern.iterrows():
            pat = row["pattern"].split(sep)
            subset = df[df[sequence_column].apply(lambda seq: contains_pattern(seq, pat))]
            
            mean_prob = subset[prob_column].mean() if not subset.empty else None
            pattern_stats.append(mean_prob)

        df_pattern["mean_prob"] = pattern_stats
        df_pattern.dropna(inplace=True)
        
        return df_pattern