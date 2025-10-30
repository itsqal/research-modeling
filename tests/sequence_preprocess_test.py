import pandas as pd
from ews.data.preprocess_sequence import preprocess_sequence
from ews.spm.prefixspan_runner import SequenceMiner

df = pd.read_csv('./ews/data/dataset/draft_1_cp_2.csv', sep=',')

# Applying preprocessing function to retrn list of sequences
sequences = preprocess_sequence(df, sequence_column='natural_signs_used', sep=',')

# initialize SequenceMiner to apply prefixspan alogrithm
miner = SequenceMiner(sequences=sequences)

miner.set_min_support(0.01)
miner.set_min_len(2)

df_patterns = miner.fit()

# Apply mean probability count column for patterns

df_patterns = miner.apply_mean_prob_column(df, df_patterns, prob_column='disaster_percentage', sequence_column='natural_signs_used')

df_patterns.to_csv('./tests/sequence_mining_output.csv', index=False)