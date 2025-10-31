def contains_pattern(sequence, pattern, sep: str =','):
    pattern_tokens = [str(p).strip() for p in pattern]

    if isinstance(sequence, str):
        seq_tokens = [s.strip() for s in sequence.split(sep) if s.strip() != '']
    else:
        seq_tokens = [str(s).strip() for s in sequence]

    seq_set = set(seq_tokens)
    return all(tok in seq_set for tok in pattern_tokens)