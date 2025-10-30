def contains_pattern(sequence: list, pattern, sep: str =','):
    seq_str = sep.join(sequence)
    return all(p in seq_str for p in pattern)