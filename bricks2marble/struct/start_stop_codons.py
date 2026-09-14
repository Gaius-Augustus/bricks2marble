START_CODONS = {
    1: [("ATG", 1), ],#[("TTG", 0.33), ("CTG", 0.33), ("ATG", 0.34), ]
    2: [("ATT", 0.2), ("ATC", 0.2), ("ATA", 0.2), ("ATG", 0.2), ("GTG", 0.2), ],
    3: [("ATA", 0.33), ("ATG", 0.34), ("GTG", 0.33), ],
    4: [("TTA", 0.125), ("TTG", 0.125), ("CTG", 0.125), ("ATT", 0.125), ("ATC", 0.125), ("ATA", 0.125), ("ATG", 0.125), ("GTG", 0.125), ],
    5: [("TTG", 0.166), ("ATT", 0.167), ("ATC", 0.167), ("ATA", 0.167), ("ATG", 0.167), ("GTG", 0.166), ],
    6: [("ATG", 1), ],
    9: [("ATG", 0.5), ("GTG", 0.5), ],
    10: [("ATG", 1), ],
    11: [("TTG", 0.143), ("CTG", 0.143), ("ATT", 0.142), ("ATC", 0.143), ("ATA", 0.143), ("ATG", 0.143), ("GTG", 0.143), ],
    12: [("CTG", 0.5), ("ATG", 0.5), ],
    13: [("TTG", 0.25), ("ATA", 0.25), ("ATG", 0.25), ("GTG", 0.25), ],
    14: [("ATG", 1), ],
    15: [("ATG", 1), ],
    16: [("ATG", 1), ],
    21: [("ATG", 0.5), ("GTG", 0.5), ],
    22: [("ATG", 1), ],
    23: [("ATT", 0.33), ("ATG", 0.34), ("GTG", 0.33), ],
    24: [("TTG", 0.25), ("CTG", 0.25), ("ATG", 0.25), ("GTG", 0.25), ],
    25: [("TTG", 0.33), ("ATG", 0.34), ("GTG", 0.33), ],
    26: [("CTG", 0.5), ("ATG", 0.5), ],
    27: [("ATG", 1), ],
    28: [("ATG", 1), ],
    29: [("ATG", 1), ],
    30: [("ATG", 1), ],
    31: [("ATG", 1), ],
    32: [("TTG", 0.143), ("CTG", 0.143), ("ATT", 0.143), ("ATC", 0.143), ("ATA", 0.143), ("ATG", 0.143), ("GTG", 0.142), ],
    33: [("TTG", 0.25), ("CTG", 0.25), ("ATG", 0.25), ("GTG", 0.25), ],
}

STOP_CODONS = {
    # some stop codons in tables 27, 28 and 31 can also translate to an amino acid according to the NCBI translation table,
    # which may result in unexpected behaviour.
    1: [("TAA", 0.33), ("TAG", 0.34), ("TGA", 0.33), ],
    2: [("TAA", 0.25), ("TAG", 0.25), ("AGA", 0.25), ("AGG", 0.25), ],
    3: [("TAA", 0.5), ("TAG", 0.5), ],
    4: [("TAA", 0.5), ("TAG", 0.5), ],
    5: [("TAA", 0.5), ("TAG", 0.5), ],
    6: [("TGA", 1), ],
    9: [("TAA", 0.5), ("TAG", 0.5), ],
    10: [("TAA", 0.5), ("TAG", 0.5), ],
    11: [("TAA", 0.33), ("TAG", 0.34), ("TGA", 0.33), ],
    12: [("TAA", 0.33), ("TAG", 0.34), ("TGA", 0.33), ],
    13: [("TAA", 0.5), ("TAG", 0.5), ],
    14: [("TAG", 1), ],
    15: [("TAA", 0.5), ("TGA", 0.5), ],
    16: [("TAA", 0.5), ("TGA", 0.5), ],
    21: [("TAA", 0.5), ("TAG", 0.5), ],
    22: [("TCA", 0.34), ("TAA", 0.33), ("TGA", 0.33), ],
    23: [("TTA", 0.25), ("TAA", 0.25), ("TAG", 0.25), ("TGA", 0.25), ],
    24: [("TAA", 0.5), ("TAG", 0.5), ],
    25: [("TAA", 0.5), ("TAG", 0.5), ],
    26: [("TAA", 0.33), ("TAG", 0.34), ("TGA", 0.33), ],
    27: [("TGA", 1), ],
    28: [("TAA", 0.33), ("TAG", 0.34), ("TGA", 0.33), ],
    29: [("TGA", 1), ],
    30: [("TGA", 1), ],
    31: [("TAA", 0.5), ("TAG", 0.5), ],
    32: [("TAA", 0.5), ("TGA", 0.5), ],
    33: [("TAG", 1), ],
}

def get_start_codons(
    translation_table: int = 1,
) -> list[tuple[str, float]]:
    if translation_table not in START_CODONS:
        raise ValueError(f"Unknown translation table: {translation_table}")
    start_codons = START_CODONS[translation_table]
    return start_codons

def get_stop_codons(
    translation_table: int = 1,
) -> list[tuple[str, float]]:
    if translation_table not in STOP_CODONS:
        raise ValueError(f"Unknown translation table: {translation_table}")
    stop_codons = STOP_CODONS[translation_table]
    return stop_codons
