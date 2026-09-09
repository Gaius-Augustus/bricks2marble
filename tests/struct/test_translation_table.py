"""Tests for :mod:`bricks2marble.struct.start_stop_codons` and :mod:`bricks2marble.struct.create_codon_table`."""

import pytest 

from bricks2marble.struct.start_stop_codons import (get_start_codons, get_stop_codons, START_CODONS, STOP_CODONS)
from bricks2marble.struct.create_codon_table import create_codon_table
from bricks2marble.tf.hmm.layer import AnnotationHMM

def test_valid_start_stop_distributions():
    assert set(START_CODONS) == set(STOP_CODONS)

    for translation_table in START_CODONS:
        start_codons = START_CODONS[translation_table]
        stop_codons = STOP_CODONS[translation_table]

        assert sum(prob for _, prob in start_codons) == pytest.approx(1.0)
        assert sum(prob for _, prob in stop_codons) == pytest.approx(1.0)

def test_translation_table_sets_start_stop_codons():
    hmm = AnnotationHMM(translation_table = 6)

    assert hmm.config.start_codons == START_CODONS[6]
    assert hmm.config.stop_codons == STOP_CODONS[6]

def test_table_1_equals_default():
    default = AnnotationHMM()
    table1 = AnnotationHMM(translation_table = 1)
    assert set(default.config.start_codons) == set(table1.config.start_codons)
    assert set(default.config.stop_codons) == set(table1.config.stop_codons)

def test_stop_codons_match_codon_table():
    for translation_table in STOP_CODONS:
        if translation_table in {27, 28, 31}:
            continue

        assert {
            codon for codon, _ in get_stop_codons(translation_table)
        } == {
            codon
            for codon, amino_acid
            in create_codon_table(translation_table).items()
            if amino_acid == "*"
        }
