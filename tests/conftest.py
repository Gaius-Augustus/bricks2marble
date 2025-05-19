from pathlib import Path

import pytest


@pytest.fixture
def fasta_path() -> Path:
    return Path(__file__).parent / "example.fa"


@pytest.fixture
def gtf_path() -> Path:
    return Path(__file__).parent / "example.gtf"
