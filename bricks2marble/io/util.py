import math
import mmap
from functools import reduce
from pathlib import Path


def index(
    fasta: Path | str,
    sort_reverse: bool | None = None,
) -> list[tuple[str, int, int, int]]:
    """Returns the names of all sequences and their offsets and lengths
    within the Fasta file.

    Args:
        fasta (Path): Path to the fasta file to index.
        sort_reverse (bool, optional): Whether to additionally sort the
            returned indices by sequence size. If true, sorts in
            descending order and if false, sorts in ascending order.
            Defaults to no sorting.
    """
    index = []

    with open(fasta, "rb") as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:

            size = mm.size()
            pos = 0

            first_header = mm.find(b">", pos)
            if first_header == -1: return []
            pos = first_header
            while pos < size:
                header_start = pos

                header_end = mm.find(b"\n", header_start)
                if header_end == -1: break
                name = mm[header_start+1:header_end].strip().decode()
                seq_start = header_end + 1

                next_header = mm.find(b"\n>", seq_start)
                if next_header == -1:
                    seq_end = size
                    pos = size
                else:
                    seq_end = next_header + 1
                    pos = next_header + 1

                block = mm[seq_start:seq_end]
                seq_length = len(block) - block.count(b"\n")

                index.append((name, seq_start, seq_end, seq_length))

    if sort_reverse is not None:
        index.sort(key=lambda x: x[3], reverse=sort_reverse)
    return index


def largest_close_to_divisible_by(L: int, factors: list[int]) -> int:
    M = reduce(math.lcm, factors)
    if M > L or M == 0: raise ValueError(
        f"Did not find an appropriate multiple <{L} of numbers {factors}."
    )
    return (1 + (L - 1) // M) * M
