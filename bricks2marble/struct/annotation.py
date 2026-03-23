import warnings
from bisect import bisect_right
from collections import OrderedDict, defaultdict
from collections.abc import Iterator
from functools import reduce
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from .fasta import Sequence

_CODON_TABLE = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",

    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",

    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",

    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}
T_Label = Literal["cds", "intron", "intergenic"]
T_StrandLabel = tuple[T_Label, T_Label]


class CDS(BaseModel):
    """Object that represents a coding region. Indexing follows Python
    conventions.
    """

    start: int
    end: int

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CDS):
            raise NotImplementedError(
                f"Can only compare CDS to CDS, but got {type(other)}"
            )
        return self.start == other.start and self.end == other.end

    def __hash__(self) -> int:
        return hash((self.start, self.end))

    model_config = {"frozen": False}


class Transcript(BaseModel):
    """Object that represents a collection of coding region."""

    name: str
    sequence: str
    strand: Literal["+", "-"]
    cds: list[CDS]

    @property
    def start(self) -> int:
        return self.cds[0].start

    @property
    def end(self) -> int:
        return self.cds[-1].end

    def cds_length(self) -> int:
        """Returns the total length of all coding regions in the
        transcript combined.
        """
        return sum(self.cds_lengths())

    def cds_lengths(self) -> list[int]:
        """Returns a list of coding region lengths in the transcript."""
        return [c.end - c.start for c in self.cds]

    def intron_lengths(self) -> list[int]:
        """Returns a list of lengths of introns in the trancript."""
        return [
            self.cds[i+1].start - self.cds[i].end
            for i in range(len(self.cds)-1)
        ]

    def classify(self, position: int) -> T_Label | None:
        if position < self.start or position >= self.end:
            return None
        idx = bisect_right([b.start for b in self.cds], position) - 1
        if idx >= 0 and self.cds[idx].start <= position < self.cds[idx].end:
            return "cds"
        return "intron"

    def coding_sequence(self, sequence: Sequence) -> Sequence:
        """Returns the sequence of nucleotides that corresponds to the
        coding sequence in this transcript.
        """
        if self.sequence != sequence.name:
            raise KeyError(
                f"Transcript {self.name!r} is in sequence {self.sequence!r}, "
                "which does not match the name of the provided sequence "
                f"{sequence.name!r}."
            )

        joined = reduce(
            lambda x, y: x.join(y),
            [sequence.positions(c.start, c.end) for c in self.cds],
        )
        if self.strand == "-":
            joined = joined.complement(reverse=True)

        return joined

    def protein_sequence(
        self,
        sequence: Sequence,
        *,
        drop_terminal_stop: bool = True,
        require_multiple_of_three: bool = False,
    ) -> str:
        """Translate the coding region of the transcript into
        standard-code amino acids. Unknown/ambiguous codons are set to
        'X' and stop codons are set to '*'.

        Args:
            drop_terminal_stop (bool, optional): If true, removes a
                trailing '*' (common convention). Defaults to True.
            require_multiple_of_three (bool, optional): If true, raises
                a ValueError if `len(CDS) % 3 != 0`. Defaults to False.
        """
        cds = self.coding_sequence(sequence).string()
        if not cds: return ""

        cds_u = cds.upper().replace("U", "T")
        if len(cds_u) % 3 != 0:
            msg = (
                f"CDS length for transcript {self.name!r} is {len(cds_u)}, "
                "not a multiple of 3."
            )
            if require_multiple_of_three: raise ValueError(msg)
            warnings.warn(
                msg + " Truncating trailing nucleotides for translation."
            )
            cds_u = cds_u[:3*(len(cds_u)//3)]

        aa = []
        for i in range(0, len(cds_u), 3):
            codon = cds_u[i:i+3]
            # if any ambiguity or non-ACGT, emit X
            if any(b not in "ACGT" for b in codon):
                aa.append("X")
            else:
                aa.append(_CODON_TABLE.get(codon, "X"))

        prot = "".join(aa)
        if drop_terminal_stop and prot.endswith("*"):
            prot = prot[:-1]
        return prot

    @staticmethod
    def from_genepred_row(line: str) -> "Transcript":
        fields = line.rstrip("\n").split("\t")
        name = fields[0]
        sequence = fields[1]
        strand: Literal["+", "-"] = fields[2]  # type: ignore
        starts = [int(x) for x in fields[8].rstrip(",").split(",")]
        ends = [int(x) for x in fields[9].rstrip(",").split(",")]
        cds = sorted(
            [CDS(start=s, end=e) for s, e in zip(starts, ends)],
            key=lambda b: b.start,
        )
        return Transcript(name=name, sequence=sequence, strand=strand, cds=cds)

    def to_genepred_row(self) -> str:
        """Serialize to a single GenePred line."""
        starts = ",".join(str(b.start) for b in self.cds) + ","
        ends = ",".join(str(b.end) for b in self.cds) + ","
        return "\t".join([
            self.name,
            self.sequence,
            self.strand,
            str(self.start),
            str(self.end),
            str(self.start),
            str(self.end),
            str(len(self.cds)),
            starts,
            ends,
        ])

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Transcript):
            raise NotImplementedError(
                "Can only compare Transcript to Transcript, but got "
                f"{type(other)}"
            )
        return (
            self.sequence == other.sequence and
            self.strand == other.strand and
            self.cds == other.cds
        )

    def __hash__(self) -> int:
        return hash((
            self.sequence,
            self.strand,
            tuple((c.start, c.end) for c in self.cds),
        ))

    model_config = {"frozen": False}


class SequenceAnnotation:

    def __init__(self, sequence: str) -> None:
        self.sequence = sequence
        self._all: list[Transcript] = []
        self._fwd: list[Transcript] = []
        self._rev: list[Transcript] = []
        self._fwd_starts: list[int] = []
        self._rev_starts: list[int] = []
        self._fwd_dirty = False
        self._rev_dirty = False

    def add(self, tx: Transcript) -> None:
        """Add a new transcript to the annotation."""
        self._all.append(tx)
        if tx.strand == "+":
            self._fwd.append(tx)
            self._fwd_dirty = True
        else:
            self._rev.append(tx)
            self._rev_dirty = True

    def remove(self, tx: Transcript) -> None:
        """Remove an already existing transcript from the annotation."""
        try:
            self._all.remove(tx)
        except ValueError:
            raise KeyError(
                f"Transcript {tx.name!r} not found in {self.sequence}"
            )
        if tx.strand == "+":
            self._fwd.remove(tx)
            self._fwd_dirty = True
        else:
            self._rev.remove(tx)
            self._rev_dirty = True

    def _sort_strand(self, strand: str) -> None:
        if strand == "+":
            if not self._fwd_dirty: return
            self._fwd.sort(key=lambda t: t.start)
            self._fwd_starts = [t.start for t in self._fwd]
            self._fwd_dirty = False
        else:
            if not self._rev_dirty: return
            self._rev.sort(key=lambda t: t.start)
            self._rev_starts = [t.start for t in self._rev]
            self._rev_dirty = False

    def _candidates(self, position: int, strand: str) -> list[Transcript]:
        self._sort_strand(strand)
        transcripts, starts = (
            (self._fwd, self._fwd_starts) if strand == "+"
            else (self._rev, self._rev_starts)
        )
        right = bisect_right(starts, position)
        return [t for t in transcripts[:right] if t.end > position]

    def _classify_strand(self, position: int, strand: str) -> T_Label:
        for t in self._candidates(position, strand):
            label = t.classify(position)
            if label is not None: return label
        return "intergenic"

    def classify(self, position: int) -> T_StrandLabel:
        return (
            self._classify_strand(position, "+"),
            self._classify_strand(position, "-"),
        )

    def classify_range(
        self,
        start: int,
        end: int,
    ) -> tuple[
        dict[str, list[tuple[int, int]]],
        dict[str, list[tuple[int, int]]],
    ]:
        return (
            self._classify_range_strand(start, end, "+"),
            self._classify_range_strand(start, end, "-"),
        )

    def _classify_range_strand(
        self,
        start: int,
        end: int,
        strand: str,
    ) -> dict[str, list[tuple[int, int]]]:
        self._sort_strand(strand)
        transcripts, starts = (
            (self._fwd, self._fwd_starts) if strand == "+"
            else (self._rev, self._rev_starts)
        )

        breakpositions = {start, end}
        right = bisect_right(starts, end)
        for t in transcripts[:right]:
            if t.end < start: continue

            for b in t.cds:
                if start < b.start < end: breakpositions.add(b.start)
                if start < b.end < end: breakpositions.add(b.end)
            if start < t.start < end:
                breakpositions.add(t.start)
            if start < t.end < end:
                breakpositions.add(t.end)

        sorted_bp = sorted(breakpositions)
        result: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, s in enumerate(sorted_bp[:-1]):
            e = sorted_bp[i+1] if i+1 < len(sorted_bp) else end
            label = self._classify_strand(s, strand)
            result[label].append((s, e))
        return dict(result)

    def __iter__(self) -> Iterator[Transcript]:
        return iter(self._all)

    def __len__(self) -> int:
        return len(self._all)


class Annotation:
    """Representation of a genome annotation, designed to align with the
    GenePred file format.
    """

    def __init__(self) -> None:
        self._sequences: OrderedDict[str, SequenceAnnotation] = OrderedDict()

    def add(self, tx: Transcript) -> None:
        """Add a new transcript to the annotation."""
        if tx.sequence not in self:
            self._sequences[tx.sequence] = SequenceAnnotation(tx.sequence)
        self._sequences[tx.sequence].add(tx)

    def remove(self, tx: Transcript) -> None:
        """Remove an already existing transcript from the annotation."""
        self[tx.sequence].remove(tx)

    def classify(
        self,
        position: int,
        sequence: str | int = 0,
    ) -> T_StrandLabel:
        """Return a 2-tuple of labels "cds", "intron" or "intergenic"
        for forward and reverse strand at the given position and
        sequence. Positions start at 0 and are end-exclusive. For
        example, position 1 is the second nucleotide in the genome.
        """
        return self[sequence].classify(position)

    def classify_range(
        self,
        start: int,
        end: int,
        sequence: str | int = 0,
    ) -> tuple[
        dict[str, list[tuple[int, int]]],
        dict[str, list[tuple[int, int]]],
    ]:
        """Returns two dictionaries ``d`` for forward and reverse
        strand. Here, ``d[label]=[(s1, e1), ..., (sk, ek)]`` is a
        collection of regions that are classified as ``label`` ("cds",
        "intron" or "intergenic").
        The regions follow Python index conventions.
        """
        return self[sequence].classify_range(start, end)

    def classify_many(
        self,
        positions: list[int],
        sequence: str | int = 0,
    ) -> list[T_StrandLabel]:
        seq_ann = self[sequence]
        order = sorted(range(len(positions)), key=lambda i: positions[i])
        labels: list[T_StrandLabel] = [
            ("intergenic", "intergenic")
        ] * len(positions)
        for i in order: labels[i] = seq_ann.classify(positions[i])
        return labels

    def sequences(self) -> list[str]:
        return list(self._sequences.keys())

    @classmethod
    def from_genepred(cls, path: Path | str) -> "Annotation":
        ann = cls()
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"): continue
                ann.add(Transcript.from_genepred_row(line))
        return ann

    def to_genepred(self, path: Path | str, mode: str = "w") -> None:
        with open(path, mode) as fh:
            for seq_ann in self:
                for tx in seq_ann:
                    fh.write(tx.to_genepred_row() + "\n")

    def __iter__(self) -> Iterator[SequenceAnnotation]:
        return iter(self._sequences.values())

    def __contains__(self, sequence: str) -> bool:
        return sequence in self._sequences

    def __getitem__(self, sequence: str | int) -> SequenceAnnotation:
        if isinstance(sequence, int):
            return list(self._sequences.values())[sequence]
        if sequence not in self:
            raise KeyError(f"Sequence {sequence!r} not part of Annotation")
        return self._sequences[sequence]
