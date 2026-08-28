import textwrap
import warnings
from bisect import bisect_right
from collections import OrderedDict, defaultdict
from collections.abc import Iterator
from functools import reduce
from pathlib import Path
from typing import Callable, Literal

from pydantic import BaseModel

from .fasta import Fasta, Sequence
from .create_codon_table import create_codon_table

T_Label = Literal["cds", "intron", "intergenic"]
T_StrandLabel = tuple[T_Label, T_Label]

_DEFAULT_GENE_PATTERN = "g{gene_id}"
_DEFAULT_TX_PATTERN = "g{gene_id}_t{transcript_id}"


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
    """Object that represents a collection of coding regions.

    ``name`` is optional and is the only identifier stored on the
    object. When it is not given, the identifier written to a file is
    generated at write time by counting genes and isoforms in order
    (see :meth:`display_name`); no index is kept on the transcript.
    """

    name: str | None = None
    sequence: str
    strand: Literal["+", "-"]
    cds: list[CDS]

    def display_name(
        self,
        gene_id: int,
        transcript_id: int,
        pattern: str = _DEFAULT_TX_PATTERN,
    ) -> str:
        """Return the identifier used for this transcript in output
        files: the explicit ``name`` if set, otherwise ``pattern``
        formatted with the given ``gene_id`` and ``transcript_id``
        counters.
        """
        if self.name is not None:
            return self.name
        return pattern.format(gene_id=gene_id, transcript_id=transcript_id)

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
        translation_table: int | None = None,
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
        if translation_table is None:
            translation_table = 1
        _CODON_TABLE = create_codon_table(translation_table)
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

    def to_genepred_row(self, name: str | None = None) -> str:
        """Serialize to a single GenePred line. ``name`` overrides the
        identifier written in the first column; if omitted, the
        transcript's own ``name`` (or a placeholder) is used.
        """
        if name is None:
            name = self.name if self.name is not None else "transcript"
        starts = ",".join(str(b.start) for b in self.cds) + ","
        ends = ",".join(str(b.end) for b in self.cds) + ","
        return "\t".join([
            name,
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

    def to_gtf_rows(
        self,
        source: str = "bricks2marble",
        gene_id: str | None = None,
        transcript_id: str | None = None,
    ) -> str:
        """Returns GTF rows for this transcript (transcript level and
        below). The enclosing 'gene' line is emitted by :class:`Gene`,
        so it is intentionally not produced here. Coordinates are
        converted from 0-based half-open to 1-based fully closed. Emits
        one line each for 'transcript', 'start_codon' and 'stop_codon',
        then for each block a 'CDS' and 'exon' line, and between
        consecutive blocks an 'intron' line. ``gene_id`` and
        ``transcript_id`` are resolved by the enclosing :class:`Gene`;
        standalone they fall back to this transcript's ``name``.
        """
        if transcript_id is None:
            transcript_id = (
                self.name if self.name is not None else "transcript"
            )
        if gene_id is None:
            gene_id = transcript_id
        attributes = f'gene_id "{gene_id}"; transcript_id "{transcript_id}";'

        rows = []
        rows.append("\t".join([
            self.sequence, source, "transcript",
            str(self.start+1), str(self.end),
            ".", self.strand, ".", attributes,
        ]))

        first = self.cds[0]
        last = self.cds[-1]
        if self.strand == "+":
            start_codon = (first.start+1, first.start+3)
            stop_codon  = (last.end-2, last.end)
        else:
            start_codon = (last.end-2, last.end)
            stop_codon  = (first.start+1, first.start+3)

        for feature, (s, e) in (
            ("start_codon", start_codon), ("stop_codon", stop_codon)
        ):
            rows.append("\t".join([
                self.sequence, source, feature,
                str(s), str(e),
                ".", self.strand, "0", attributes,
            ]))

        order = (
            range(len(self.cds)) if self.strand == "+"
            else range(len(self.cds) - 1, -1, -1)
        )
        frames = [0] * len(self.cds)
        bases = 0
        for i in order:
            frames[i] = (3 - (bases % 3)) % 3
            bases += self.cds[i].end - self.cds[i].start

        for i, block in enumerate(self.cds):
            frame = frames[i]
            for feature in ("CDS", "exon"):
                rows.append("\t".join([
                    self.sequence, source, feature,
                    str(block.start+1), str(block.end),
                    ".", self.strand, str(frame),
                    attributes,
                ]))

            if i < len(self.cds) - 1:
                intron_start = block.end + 1
                intron_end = self.cds[i+1].start
                rows.append("\t".join([
                    self.sequence, source, "intron",
                    str(intron_start), str(intron_end),
                    ".", self.strand, ".",
                    attributes,
                ]))

        return "\n".join(rows)

    def to_gff3_rows(
        self,
        source: str = "bricks2marble",
        gene_id: str | None = None,
        transcript_id: str | None = None,
    ) -> str:
        """Returns GFF3 rows for this transcript (the 'mRNA' feature and
        its 'CDS'/'exon' children). Coordinates are converted from
        0-based half-open to 1-based fully closed. ``transcript_id``
        defaults to the transcript's ``name``; if ``gene_id`` is given,
        the mRNA is linked to that gene through a ``Parent`` attribute;
        the enclosing 'gene' line itself is emitted by :class:`Gene`.
        """
        if transcript_id is None:
            transcript_id = (
                self.name if self.name is not None else "transcript"
            )

        rows = []

        def child(feature: str, s: int, e: int, frame: str, i: int) -> str:
            attrs = (
                f"ID={transcript_id}.{feature}.{i};Parent={transcript_id}"
            )
            return "\t".join([
                self.sequence, source, feature,
                str(s), str(e),
                ".", self.strand, frame, attrs,
            ])

        tx_attrs = f"ID={transcript_id};Name={transcript_id}"
        if gene_id is not None:
            tx_attrs += f";Parent={gene_id}"
        rows.append("\t".join([
            self.sequence, source, "mRNA",
            str(self.start + 1), str(self.end),
            ".", self.strand, ".", tx_attrs,
        ]))

        order = (
            range(len(self.cds)) if self.strand == "+"
            else range(len(self.cds) - 1, -1, -1)
        )
        frames = [0] * len(self.cds)
        cumulative_bases = 0
        for i in order:
            frames[i] = (3 - (cumulative_bases % 3)) % 3
            cumulative_bases += self.cds[i].end - self.cds[i].start

        exon_i = cds_i = 1
        for i, block in enumerate(self.cds):
            frame = frames[i]
            s = block.start + 1
            e = block.end

            rows.append(child("CDS", s, e, str(frame), cds_i))
            cds_i += 1
            rows.append(child("exon", s, e, str(frame), exon_i))
            exon_i += 1

        return "\n".join(rows)

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


class Gene(BaseModel):
    """A gene groups one or more :class:`Transcript` isoforms that share
    a locus and strand. The annotation tools built on this framework
    currently emit a single isoform per gene, but modelling the gene
    explicitly keeps the gene- and transcript-level identifiers
    consistent across every output format.

    ``name`` is optional and is the only identifier stored on the
    object. When it is not given, the identifier written to a file is
    generated at write time from the gene's position among all genes
    (see :meth:`display_name`); no index is kept on the gene.
    """

    name: str | None = None
    sequence: str
    strand: Literal["+", "-"]
    transcripts: list[Transcript]

    @classmethod
    def from_transcript(
        cls,
        tx: Transcript,
        name: str | None = None,
    ) -> "Gene":
        """Wrap a single transcript into a gene."""
        return cls(
            name=name,
            sequence=tx.sequence,
            strand=tx.strand,
            transcripts=[tx],
        )

    def display_name(
        self,
        gene_id: int,
        pattern: str = _DEFAULT_GENE_PATTERN,
    ) -> str:
        """Return the identifier used for this gene in output files: the
        explicit ``name`` if set, otherwise ``pattern`` formatted with
        the given ``gene_id`` counter.
        """
        if self.name is not None:
            return self.name
        return pattern.format(gene_id=gene_id)

    @property
    def start(self) -> int:
        return min(tx.start for tx in self.transcripts)

    @property
    def end(self) -> int:
        return max(tx.end for tx in self.transcripts)

    def add(self, tx: Transcript) -> None:
        """Add a new transcript (isoform) to this gene."""
        self.transcripts.append(tx)

    def longest(self) -> Transcript:
        """Returns the transcript with the longest coding region."""
        return max(self.transcripts, key=lambda tx: tx.cds_length())

    def classify(self, position: int) -> T_Label | None:
        """Classify a position across all isoforms of this gene. A "cds"
        label in any isoform takes precedence over "intron"; ``None``
        means the position lies outside every isoform.
        """
        label: T_Label | None = None
        for tx in self.transcripts:
            tx_label = tx.classify(position)
            if tx_label == "cds": return "cds"
            if tx_label == "intron": label = "intron"
        return label

    def to_genepred_rows(
        self,
        gene_id: int,
        transcript_pattern: str = _DEFAULT_TX_PATTERN,
    ) -> str:
        """Serialize every isoform to a GenePred line (one per line).
        ``gene_id`` is the gene's position, used to build transcript
        names when they are unnamed.
        """
        return "\n".join(
            tx.to_genepred_row(
                tx.display_name(gene_id, ti, transcript_pattern)
            )
            for ti, tx in enumerate(self.transcripts, start=1)
        )

    def to_gtf_rows(
        self,
        gene_id: int,
        source: str = "bricks2marble",
        gene_pattern: str = _DEFAULT_GENE_PATTERN,
        transcript_pattern: str = _DEFAULT_TX_PATTERN,
    ) -> str:
        """Returns GTF rows for this gene: a single 'gene' line spanning
        all isoforms, followed by the transcript-level rows of each
        isoform. ``gene_id`` is the gene's position among all genes,
        used for naming when unnamed. Coordinates are 1-based fully
        closed.
        """
        name = self.display_name(gene_id, gene_pattern)
        rows = ["\t".join([
            self.sequence, source, "gene",
            str(self.start+1), str(self.end),
            ".", self.strand, ".", f'gene_id "{name}";',
        ])]
        for ti, tx in enumerate(self.transcripts, start=1):
            rows.append(tx.to_gtf_rows(
                source=source,
                gene_id=name,
                transcript_id=tx.display_name(gene_id, ti, transcript_pattern),
            ))
        return "\n".join(rows)

    def to_gff3_rows(
        self,
        gene_id: int,
        source: str = "bricks2marble",
        gene_pattern: str = _DEFAULT_GENE_PATTERN,
        transcript_pattern: str = _DEFAULT_TX_PATTERN,
    ) -> str:
        """Returns GFF3 rows for this gene: a single 'gene' feature
        followed by the 'mRNA' subtree of each isoform. ``gene_id`` is
        the gene's position among all genes, used for naming when
        unnamed. Coordinates are 1-based fully closed.
        """
        name = self.display_name(gene_id, gene_pattern)
        rows = ["\t".join([
            self.sequence, source, "gene",
            str(self.start+1), str(self.end),
            ".", self.strand, ".", f"ID={name};Name={name}",
        ])]
        for ti, tx in enumerate(self.transcripts, start=1):
            rows.append(tx.to_gff3_rows(
                source=source,
                gene_id=name,
                transcript_id=tx.display_name(gene_id, ti, transcript_pattern),
            ))
        return "\n".join(rows)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Gene):
            raise NotImplementedError(
                f"Can only compare Gene to Gene, but got {type(other)}"
            )
        return (
            self.sequence == other.sequence and
            self.strand == other.strand and
            set(self.transcripts) == set(other.transcripts)
        )

    def __hash__(self) -> int:
        return hash((
            self.sequence,
            self.strand,
            frozenset(self.transcripts),
        ))

    model_config = {"frozen": False}


class SequenceAnnotation:
    """Holds the :class:`Gene` objects of a single sequence (chromosome/
    contig). Genes are stored in the order they are added.

    A :class:`SequenceAnnotation` is deliberately not iterable, since it
    would be ambiguous whether iteration should yield genes or
    transcripts. Use :meth:`genes` for the gene-level view and
    :meth:`transcripts` for the flat transcript-level view.
    """

    def __init__(self, sequence: str) -> None:
        self.sequence = sequence
        self._genes: list[Gene] = []
        self._fwd: list[Gene] = []
        self._rev: list[Gene] = []
        self._fwd_starts: list[int] = []
        self._rev_starts: list[int] = []
        self._fwd_dirty = False
        self._rev_dirty = False

    def add(self, item: Gene | Transcript) -> None:
        """Add a gene, or a single transcript wrapped in a new gene of
        its own, to this sequence.
        """
        gene = item if isinstance(item, Gene) else Gene.from_transcript(item)
        if gene.sequence != self.sequence:
            raise ValueError(
                f"Cannot add gene on sequence {gene.sequence!r} to the "
                f"annotation of sequence {self.sequence!r}."
            )
        self._genes.append(gene)
        (self._fwd if gene.strand == "+" else self._rev).append(gene)
        self._mark_dirty(gene.strand)

    def remove(self, item: Gene | Transcript) -> None:
        """Remove a gene or a single transcript from the annotation.
        When the last isoform of a gene is removed, the gene is dropped.
        """
        if isinstance(item, Gene):
            self._remove_gene(item)
        else:
            self._remove_transcript(item)

    def _remove_gene(self, gene: Gene) -> None:
        if not self._discard(self._genes, gene):
            raise KeyError(f"Gene not found in {self.sequence}")
        self._discard(self._fwd if gene.strand == "+" else self._rev, gene)
        self._mark_dirty(gene.strand)

    def _remove_transcript(self, tx: Transcript) -> None:
        gene = next(
            (g for g in self._genes
             if any(t is tx for t in g.transcripts)),
            None,
        )
        if gene is None:
            raise KeyError(f"Transcript not found in {self.sequence}")
        self._discard(gene.transcripts, tx)
        if not gene.transcripts:
            self._discard(self._genes, gene)
            self._discard(
                self._fwd if gene.strand == "+" else self._rev, gene
            )
        self._mark_dirty(gene.strand)

    @staticmethod
    def _discard(items: list, obj: object) -> bool:
        """Remove ``obj`` from ``items`` by identity. Returns whether an
        element was removed.
        """
        for i, item in enumerate(items):
            if item is obj:
                del items[i]
                return True
        return False

    def _mark_dirty(self, strand: str) -> None:
        if strand == "+": self._fwd_dirty = True
        else: self._rev_dirty = True

    def _sort_strand(self, strand: str) -> None:
        if strand == "+":
            if not self._fwd_dirty: return
            self._fwd.sort(key=lambda g: g.start)
            self._fwd_starts = [g.start for g in self._fwd]
            self._fwd_dirty = False
        else:
            if not self._rev_dirty: return
            self._rev.sort(key=lambda g: g.start)
            self._rev_starts = [g.start for g in self._rev]
            self._rev_dirty = False

    def _candidates(self, position: int, strand: str) -> list[Gene]:
        self._sort_strand(strand)
        genes, starts = (
            (self._fwd, self._fwd_starts) if strand == "+"
            else (self._rev, self._rev_starts)
        )
        right = bisect_right(starts, position)
        return [g for g in genes[:right] if g.end > position]

    def _classify_strand(self, position: int, strand: str) -> T_Label:
        for g in self._candidates(position, strand):
            label = g.classify(position)
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
        genes, starts = (
            (self._fwd, self._fwd_starts) if strand == "+"
            else (self._rev, self._rev_starts)
        )

        breakpositions = {start, end}
        right = bisect_right(starts, end)
        for g in genes[:right]:
            if g.end < start: continue

            for tx in g.transcripts:
                for b in tx.cds:
                    if start < b.start < end: breakpositions.add(b.start)
                    if start < b.end < end: breakpositions.add(b.end)
            if start < g.start < end:
                breakpositions.add(g.start)
            if start < g.end < end:
                breakpositions.add(g.end)

        sorted_bp = sorted(breakpositions)
        result: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, s in enumerate(sorted_bp[:-1]):
            e = sorted_bp[i+1] if i+1 < len(sorted_bp) else end
            label = self._classify_strand(s, strand)
            result[label].append((s, e))
        return dict(result)

    def genes(self) -> Iterator[Gene]:
        """Iterate over the genes of this sequence."""
        return iter(self._genes)

    def transcripts(self) -> Iterator[Transcript]:
        """Iterate the transcripts of this sequence, gene by gene."""
        for gene in self._genes:
            yield from gene.transcripts

    def __len__(self) -> int:
        """Number of transcripts held in this sequence annotation."""
        return sum(len(gene.transcripts) for gene in self._genes)


class Annotation:
    """Representation of a genome annotation, designed to align with the
    GenePred file format.
    """

    def __init__(self, start_gene_id: int = 1) -> None:
        """Args:
            start_gene_id (int, optional): The counter value the first
                gene is named after; genes are numbered from there in
                write order. Nothing is stored per gene, so removing a
                gene never leaves a gap. Use it to continue numbering
                across several annotations. Defaults to 1.
        """
        self._sequences: OrderedDict[str, SequenceAnnotation] = OrderedDict()
        self._start_gene_id = start_gene_id
        self._gene_pattern = _DEFAULT_GENE_PATTERN
        self._transcript_pattern = _DEFAULT_TX_PATTERN

    @property
    def next_gene_id(self) -> int:
        """The counter value a gene appended now would be named after:
        ``start_gene_id`` plus the number of genes currently held. Use
        it as the ``start_gene_id`` of a following annotation to keep
        numbering contiguous.
        """
        return self._start_gene_id + sum(1 for _ in self.genes())

    def set_naming(
        self,
        gene_pattern: str | None = None,
        transcript_pattern: str | None = None,
    ) -> None:
        """Change the fallback naming patterns used when a gene or
        transcript has no explicit ``name``.

        Both are ``str.format`` templates receiving the integer counters
        ``gene_id`` and ``transcript_id``. Defaults are ``"g{gene_id}"``
        for genes and ``"g{gene_id}_t{transcript_id}"`` for transcripts.
        """
        if gene_pattern is not None:
            self._gene_pattern = gene_pattern
        if transcript_pattern is not None:
            self._transcript_pattern = transcript_pattern

    def add(self, item: Gene | Transcript) -> None:
        """Add a gene, or a single transcript wrapped in a new gene of
        its own, to the annotation. Genes are stored in add order; their
        output identifiers are only determined at write time.
        """
        gene = item if isinstance(item, Gene) else Gene.from_transcript(item)
        if gene.sequence not in self:
            self._sequences[gene.sequence] = SequenceAnnotation(gene.sequence)
        self._sequences[gene.sequence].add(gene)

    def remove(self, item: Gene | Transcript) -> None:
        """Remove an already existing gene or transcript from the
        annotation.
        """
        self[item.sequence].remove(item)

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

    def genes(self) -> Iterator[Gene]:
        """Iterate over all genes of the annotation, sequence by
        sequence.
        """
        for seq_ann in self:
            yield from seq_ann.genes()

    def transcripts(self) -> Iterator[Transcript]:
        """Iterate over all transcripts of the annotation, sequence by
        sequence.
        """
        for seq_ann in self:
            yield from seq_ann.transcripts()

    @classmethod
    def from_genepred(cls, path: Path | str) -> "Annotation":
        ann = cls()
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"): continue
                tx = Transcript.from_genepred_row(line)
                if len(tx.cds) > 0: ann.add(tx)
        return ann

    def to_genepred(self, path: Path | str, append: bool = False) -> None:
        """Write every transcript as one GenePred row per isoform."""
        with open(path, "a" if append else "w") as fh:
            for gi, gene in enumerate(self.genes()):
                gene_id = self._start_gene_id + gi
                for ti, tx in enumerate(gene.transcripts, start=1):
                    if len(tx.cds) == 0: continue
                    name = tx.display_name(
                        gene_id, ti, self._transcript_pattern
                    )
                    fh.write(tx.to_genepred_row(name) + "\n")

    def to_gtf(
        self,
        path: Path | str,
        source: str = "bricks2marble",
        append: bool = False,
    ) -> None:
        """Write the annotation as GTF. Each gene emits a single 'gene'
        line and one transcript subtree per isoform; gene and transcript
        identifiers are the explicit names when set, otherwise the
        configured fallback patterns (see :meth:`set_naming`), so they
        stay consistent with the other formats.
        """
        with open(path, "a" if append else "w") as fh:
            for gi, gene in enumerate(self.genes()):
                fh.write(gene.to_gtf_rows(
                    gene_id=self._start_gene_id + gi,
                    source=source,
                    gene_pattern=self._gene_pattern,
                    transcript_pattern=self._transcript_pattern,
                ) + "\n")

    def to_gff3(
        self,
        path: Path | str,
        source: str = "bricks2marble",
        append: bool = False,
    ) -> None:
        """Write the annotation as GFF3. Each gene emits a 'gene'
        feature followed by the 'mRNA' subtree of each isoform.
        """
        file_exists = Path(path).exists()
        with open(path, "a" if append else "w") as fh:
            if not append or not file_exists:
                fh.write("##gff-version 3\n")
            for gi, gene in enumerate(self.genes()):
                fh.write(gene.to_gff3_rows(
                    gene_id=self._start_gene_id + gi,
                    source=source,
                    gene_pattern=self._gene_pattern,
                    transcript_pattern=self._transcript_pattern,
                ) + "\n")

    def sequence_to_file(
        self,
        target: Literal["coding", "protein"],
        fasta: "Fasta",
        path: Path | str,
        mode: str = "w",
        line_width: int = 60,
        header_fn: Callable[[Gene, Transcript], str] | None = None,
        skip_empty: bool = True,
        translation_table: int | None = None,
    ) -> None:
        """For a given :class:`Fasta` object, write the coding or
        protein sequence to a given file that corresponds to this
        annotation.

        Args:
            target (str): Can be either "coding" or "protein".
            fasta (Fasta): Genome Fasta used to extract coding
                sequences.
            path (Path | str): Output Fasta path.
            mode (str, optional): Mode for opening the file to write the
                sequence to. Defaults to "w".
            line_width (int, optional): Wrap sequence to this line width
                (Fasta style). Defaults to 60.
            header_fn (callable, optional): Custom header builder,
                receiving the enclosing :class:`Gene` and the
                :class:`Transcript`. If None, uses:
                "{gene_id}|{tx_id}|{seqname}:{start}-{end}({strand})".
            skip_empty (bool, optional): If True, skips empty coding
                sequences. Defaults to True.
        """
        path = Path(path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)

        def default_header(
            gene: Gene, tx: Transcript, gene_id: int, tx_id: int
        ) -> str:
            gene_name = gene.display_name(gene_id, self._gene_pattern)
            tx_name = tx.display_name(gene_id, tx_id, self._transcript_pattern)
            return (
                f"{gene_name}|{tx_name}|{tx.sequence}"
                f":{tx.start}-{tx.end}({tx.strand})"
            )

        with open(path, mode) as fh:
            gene_id = self._start_gene_id
            for seqanno in self:
                sequence = fasta[seqanno.sequence]
                for gene in seqanno.genes():
                    for ti, tx in enumerate(gene.transcripts, start=1):
                        if target == "coding":
                            seq = tx.coding_sequence(sequence).string()
                        elif target == "protein":
                            seq = tx.protein_sequence(sequence, translation_table)

                        if skip_empty and (seq is None or len(seq) == 0):
                            continue

                        header = (
                            header_fn(gene, tx) if header_fn is not None
                            else default_header(gene, tx, gene_id, ti)
                        )
                        fh.write(f">{header}\n")
                        if line_width and line_width > 0: fh.write(
                            "\n".join(textwrap.wrap(seq, width=line_width))
                            + "\n"
                        )
                        else: fh.write(seq + "\n")
                    gene_id += 1

    def __iter__(self) -> Iterator[SequenceAnnotation]:
        return iter(self._sequences.values())

    def __contains__(self, sequence: str) -> bool:
        return sequence in self._sequences

    def __getitem__(self, sequence: str | int) -> SequenceAnnotation:
        if isinstance(sequence, int):
            return list(self._sequences.values())[sequence]
        if sequence not in self:
            raise KeyError(f"Sequence {sequence!r} is not part of Annotation")
        return self._sequences[sequence]
