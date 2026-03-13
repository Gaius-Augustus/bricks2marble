import csv
import textwrap
from collections import OrderedDict
from collections.abc import Generator, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Literal

from .transcript import FeatureType, GTFEntry, Transcript

if TYPE_CHECKING:
    from .fasta import Fasta


class Gene:
    """Class representing a gene containing multiple transcripts."""

    def __init__(self, id: str) -> None:
        self.id = id
        self.seqname: str | None = None
        self.strand: Literal["+", "-"] | None = None
        self.start = -1
        self.end = -1
        self._transcripts: OrderedDict[str, Transcript] = OrderedDict()

    def add(self, entry: GTFEntry) -> None:
        """Adds a gtf entry to this gene by adding it to one of the
        contained transcripts.

        Args:
            entry (GTFEntry): The entry that should be added to the
                gene. Can be of any :class:`FeatureType`.

                *Warning*: The ``start`` and ``end`` attribute of the
                entry has to align with Python indexing, starting at
                zero and the end is exclusive.
        """
        if entry.feature == FeatureType.Gene:
            return

        if self.seqname is None: self.seqname = entry.name
        if self.strand is None: self.strand = entry.strand
        t_id = entry.attribute("transcript_id")
        if t_id not in self._transcripts:
            self._transcripts[t_id] = Transcript(t_id)

        self._transcripts[t_id].add(entry)

        if (self.start < 0 or self._transcripts[t_id].start < self.start):
            self.start = self._transcripts[t_id].start
        if self.end < 0 or self._transcripts[t_id].end > self.end:
            self.end = self._transcripts[t_id].end

    def rename(self, name: str | Callable[[str], str]) -> None:
        """Changes the name of the sequence this gene is located in by
        changing the names in all transcripts of this gene.

        Args:
            name (str | callable): If a string is given, changes all
                sequence names to that string. If a callable is given,
                this callable is applied to the sequence names, which
                are then set to the returned string.
        """
        for key in self._transcripts:
            self._transcripts[key].rename(name)

    def at(self, position: int) -> FeatureType:
        """Returns the type of feature at the given position in the
        Gene. Indexing follows Python convention.
        """
        if position < self.start or position >= self.end:
            raise IndexError(
                f"Position {position} is out of bounds for Gene at "
                f"[{self.start}, {self.end})."
            )
        for transcript in self._transcripts.values():
            if transcript.start <= position < transcript.end:
                return transcript.at(position)
        return FeatureType.Unknown

    def finalize(self) -> None:
        """Finalizes all transcripts."""
        for k in self._transcripts:
            self._transcripts[k].finalize()

    def to_list(self) -> list[GTFEntry]:
        if len(self._transcripts) == 0:
            return []
        gtf = []
        for tx in sorted(
            (tx for tx in self),
            key=lambda x: (x.start, x.end),
        ):
            gtf.extend(tx.to_list())
        gtf.insert(0, GTFEntry(
            name=gtf[-1].name,
            source=gtf[-1].source,
            feature=FeatureType.Gene,
            start=self.start+1,
            end=self.end,
            score=None,
            strand=gtf[-1].strand,
            frame=None,
            attributes=f"gene_id \"{self.id}\";",
        ))
        return gtf

    def __iter__(self) -> Iterator[Transcript]:
        return iter(list(self._transcripts.values()))

    def __getitem__(self, key: str) -> Transcript:
        return self._transcripts[key]


class Annotation:
    """Class handling the data structures and methods for a one genome
    annotation file.
    """

    def __init__(self) -> None:
        self._genes: OrderedDict[str, Gene] = OrderedDict()
        self._sequences: dict[str, list[str]] = OrderedDict()

    def add(self, entry: GTFEntry) -> None:
        """Adds the given gtf entry to the gene.

        Args:
            entry (GTFEntry): The entry that should be added to the
                gene. Can be of any :class:`FeatureType`.

                *Warning*: The ``start`` and ``end`` attribute of the
                entry has to align with Python indexing, starting at
                zero and the end is exclusive.
        """
        gene_id = entry.attribute("gene_id")
        if gene_id not in self._genes:
            self._genes[gene_id] = Gene(gene_id)
            if entry.name not in self._sequences:
                self._sequences[entry.name] = []
            self._sequences[entry.name].append(gene_id)
        self._genes[gene_id].add(entry)

    def rename(self, name: str | Callable[[str], str]) -> None:
        """Changes the names of all sequences in this annotation.

        Args:
            name (str | callable): If a string is given, changes all
                sequence names to that string. If a callable is given,
                this callable is applied to the sequence names, which
                are then set to the returned string.
        """
        for key in self._genes:
            self._genes[key].rename(name)

    def finalize(self) -> None:
        """Finalizes all transcripts in the annotation."""
        for gene in self:
            gene.finalize()

    def remove(self, obj: Transcript | Gene) -> None:
        """Remove the given transcript or gene from the annotation.
        Raises an error if it does not exist.
        """
        if isinstance(obj, Gene):
            self._genes.pop(obj.id)
            self._sequences[obj.seqname].remove(obj.id)
        elif isinstance(obj, Transcript):
            self._genes[obj.gene_id]._transcripts.pop(obj.id)

    def clean(
        self,
        fasta: "Fasta",
        inframe_stop_codons: bool = True,
        min_coding_length: int | None = None,
        exon_boundaries: bool = False,
        coding_repeats: bool = False,
        out_of_bounds: bool = False,
    ) -> None:
        """Removes any transcripts that do not meet the given
        requirements. For fine-grained options, have a look at the
        corresponding functions in `bricks2marble.tools.post`.

        Args:
            fasta (Fasta): A fasta is required to be specified for all
                subsequent arguments.
            inframe_stop_codons (list[str], optional): Removes all
                transcripts with inframe stop codons from the
                annotation. Defaults to True.
            min_coding_length (int, optional): Minimal length of coding
                regions. All transcripts with a shorter coding region
                will be deleted. Defaults to no checks for length.
            exon_boundaries (bool, optional): Removes transcripts with
                wrong exon boundaries from the annotation. Is based on
                the default border codons. Defaults to no action.
            coding_repeats (bool, optional): Removes transcripts that
                have coding regions that overlap with repeats. Defaults
                to no action.
            out_of_bounds (bool, optional): Removes transcripts that are
                out-of-bounds for the given fasta. Defaults to no
                action.
        """
        if inframe_stop_codons:
            from ..tools.post import check_inframe_stop_codons
            check_inframe_stop_codons(self, fasta, remove=True)
        if min_coding_length is not None:
            from ..tools.post import check_min_coding_length
            check_min_coding_length(self, min_coding_length, remove=True)
        if exon_boundaries:
            from ..tools.post import check_exon_boundaries
            check_exon_boundaries(self, fasta, remove=True)
        if coding_repeats:
            from ..tools.post import check_coding_repeats
            check_coding_repeats(self, fasta, remove=True)
        if out_of_bounds:
            from ..tools.post import check_out_of_bounds
            check_out_of_bounds(self, fasta, remove=True)

    def at(
        self,
        position: int,
    ) -> tuple[FeatureType | None, FeatureType | None]:
        """Returns the type of label at the given position. Can be used
        to determine whether a position is coding or non-coding.

        Args:
            position (int): The position to identify. Indexing starts at
                0, and the last position is exclusive, following Python
                indexing.

        Returns:
            tuple: Two feature types for the forward and backward
            strand.
        """
        fwd = None
        bwd = None
        for gene in self:
            if fwd is not None and bwd is not None: break
            if gene.start <= position < gene.end:
                if gene.strand == "+": fwd = gene.at(position)
                if gene.strand == "-": bwd = gene.at(position)
        return fwd, bwd

    def select(
        self,
        sequence: str | None = None,
        start: int | None = None,
        end: int | None = None,
    ) -> None:
        """Select only specific transcripts from the annotation. Changes
        the annotation in-place.

        Args:
            sequence (str, optional): Select only transcripts with the
                given name.
            start (int, optional): Select only transcripts that are
                starting at or after the given position.
            end (int, optional): Select only transcripts that are
                ending strictly before the given position.
        """
        if sequence is not None:
            drop_keys = []
            for gid in self._genes:
                if self._genes[gid].seqname != sequence:
                    drop_keys.append(gid)
            for gid in drop_keys:
                self.remove(self._genes[gid])

        if start is not None:
            drop_keys = []
            for gid in self._genes:
                if self._genes[gid].start < start:
                    drop_keys.append(gid)
            for gid in drop_keys:
                self.remove(self._genes[gid])

        if end is not None:
            drop_keys = []
            for gid in self._genes:
                if self._genes[gid].end >= end:
                    drop_keys.append(gid)
            for gid in drop_keys:
                self.remove(self._genes[gid])

    def merge(self, annotation: "Annotation") -> None:
        """Merges this annotation with the given."""
        for ex_gene in annotation:
            for gene in self:
                if (
                    gene.seqname == ex_gene.seqname
                    and gene.strand == ex_gene.strand
                    and gene.start == ex_gene.start
                    and gene.end == ex_gene.end
                ): break
            else: # no break
                for ex_tx in ex_gene:
                    for entry in ex_tx.entries:
                        g_id = entry.attribute("gene_id")
                        t_id = entry.attribute("transcript_id")
                        entry.attributes = (
                            f"gene_id \"merged_{g_id}\"; "
                            f"transcript_id \"merged_{t_id}\";"
                        )
                        self.add(entry)

    def to_list(self) -> list[GTFEntry]:
        """Returns a list of :class:`GTFEntry` objects."""
        gtf = []
        for gene in self:
            gtf.extend(gene.to_list())
        return gtf

    def to_gtf(
        self,
        path: Path | str,
        mode: Literal["w", "a", "x"] = "w",
    ) -> None:
        """Write the annotation in gtf format to the given path.

        Args:
            path (str): Path to the output file, ends with ".gtf".
            mode (str): Mode in which to open the file. Possible choices
                are "w", "a" or "x". Defaults to "w".
        """
        path = Path(path)

        with open(path, mode) as file:
            out_writer = csv.writer(
                file,
                delimiter='\t',
                quotechar="|",
                lineterminator='\n',
            )
            for line in self.to_list():
                out_writer.writerow(line.to_list())

    def extract_to_file(
        self,
        target: Literal["coding", "protein"],
        fasta: "Fasta",
        path: Path | str,
        mode: Literal["w", "a", "x"] = "w",
        line_width: int = 60,
        header_fn: Callable[[Gene, Transcript], str] | None = None,
        skip_empty: bool = True
    ) -> None:
        """Combine Annotation and Fasta data to write a target sequence
        to a given file.

        Args:
            target (str): Can be either "coding" or "protein".
            fasta (Fasta): Genome Fasta used to extract coding
                sequences.
            path (Path | str): Output Fasta path.
            mode (Literal["w","a","x"], optional): File open mode.
                Defaults to "w".
            line_width (int, optional): Wrap sequence to this line width
                (Fasta style). Defaults to 60.
            header_fn (callable, optional): Custom header builder. If
                None, uses:
                "{gene_id}|{tx_id}|{seqname}:{start}-{end}({strand})".
            skip_empty (bool, optional): If True, skips empty coding
                sequences. Defaults to True.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        def default_header(g: Gene, tx: Transcript) -> str:
            return (
                f"{g.id}|{tx.id}|{g.seqname}:{tx.start}-{tx.end}({g.strand})"
            )
        mk_header = header_fn or default_header

        with open(path, mode) as fh:
            for gene in self:
                sequence = fasta[gene.seqname]
                for tx in gene:
                    if target == "coding":
                        seq = tx.coding_sequence(sequence).string()
                    elif target == "protein":
                        seq = tx.protein_sequence(sequence)

                    if skip_empty and (seq is None or len(seq) == 0):
                        continue

                    header = mk_header(gene, tx)
                    fh.write(f">{header}\n")
                    if line_width and line_width > 0:
                        fh.write(
                            "\n".join(textwrap.wrap(seq, width=line_width))
                        )
                        fh.write("\n")
                    else:
                        fh.write(seq + "\n")

    def in_sequence(self, name: str) -> Generator[Gene, None, None]:
        """Yields genes in the sequence with given name. If the sequence
        name does not exist, this yields no genes, but also does not
        raise an Error.
        """
        for i in (self._sequences[name] if name in self._sequences else []):
            yield self._genes[i]

    def __iter__(self) -> Iterator[Gene]:
        return iter(list(self._genes.values()))

    def __getitem__(self, key: str) -> Gene:
        return self._genes[key]
