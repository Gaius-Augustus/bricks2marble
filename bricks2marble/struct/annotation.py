import csv
from collections import OrderedDict
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Literal

from .transcript import FeatureType, GTFEntry, Transcript

if TYPE_CHECKING:
    from .fasta import FASTA


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

    def clean(self, min_cds_length: int | None = None) -> None:
        """Removes any transcripts that do not meet the given
        requirements.

        Args:
            min_cds_length (int, optional): Minimal length of coding
                regions. All transcripts with a shorter coding region
                will be deleted. Defaults to no checks for length.
        """
        drop_tx = []
        for k in self._transcripts:
            if (
                min_cds_length is not None
                and self._transcripts[k].cds_length() < min_cds_length
            ):
                drop_tx.append(k)
        for k in drop_tx:
            self._transcripts.pop(k)

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
        self._iter_index = -1

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

    def clean(
        self,
        min_cds_length: int | None = None,
        fasta: "FASTA | None" = None,
        boundaries: bool = False,
        start_codons: list[str] | bool = True,
        stop_codons: list[str] | bool = True,
        intron_begin: list[str] | bool = True,
        intron_end: list[str] | bool = True,
        no_repeats: bool = False,
    ) -> None:
        """Removes any transcripts that do not meet the given
        requirements.

        Args:
            min_cds_length (int, optional): Minimal length of coding
                regions. All transcripts with a shorter coding region
                will be deleted. Defaults to no checks for length.
            fasta (FASTA, optional): If a FASTA is given, all
                transcripts are removed that do not start or end at
                specific start-/stop-codons. Also removes any
                out-of-bounds transcripts.
            start_codons (list[str], optional): A list of strings of
                possible start codons or a boolean value. If true,
                defaults to only "ATG" and if false, does no checks for
                start codons.
            stop_codons (list[str], optional): A list of strings of
                possible stop codons or a boolean value. If true,
                defaults to "TAG", "TAA" or "TGA" and if false, does no
                checks for stop codons.
            intron_begin (list[str], optional): A list of strings of
                possible begin patterns of introns, or a boolean value.
                If true, defaults to only "GT" and if false, does no
                checks for begin patterns.
            intron_end (list[str], optional): A list of strings of
                possible end patterns of introns, or a boolean value. If
                true, defaults to only "AG" and if false, does no checks
                for end patterns.
        """
        if boundaries:
            if fasta is None:
                raise ValueError(
                    "If boundaries is True, a FASTA object has to be given."
                )
            from ..tools.post import check_annotation_boundaries
            check_annotation_boundaries(
                self, fasta,
                start_codons=start_codons,
                stop_codons=stop_codons,
                intron_begin=intron_begin,
                intron_end=intron_end,
                remove=True,
            )
        if no_repeats:
            if fasta is None:
                raise ValueError(
                    "If no_repeats is True, a FASTA object has to be given."
                )
            from ..tools.post import check_repeat_masked
            check_repeat_masked(self, fasta, remove=True)
        for gene in self:
            gene.clean(min_cds_length=min_cds_length)

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
                self._genes.pop(gid)

        if start is not None:
            drop_keys = []
            for gid in self._genes:
                if self._genes[gid].start < start:
                    drop_keys.append(gid)
            for gid in drop_keys:
                self._genes.pop(gid)

        if end is not None:
            drop_keys = []
            for gid in self._genes:
                if self._genes[gid].end >= end:
                    drop_keys.append(gid)
            for gid in drop_keys:
                self._genes.pop(gid)

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

    def to_gtf(self, path: Path | str) -> None:
        """Write the annotation in gtf format to the given path.

        Args:
            path (str): Path to the output file, ends with ".gtf".
        """
        with open(path, 'w+') as file:
            out_writer = csv.writer(
                file,
                delimiter='\t',
                quotechar="|",
                lineterminator='\n',
            )
            for line in self.to_list():
                out_writer.writerow(line.to_list())

    def __iter__(self) -> Iterator[Gene]:
        return iter(list(self._genes.values()))

    def __getitem__(self, key: str) -> Gene:
        return self._genes[key]
