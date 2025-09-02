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

    def finalize(self, min_cds_length: int | None = None) -> None:
        """Add to all Transcript objects transcript, intron, CDS, exon
        coordinates if they were not included in the gtf file.

        Args:
            min_cds_length (int, optional): Minimal length of coding
                regions. All transcripts with a shorter coding region
                will be deleted. Defaults to no checks for length.
        """
        for k in self._transcripts:
            flag = self._transcripts[k].finalize(min_cds_length=min_cds_length)
            if not flag:
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

    def finalize(self, min_cds_length: int | None = None) -> None:
        """Add to all Transcript objects transcript, intron, CDS, exon
        coordinates if they were not included in the gtf file. Delete
        all transripts that have no exons or CDS.

        Args:
            min_cds_length (int, optional): Minimal length of coding
                regions. All transcripts with a shorter coding region
                will be deleted. Defaults to no checks for length.
        """
        for gene in self:
            gene.finalize(min_cds_length=min_cds_length)

    def remove_wrong_transcripts(
        self,
        fasta: "FASTA",
        start_codons: list[str] | None = None,
        stop_codons: list[str] | None = None,
    ) -> None:
        """Removes all transcripts from this annotation that do not
        start with one of the given start codons or do not end with
        given stop codons.
        Also removes all transcripts that are out-of-bounds for the
        given FASTA.

        Args:
            fasta (FASTA): The corresponding fasta object to this
                annotation.
            start_codons (list[str], optional): A list of strings of
                possible start codons. Defaults to only "ATG".
            stop_codons (list[str], optional): A list of strings of
                possible stop codons. Defaults to "TAG", "TAA" or "TGA".
        """
        from ..tools import check_annotation_boundaries
        check_annotation_boundaries(
            self, fasta,
            start_codons=start_codons,
            stop_codons=stop_codons,
            remove=True,
        )

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
