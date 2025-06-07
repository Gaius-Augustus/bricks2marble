import csv
from pathlib import Path
from typing import Literal
from collections import OrderedDict
from collections.abc import Iterator

from .transcript import FeatureType, GTFEntry, Transcript


class Gene:
    """Class representing a gene containing multiple transcripts."""

    def __init__(self, id: str, strand: Literal["+", "-"] = "+") -> None:
        self.id = id
        self.strand: Literal["+", "-"] = strand
        self.start = -1
        self.end = -1
        self._transcripts: OrderedDict[str, Transcript] = OrderedDict()

    def finalize(self) -> None:
        """Add to all Transcript objects transcript, intron, CDS, exon
        coordinates if they were not included in the gtf file. Delete
        all transripts that have no exons or CDS.
        """
        tx_no_cds = []
        for k in self._transcripts:
            if not self._transcripts[k].finalize():
                tx_no_cds.append(k)
        for k in tx_no_cds:
            del self._transcripts[k]

    def add(self, entry: GTFEntry, transcript_id: str) -> None:
        if transcript_id not in self._transcripts:
            self._transcripts[transcript_id] = Transcript(
                transcript_id,
                gene_id=self.id,
                seqname=entry.name,
                strand=self.strand,
            )
        self._transcripts[transcript_id].add(entry)
        if (self.start < 0
                or self._transcripts[transcript_id].start < self.start):
            self.start = self._transcripts[transcript_id].start
        if self.end < 0 or self._transcripts[transcript_id].end > self.end:
            self.end = self._transcripts[transcript_id].end

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
            strand=self.strand,
            frame=None,
            attributes=f"gene_id \"{self.id}\";",
        ))
        return gtf

    def __iter__(self) -> Iterator[Transcript]:
        return iter(list(self._transcripts.values()))


class Annotation:
    """Class handling the data structures and methods for a one genome
    annotation file.
    """

    def __init__(self) -> None:
        self._genes: OrderedDict[str, Gene] = OrderedDict()
        self._iter_index = -1

    def add(
        self,
        entry: GTFEntry | None = None,
        gene_id: str | None = None,
        strand: Literal["+", "-"] = "+",
        transcript_id: str | None = None,
    ) -> None:
        """Adds the given entry to the gene with reported gene and
        transcript ID. If only the gene ID is given, instead creates a
        new gene.
        """
        if gene_id is not None:
            if entry is None and gene_id in self._genes:
                raise KeyError(f"Gene ID {gene_id!r} is not unique")

            if gene_id not in self._genes:
                self._genes[gene_id] = Gene(gene_id, strand=strand)

            if entry is not None:
                if transcript_id is None:
                    raise ValueError(
                        "If entry is given, "
                        "'transcript_id' has to be specified"
                    )
                self._genes[gene_id].add(entry, transcript_id)
        else:
            raise ValueError("gene_id has to be supplied")

    def finalize(self) -> None:
        """Add to all Transcript objects transcript, intron, CDS, exon
        coordinates if they were not included in the gtf file. Delete
        all transripts that have no exons or CDS.
        """
        for gene in self:
            gene.finalize()

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
