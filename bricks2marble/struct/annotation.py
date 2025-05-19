import csv
from pathlib import Path
from typing import Literal

from .transcript import FeatureType, GTFEntry, Transcript


class Gene:
    """Class representing a gene containing multiple transcripts."""

    def __init__(self, id: str, strand: Literal["+", "-"] = "+") -> None:
        self.id = id
        self.strand: Literal["+", "-"] = strand
        self.start = -1
        self.end = -1
        self.transcripts: dict[str, Transcript] = {}

    def finalize(self) -> None:
        """Add to all Transcript objects transcript, intron, CDS, exon
        coordinates if they were not included in the gtf file. Delete
        all transripts that have no exons or CDS.
        """
        tx_no_cds = []
        for k in self.transcripts:
            if not self.transcripts[k].finalize():
                tx_no_cds.append(k)
        for k in tx_no_cds:
            del self.transcripts[k]

    def add(self, entry: GTFEntry, transcript_id: str) -> None:
        if transcript_id not in self.transcripts:
            self.transcripts[transcript_id] = Transcript(
                transcript_id,
                gene_id=self.id,
                seqname=entry.name,
                strand=self.strand,
            )
        self.transcripts[transcript_id].add(entry)
        if (self.start < 0
                or self.transcripts[transcript_id].start < self.start):
            self.start = self.transcripts[transcript_id].start
        if self.end < 0 or self.transcripts[transcript_id].end > self.end:
            self.end = self.transcripts[transcript_id].end

    def to_list(self) -> list[GTFEntry]:
        if len(self.transcripts) == 0:
            return []
        gtf = []
        for tx in sorted(
            (tx for tx in self.transcripts.values()),
            key=lambda x: (x.start, x.end),
        ):
            gtf.extend(tx.to_list())
        gtf.insert(0, GTFEntry(
            name=gtf[-1].name,
            source=gtf[-1].source,
            feature=FeatureType.Gene,
            start=self.start,
            end=self.end,
            score=None,
            strand=self.strand,
            frame=None,
            attributes=f"gene_id \"{self.id}\";",
        ))
        return gtf


class Annotation:
    """Class handling the data structures and methods for a one genome
    annotation file.
    """

    def __init__(self) -> None:
        self.genes: dict[str, Gene] = {}

    def add(
        self,
        entry: GTFEntry | None = None,
        gene_id: str | None = None,
        transcript_id: str | None = None,
    ) -> None:
        """Adds the given entry to the gene with reported gene and
        transcript ID. If only the gene ID is given, instead creates a
        new gene.
        """
        if gene_id is not None:
            if entry is None and gene_id in self.genes:
                raise KeyError(f"Gene ID {gene_id!r} is not unique")
            if gene_id not in self.genes:
                self.genes[gene_id] = Gene(gene_id)

            if entry is not None:
                if transcript_id is None:
                    raise ValueError(
                        "If entry is given, transcript_id has to be supplied"
                    )
                self.genes[gene_id].add(entry, transcript_id)
        else:
            raise ValueError("gene_id has to be supplied")

    def finalize(self) -> None:
        """Add to all Transcript objects transcript, intron, CDS, exon
        coordinates if they were not included in the gtf file. Delete
        all transripts that have no exons or CDS.
        """
        for gene in self.genes.values():
            gene.finalize()

    # def find_genes(self) -> None:
    #     """Find all genes in the annotation and find the transcripts
    #     that belong to each gene. Also, cretae a dict with the gtf lines
    #     for each gene.
    #     """
    #     self.gene_gtf = {}
    #     self.genes = {}
    #     for tx in self.transcripts.values():
    #         if tx.gene_id in self.genes:
    #             if not (
    #                 tx.seqname == self.gene_gtf[tx.gene_id].name
    #                 and tx.strand == self.gene_gtf[tx.gene_id].strand
    #             ):
    #                 tx.gene_id = tx.gene_id + '.' + tx.seqname + '.' + tx.strand
    #             else:
    #                 self.genes[tx.gene_id].append(tx.id)
    #                 self.gene_gtf[tx.gene_id].start = min(
    #                     self.gene_gtf[tx.gene_id].start,
    #                     tx.start,
    #                 )
    #                 self.gene_gtf[tx.gene_id].end = max(
    #                     self.gene_gtf[tx.gene_id].end,
    #                     tx.end,
    #                 )
    #                 continue
    #         self.genes.update({tx.gene_id: [tx.id]})
    #         self.gene_gtf.update({tx.gene_id: GTFEntry(
    #             name=tx.seqname,
    #             source=tx.source,
    #             feature='gene',
    #             start=tx.start,
    #             end=tx.end,
    #             score=None,
    #             strand=tx.strand,
    #             frame=None,
    #             attributes=tx.gene_id,
    #         )})

    def get_transcripts(self) -> list[Transcript]:
        """Returns a list of all transcripts."""
        return [
            tx for gene in self.genes.values()
            for tx in gene.transcripts.values()
        ]

    # def rename_transcript_ids(self, prefix: str = "") -> dict[str, str]:
    #     """Renames all transcripts and genes and returns translation
    #     table for old transcripts id to new transcripts id.

    #     Args:
    #         prefix (string): String added in front of each transcript
    #             and gene ID.

    #     Returns:
    #         dict[str, str]: Translation dictionary for old transcript id
    #             to new transcript id.
    #     """
    #     lookup = {}
    #     gene_numb = 1
    #     old_gene_gtf = sorted(
    #         self.gene_gtf.values(),
    #         key=lambda g: (g.name, g.start, g.end),
    #     )
    #     self.gene_gtf = {}
    #     old_genes = self.genes
    #     self.genes = {}
    #     old_txs = self.transcripts
    #     self.transcripts = {}
    #     if prefix:
    #         prefix += '_'
    #     for gene in old_gene_gtf:
    #         tx_numb = 1
    #         old_gene_id = gene.attributes
    #         new_gene_id = "{}g{}".format(prefix, gene_numb)
    #         gene.attributes = new_gene_id
    #         self.genes.update({new_gene_id : []})
    #         self.gene_gtf.update({new_gene_id : gene})
    #         for old_tx_id in old_genes[old_gene_id]:
    #             new_tx_id = "{}g{}.t{}".format(prefix, gene_numb, tx_numb)
    #             self.transcripts.update({new_tx_id : old_txs[old_tx_id]})
    #             self.transcripts[new_tx_id].id = new_tx_id
    #             self.transcripts[new_tx_id].gene_id = new_gene_id
    #             self.genes[new_gene_id].append(new_tx_id)
    #             tx_numb +=1
    #             lookup[new_tx_id] = old_tx_id
    #         gene_numb += 1
    #     return lookup

    def to_list(self) -> list[GTFEntry]:
        """Returns a list of :class:`GTFEntry` objects."""
        gtf = []
        for gene in sorted(
            self.genes.values(),
            key=lambda g: (g.id, g.start, g.end),
        ):
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
