import csv
from pathlib import Path
from typing import Literal

from .transcript import GTFEntry, Transcript


class Annotation:
    """Class handling the data structures and methods for a one genome
    annotation file.
    """

    def __init__(self) -> None:
        self.transcripts: dict[str, Transcript] = {}
        self.gene_gtf: dict[str, GTFEntry] = {}
        self.genes: dict[str, list[str]] = {}
        self._earmarked_genes: list[str] = []

    def add_gene(
        self,
        gene_id: str | None = None,
        transcript_id: str | None = None,
    ) -> None:
        """Add a gene to the annotation, optionally with the
        corresponding transcript ID.

        Args:
            gene_id (str): The identifier of the gene.
            transcript_id (str, optional): The identifier of the
                transcript.
        """
        if gene_id is not None and gene_id not in self.genes:
            self.genes[gene_id] = []

        if (transcript_id is not None and gene_id is not None
                and transcript_id not in self.genes[gene_id]):
            self.genes[gene_id].append(transcript_id)

        if transcript_id in self._earmarked_genes and gene_id is not None:
            self._earmarked_genes.remove(transcript_id)
            self.transcripts[transcript_id].gene_id = gene_id

    def add_transcript(
        self,
        t_id: str,
        g_id: str,
        chr: str,
        strand: Literal["+", "-"] = "+",
    ) -> None:
        """Update transcript ID dict.

        Args:
            t_id (str): Transcript ID
            g_id (str): Gene ID
            chr (str): Chromosome name
            strand (str): Strand (+/-)
        """
        if t_id not in self.transcripts:
            self.transcripts[t_id] = Transcript(t_id, g_id, chr, strand)

    def add_transcripts(
        self,
        transcripts: dict[str, Transcript],
        id_prefix: str | None = None,
    ) -> None:
        """Adds a dict of transcripts to the transcripts of the
        annotation.

        Args:
            txs (dict[str, Transcript]): Dictionary of Transcripts added
                to the annotation.
        """
        if id_prefix is None:
            self.transcripts.update(transcripts)
        else:
            self.transcripts.update({
                id_prefix+txid: tx for txid, tx in transcripts.items()
            })

    def norm_transcripts(self) -> None:
        """Add to all Transcript objects transcript, intron, CDS, exon
        coordinates if they were not included in the gtf file. Delete
        all transripts that have no exons or CDS.
        """
        tx_no_cds = []
        for k in self.transcripts:
            if not self.transcripts[k].add_missing_lines():
                tx_no_cds.append(k)

        for k in tx_no_cds:
            del self.transcripts[k]

    def find_genes(self) -> None:
        """Find all genes in the annotation and find the transcripts
        that belong to each gene. Also, cretae a dict with the gtf lines
        for each gene.
        """
        self.gene_gtf = {}
        self.genes = {}
        for tx in self.transcripts.values():
            if tx.gene_id in self.genes.keys():
                if not (
                    tx.chr == self.gene_gtf[tx.gene_id].name
                    and tx.strand == self.gene_gtf[tx.gene_id].strand
                ):
                    tx.gene_id = tx.gene_id + '.' + tx.chr + '.' + tx.strand
                else:
                    self.genes[tx.gene_id].append(tx.id)
                    self.gene_gtf[tx.gene_id].start = min(
                        self.gene_gtf[tx.gene_id].start,
                        tx.start,
                    )
                    self.gene_gtf[tx.gene_id].end = max(
                        self.gene_gtf[tx.gene_id].end,
                        tx.end,
                    )
                    continue
            self.genes.update({tx.gene_id: [tx.id]})
            self.gene_gtf.update({tx.gene_id: GTFEntry(
                name=tx.chr,
                source=tx.source,
                feature='gene',
                start=tx.start,
                end=tx.end,
                score='.',
                strand=tx.strand,
                frame='.',
                attributes=tx.gene_id,
            )})

    def get_subset(self, tx_list: list[str]) -> dict[str, Transcript]:
        """Get annotation file for a subset of transcripts.

        Args:
            tx_list (list[str]): List of transcript IDs.

        Returns:
            list[list[str]]: Gtf file as list of lists
        """
        return {tx : self.transcripts[tx] for tx in tx_list}

    def get_transcripts(self) -> list[Transcript]:
        """Returns a list of all transcripts."""
        return list(self.transcripts.values())

    def rename_transcript_ids(self, prefix: str = "") -> dict[str, str]:
        """Renames all transcripts and genes and returns translation
        table for old transcripts id to new transcripts id.

        Args:
            prefix (string): String added in front of each transcript
                and gene ID.

        Returns:
            dict[str, str]: Translation dictionary for old transcript id
                to new transcript id.
        """
        lookup = {}
        gene_numb = 1
        old_gene_gtf = sorted(
            self.gene_gtf.values(),
            key=lambda g: (g.name, g.start, g.end),
        )
        self.gene_gtf = {}
        old_genes = self.genes
        self.genes = {}
        old_txs = self.transcripts
        self.transcripts = {}
        if prefix:
            prefix += '_'
        for gene in old_gene_gtf:
            tx_numb = 1
            old_gene_id = gene.attributes
            new_gene_id = "{}g{}".format(prefix, gene_numb)
            gene.attributes = new_gene_id
            self.genes.update({new_gene_id : []})
            self.gene_gtf.update({new_gene_id : gene})
            for old_tx_id in old_genes[old_gene_id]:
                new_tx_id = "{}g{}.t{}".format(prefix, gene_numb, tx_numb)
                self.transcripts.update({new_tx_id : old_txs[old_tx_id]})
                self.transcripts[new_tx_id].id = new_tx_id
                self.transcripts[new_tx_id].gene_id = new_gene_id
                self.genes[new_gene_id].append(new_tx_id)
                tx_numb +=1
                lookup[new_tx_id] = old_tx_id
            gene_numb += 1
        return lookup

    def to_list(self) -> list[GTFEntry]:
        """Returns a list of :class:`GTFEntry` objects."""
        gtf = []
        gene_gtf = sorted(
            self.gene_gtf.values(),
            key=lambda g: (g.name, g.start, g.end),
        )
        for gene in gene_gtf:
            gtf.append(gene)
            for tx_id in self.genes[gene.attributes]:
                gtf += self.transcripts[tx_id].to_list()
        return gtf

    def write(self, path: Path | str) -> None:
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
