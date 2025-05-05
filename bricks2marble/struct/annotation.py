import csv
import sys
from pathlib import Path
from typing import Literal

from .transcript import GTFEntry, NotGTFFormat, Transcript


class Annotation:
    """Class handling the data structures and methods for a one genome
    annotation file.

    Args:
        path (str): Path to the annotation / gene prediction file in gtf
            format.
        id (str): Annotation ID.
    """

    def __init__(self, path: Path | str, id: str) -> None:
        self.id = id
        self.genes = {'None' : []}
        self.gene_gtf = {}
        self.transcripts: dict[str, Transcript] = {}
        self.path = path if isinstance(path, Path) else Path(path)
        self.translation_tab = []

    def add_gtf(self) -> None:
        """Read a gtf file and create a dictionary of
        :class:`Transcript` objects for all transcripts in the file.
        """
        with open(self.path, 'r') as file:
            file_lines = csv.reader(file, delimiter='\t')
            for line_ in file_lines:
                line = [l.strip(" ") for l in line_]
                if line[0].startswith("#"):
                    continue

                entry = GTFEntry.from_list(line)

                if entry.feature == 'gene':
                    gene_id = entry.attributes
                    self.genes_update(gene_id)
                    if not gene_id in self.gene_gtf.keys():
                        self.gene_gtf.update({gene_id: line})
                    else:
                        sys.stderr.write(
                            f"ERROR, gene_id not unique: {gene_id}"
                        )
                elif entry.feature == 'transcript':
                    transcript_id = entry.attributes
                    gene_id = ''
                    self.transcript_update(
                        transcript_id, gene_id, entry.seqname, entry.strand,
                    )
                    self.transcripts[transcript_id].add_line(entry)
                else:
                    transcript_id = entry.attributes.split('transcript_id "')
                    if len(transcript_id) > 1:
                        transcript_id = transcript_id[1].split('";')[0]
                    else:
                        raise NotGTFFormat(
                            f"File {self.path} is not in gtf format.\n"
                            f"Error in line {entry}"
                        )

                    gene_id = entry.attributes.split('gene_id "')
                    if len(gene_id) > 1:
                        gene_id = gene_id[1].split('";')[0]
                    else:
                        gene_id = 'None'
                        for key, value in self.genes.items():
                            if value == transcript_id: gene_id = key

                    self.transcript_update(
                        transcript_id,
                        gene_id,
                        entry.seqname,
                        entry.strand,
                    )
                    self.genes_update(gene_id, transcript_id)
                    self.transcripts[transcript_id].add_line(line)

        for tx_id in self.genes['None']:
            gene_id = tx_id + '_g'
            self.genes_update(gene_id, tx_id)

    def norm_tx_format(self) -> None:
        """Add to all Transcript objects transcript, intron, CDS, exon
        coordinates if they were not included in the gtf file. Delete
        all transripts that have no exons or CDS.
        """
        tx_no_cds = []
        for k in self.transcripts.keys():
            if not self.transcripts[k].add_missing_lines():
                tx_no_cds.append(k)
        for k in tx_no_cds:
            del self.transcripts[k]

    def genes_update(
        self,
        gene_id: str,
        transcript_id: str | None = None,
    ) -> None:
        """Update gene ID dict.

        Args:
            gene_id (str): Gene ID
            transcript_id (str, optional): Transcript ID.
        """
        if gene_id not in self.genes.keys():
            self.genes.update({gene_id : []})
        if (transcript_id is not None
                and transcript_id not in self.genes[gene_id]):
            self.genes[gene_id].append(transcript_id)
        if transcript_id in self.genes['None'] and not gene_id == 'None':
            self.genes['None'].remove(transcript_id)
            self.transcripts[transcript_id].gene_id = gene_id

    def transcript_update(
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
        if t_id not in self.transcripts.keys():
            self.transcripts.update(
                {t_id : Transcript(t_id, g_id, chr, self.id, strand)}
            )

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
                    tx.chr == self.gene_gtf[tx.gene_id][0]
                    and tx.strand == self.gene_gtf[tx.gene_id][6]
                ):
                    tx.gene_id = tx.gene_id + '.' + tx.chr + '.' + tx.strand
                else:
                    self.genes[tx.gene_id].append(tx.id)
                    self.gene_gtf[tx.gene_id][3] = min(
                        self.gene_gtf[tx.gene_id][3],
                        tx.start,
                    )
                    self.gene_gtf[tx.gene_id][4] = max(
                        self.gene_gtf[tx.gene_id][4],
                        tx.end,
                    )
                    continue
            self.genes.update({tx.gene_id : [tx.id]})
            self.gene_gtf.update({tx.gene_id : [
                tx.chr,
                tx.source_method,
                'gene',
                tx.start,
                tx.end,
                '.',
                tx.strand,
                '.',
                tx.gene_id,
            ]})

    def add_transcripts(
        self,
        txs: dict[str, Transcript],
        id_prefix: str = "",
    ) -> None:
        """Adds a dict of transcripts to the transcripts of the
        annotation.

        Args:
            txs (dict[str, Transcript]): Dictionary of Transcripts added
                to the annotation.
        """
        if not id_prefix:
            self.transcripts.update({txs})
        else:
            for tx in txs.values():
                tx.id = id_prefix + tx.id
                self.transcripts.update({tx.id : tx})

    def get_subset(self, tx_list: list[str]) -> dict[str, Transcript]:
        """Get annotation file for a subset of transcripts.

        Args:
            tx_list (list[str]): List of transcript IDs.

        Returns:
            list[list[str]]: Gtf file as list of lists
        """
        tx_subset = {}
        for tx in tx_list:
            tx_subset.update({tx : self.transcripts[tx]})
        return tx_subset

    def change_id(self, new_id: str) -> None:
        """Change annotation file ID."""
        self.id = new_id
        for k in self.transcripts.keys():
            self.transcripts[k].source_anno = self.id

    def get_transcript_list(self) -> list[Transcript]:
        """Returns a list of all transcripts."""
        return list(self.transcripts.values())

    def rename_tx_ids(self, prefix: str = "") -> list[tuple[str, str]]:
        """Renames all tx and genes and returns translation table for
        old tx id to new tx id.

        Args:
            prefix (string): String added before each tx and gene ID.

        Returns:
            translation_tab (list[tuple[str, str]]): Translation table
                for old tx id to new tx id.
        """
        self.translation_tab = []
        gene_numb = 1
        old_gene_gtf = sorted(
            self.gene_gtf.values(),
            key=lambda g: (g[0], g[3], g[4]),
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
            old_gene_id = gene[8]
            new_gene_id = "{}g{}".format(prefix, gene_numb)
            gene[8] = new_gene_id
            self.genes.update({new_gene_id : []})
            self.gene_gtf.update({new_gene_id : gene})
            for old_tx_id in old_genes[old_gene_id]:
                new_tx_id = "{}g{}.t{}".format(prefix, gene_numb, tx_numb)
                self.transcripts.update({new_tx_id : old_txs[old_tx_id]})
                self.transcripts[new_tx_id].id = new_tx_id
                self.transcripts[new_tx_id].gene_id = new_gene_id
                self.genes[new_gene_id].append(new_tx_id)
                tx_numb +=1
                self.translation_tab.append([new_tx_id, old_tx_id])
            gene_numb += 1
        return self.translation_tab

    def get_gtf(self) -> list[list[str]]:
        """Get annotation file as a gtf list.

        Returns:
            list[list[str]]: Gtf file as a list of lists.
        """
        gtf = []
        gene_gtf = sorted(
            self.gene_gtf.values(),
            key=lambda g: (g[0], g[3], g[4]),
        )
        for gene in gene_gtf:
            gtf.append(gene)
            for tx_id in self.genes[gene[8]]:
                gtf += self.transcripts[tx_id].get_gtf()
        return gtf

    def write(self, path: Path) -> None:
        """Write the annotation in gtf format to the given path.

        Args:
            path (str): Path to the output file.
        """
        with open(path, 'w+') as file:
            out_writer = csv.writer(
                file,
                delimiter='\t',
                quotechar="|",
                lineterminator='\n',
            )
            for line in self.get_gtf():
                out_writer.writerow(line)
