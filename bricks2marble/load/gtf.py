import csv
import sys
from pathlib import Path

from ..struct import Annotation, GTFEntry


def load_gtf(
    path: Path | str,
) -> Annotation:
    with open(path, 'r') as file:
        file_lines = csv.reader(file, delimiter='\t')

    annotation = Annotation()

    for line_ in file_lines:
        line = [l.strip(" ") for l in line_]
        if line[0].startswith("#"):
            continue

        entry = GTFEntry.from_list(line)

        if entry.feature == 'gene':
            gene_id = entry.attributes
            annotation.add_gene(gene_id)
            if not gene_id in annotation.gene_gtf.keys():
                annotation.gene_gtf[gene_id] = entry
            else:
                sys.stderr.write(
                    f"ERROR, gene_id not unique: {gene_id}"
                )
        elif entry.feature == 'transcript':
            transcript_id = entry.attributes
            gene_id = ''
            annotation.add_transcript(
                transcript_id,
                gene_id,
                entry.name,
                entry.strand,
            )
            annotation.transcripts[transcript_id].add(entry)
        else:
            transcript_id = entry.attributes.split('transcript_id "')
            if len(transcript_id) > 1:
                transcript_id = transcript_id[1].split('";')[0]
            else:
                raise RuntimeError(
                    f"File {path} is not in gtf format.\n"
                    f"Error in line {entry}"
                )

            gene_id = entry.attributes.split('gene_id "')
            if len(gene_id) > 1:
                gene_id = gene_id[1].split('";')[0]
            else:
                gene_id = 'None'
                for key, value in annotation.genes.items():
                    if value == transcript_id: gene_id = key

            annotation.add_transcript(
                transcript_id,
                gene_id,
                entry.name,
                entry.strand,
            )
            annotation.add_gene(gene_id, transcript_id)
            annotation.transcripts[transcript_id].add(entry)


    for tx_id in annotation.genes['None']:
        gene_id = tx_id + '_g'
        annotation.add_gene(gene_id, tx_id)

    return annotation
