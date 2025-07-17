import csv
from pathlib import Path

from ..struct import Annotation, FeatureType, GTFEntry


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
            if entry.feature == FeatureType.Gene:
                annotation.add(gene_id=entry.attribute("gene_id"))
            else:
                transcript_id = entry.attribute("transcript_id")
                gene_id = entry.attribute("gene_id")
                annotation.add(
                    entry=entry,
                    gene_id=gene_id,
                    transcript_id=transcript_id,
                )
        return annotation
