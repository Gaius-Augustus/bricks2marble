import csv
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
            entry.start -= 1
            annotation.add(entry)

        return annotation
