from pathlib import Path

from ..struct import Annotation


def load_annotation(path: Path | str) -> Annotation:
    path = Path(path)
    return Annotation.from_genepred(path)
