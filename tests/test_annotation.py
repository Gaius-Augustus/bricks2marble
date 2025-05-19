from pathlib import Path

import bricks2marble as b2m


def test_load(gtf_path: Path) -> None:
    annotation = b2m.load_gtf(gtf_path)

    print(annotation.to_list())
    assert False