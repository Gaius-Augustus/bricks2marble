import os
import subprocess
import tempfile
from enum import Enum
from pathlib import Path
from typing import Literal

from ..io import load_annotation
from ..struct import Annotation
from .external import get_tool_path


class Types(Enum):

    GP = 1
    GTF = 2
    ANNOTATION = 3


def _convert(path_in: str | Path, path_out: str | Path, tool: str):
    tool_path = get_tool_path(tool)

    if tool == "genePredToGtf":
        subprocess.run(
            [tool_path, "file", str(path_in), str(path_out)],
            check=True,
            stderr=subprocess.DEVNULL,
        )

    elif tool == "gtfToGenePred":
        subprocess.run(
            [tool_path, str(path_in), str(path_out)],
            check=True,
            stderr=subprocess.DEVNULL,
        )


def _annotation_to_gtf(i, o):
    with tempfile.TemporaryDirectory() as D:
        f = Path(D) / "annotation.gp"
        i.to_genepred(f, "wt")
        _convert(f, o, "genePredToGtf")


class Converter:

    ALLOW: dict[Types, dict] = {
        Types.ANNOTATION: {
            Types.GP: lambda i, o: i.to_genepred(o, "wt"),
            Types.GTF: _annotation_to_gtf
        },
        Types.GP: {
            Types.GTF: lambda i, o: _convert(i, o, "genePredToGtf"),
            Types.ANNOTATION: lambda i, o: load_annotation(i),
        },
        Types.GTF: {
            Types.GP: lambda i, o: _convert(i, o, "gtfToGenePred"),
        },
    }

    def __init__(
        self,
        obj: Annotation | str | Path,
        to: Literal["gp", "gtf"] | type[Annotation],
    ) -> None:
        if isinstance(to, type) and to is not Annotation:
            raise ValueError(f"Unsupported type for conversion: {to}")
        if isinstance(to, str) and to not in ["gtf", "gp"]:
            raise ValueError(f"Unsupported suffix for conversion: {to!r}")
        if isinstance(obj, (str, Path)):
            self.obj = Path(obj).expanduser()
            self.obj_type = Types[self.obj.suffix[1:].upper()]
        else:
            self.obj = obj
            self.obj_type = Types.ANNOTATION

        self.to = to
        self.to_type = (
            Types[to.upper()] if isinstance(to, str) else Types.ANNOTATION
        )
        if self.to_type not in self.ALLOW[self.obj_type].keys() and (
            self.obj_type is not self.to_type
        ):
            raise ValueError(
                f"Unsupported conversion from {self.obj_type.name} to "
                f"{self.to_type.name}"
            )

    def __enter__(self) -> Annotation | Path:
        self._out_file = None
        if self.obj_type == self.to_type:
            return self.obj
        if self.to_type is not Types.ANNOTATION:
            fd, self._out_file = tempfile.mkstemp(
                suffix="."+self.to_type.name.lower(),
            )
            print(f"Created file {self._out_file}", flush=True)
            os.close(fd)
            self.ALLOW[self.obj_type][self.to_type](
                self.obj,
                self._out_file,
            )
            return Path(self._out_file)
        else:
            return self.ALLOW[self.obj_type][self.to_type](
                self.obj,
                self._out_file,
            )

    def __exit__(self, type, value, traceback):
        if self._out_file is not None and os.path.exists(self._out_file):
            os.unlink(self._out_file)
            print(f"Deleted file {self._out_file}", flush=True)
        return False
