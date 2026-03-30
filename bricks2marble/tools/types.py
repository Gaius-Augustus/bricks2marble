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

    ANNOTATION = 0
    GP = 1
    GTF = 2
    GFF3 = 3


def _convert_external(
    path_in: str | Path,
    path_out: str | Path,
    tool: str,
) -> None:
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


ALLOWED: dict[Types, dict] = {
    Types.ANNOTATION: {
        Types.GP: lambda i, o, **kwargs: i.to_genepred(
            o,
            append=kwargs.get("append", False),
        ),
        Types.GTF: lambda i, o, **kwargs: i.to_gtf(
            o,
            append=kwargs.get("append", False),
            source=kwargs.get("source", "bricks2marble"),
        ),
        Types.GFF3: lambda i, o, **kwargs: i.to_gff3(
            o,
            append=kwargs.get("append", False),
            source=kwargs.get("source", "bricks2marble"),
        ),
    },
    Types.GP: {
        Types.GTF: lambda i, o, **kwargs: load_annotation(i).to_gtf(
            o,
            append=kwargs.get("append", False),
            source=kwargs.get("source", "bricks2marble"),
        ),
        Types.GFF3: lambda i, o, **kwargs: load_annotation(i).to_gff3(
            o,
            append=kwargs.get("append", False),
            source=kwargs.get("source", "bricks2marble"),
        ),
        Types.ANNOTATION: lambda i, o, **kwargs: load_annotation(i),
    },
    Types.GTF: {
        Types.GP: lambda i, o, **kwargs: _convert_external(
            i, o, "gtfToGenePred",
        ),
    },
}


def _infer_type(obj) -> Types:
    suffixes = ["gff3", "gtf", "gp"]
    if isinstance(obj, str):
        if obj not in suffixes:
            obj = Path(obj).expanduser()
            if obj.suffix[1:] not in suffixes:
                raise ValueError(
                    f"Unsupported suffix for conversion: {obj.suffix[1:]}. "
                    f"Supported: {suffixes}"
                )

    if isinstance(obj, str):
        type_ = Types[obj.upper()]
    elif isinstance(obj, Path):
        if obj.suffix[1:] not in suffixes:
            raise ValueError(
                f"Unsupported suffix for conversion: {obj.suffix[1:]}. "
                f"Supported: {suffixes}"
            )
        type_ = Types[obj.suffix[1:].upper()]
    elif obj is Annotation or isinstance(obj, Annotation):
        type_ = Types.ANNOTATION
    else:
        raise ValueError(
            f"Cannot determine conversion suffix for type {type(obj)}"
        )
    return type_


def allowed(
    obj: Annotation | str | Path,
    to: Path | str | Literal["gp", "gtf", "gff3"] | type[Annotation],
    ignore: list[Types] | None = None
) -> tuple[Types, Types]:
    from_ = _infer_type(obj)
    to_ = _infer_type(to)

    if from_ is not to_ and (
        from_ not in ALLOWED or to_ not in ALLOWED[from_].keys()
    ) and (ignore is None or from_ not in ignore):
        raise ValueError(f"Unsupported conversion: {from_.name} -> {to_.name}")

    return from_, to_


def convert(
    obj: Annotation | str | Path,
    to: Path | str | type[Annotation],
    **kwargs,
) -> Path | Annotation:
    from_type, to_type = allowed(obj, to)
    returned = ALLOWED[from_type][to_type](obj, to, **kwargs)
    return returned if to is Annotation else Path(to)  # type: ignore


class Converter:
    """This class is used for temporary file type conversion. Use as a
    context manager to ensure proper removal of the file after usage.

    Example:
    ```python
        with Converter("path/to/my/file.gtf", "gp") as tmp_file_path:
            bricks2marble.io.load_annotation(tmp_file_path)
    ```

    Args:
        obj (Annotation, str or Path): Object to convert. Either a
            :class:`bricks2marble.struct.Annotation` object or a path to
            a file with an supported suffix.
        to (str or Annotation): Either target file suffix (e.g. "gp",
            "gtf") or :class:`bricks2marble.struct.Annotation`.
    """
    def __init__(
        self,
        obj: Annotation | str | Path,
        to: Literal["gp", "gtf", "gff3"] | type[Annotation],
        ignore: list[
            Literal["gp", "gtf", "gff3"] | type[Annotation]
        ] | None = None,
    ) -> None:
        self.obj = (
            Path(obj).expanduser() if isinstance(obj, (str, Path)) else obj
        )
        self.to = to
        self.ignore = [
            _infer_type(t) for t in ignore
        ] if ignore is not None else []
        self.obj_type, self.to_type = allowed(self.obj, self.to, self.ignore)

    def __enter__(self) -> Annotation | Path:
        self._out_file = None
        if self.obj_type is self.to_type or self.obj_type in self.ignore:
            return self.obj
        if self.to_type is not Types.ANNOTATION:
            fd, self._out_file = tempfile.mkstemp(
                suffix="."+self.to_type.name.lower(),
            )
            try:
                os.close(fd)
                out_path = ALLOWED[self.obj_type][self.to_type](
                    self.obj,
                    self._out_file,
                    False,
                )
                out_path = Path(self._out_file)
            except Exception as e:
                if self._out_file is not None and os.path.exists(
                    self._out_file
                ):
                    os.unlink(self._out_file)
                raise e
            return out_path
        else:
            return ALLOWED[self.obj_type][self.to_type](
                self.obj,
                self._out_file,
                False,
            )

    def __exit__(self, type, value, traceback):
        if self._out_file is not None and os.path.exists(self._out_file):
            os.unlink(self._out_file)
        return False
