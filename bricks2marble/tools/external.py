import shutil
from pathlib import Path

GFFCOMPARE: Path | None = None
GENEPREDTOGTF: Path | None = None
GTFTOGENEPRED: Path | None = None


def _validate_tool(path: Path) -> None:
    if not path.exists() or path.is_dir():
        raise FileNotFoundError(
            f"Path {path} does not point to an existing executable"
        )


def configure(
    gffcompare: str | Path | None = None,
    genePredToGtf: str | Path | None = None,
    gtfToGenePred: str | Path | None = None,
) -> None:
    """Configure paths for external tools used in bricks2marble."""
    global GFFCOMPARE, GENEPREDTOGTF, GTFTOGENEPRED
    if gffcompare is not None:
        gffcompare = Path(gffcompare).expanduser()
        _validate_tool(gffcompare)
        GFFCOMPARE = gffcompare

    if genePredToGtf is not None:
        genePredToGtf = Path(genePredToGtf).expanduser()
        _validate_tool(genePredToGtf)
        GENEPREDTOGTF = genePredToGtf

    if gtfToGenePred is not None:
        gtfToGenePred = Path(gtfToGenePred).expanduser()
        _validate_tool(gtfToGenePred)
        GTFTOGENEPRED = gtfToGenePred


def get_tool_path(tool: str) -> Path:
    global GFFCOMPARE, GENEPREDTOGTF, GTFTOGENEPRED
    match tool:
        case "gffcompare":
            tool_ = GFFCOMPARE
        case "genePredToGtf":
            tool_ = GENEPREDTOGTF
        case "gtfToGenePred":
            tool_ = GTFTOGENEPRED
        case _:
            raise ValueError(f"{tool!r} is not a recognized tool name")
    if tool_ is None:
        tool_ = shutil.which(tool)
        if tool_ is not None: return Path(tool_)
        raise ValueError(
            f"{tool!r} is requested but not found. Either add it to your PATH "
            f"or call bricks2marble.tools.configure({tool}='path/to/tool') "
            "at the start of your script."
        )
    return tool_
