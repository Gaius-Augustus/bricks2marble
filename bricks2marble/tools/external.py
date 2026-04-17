import shutil
from pathlib import Path

TOOLS: dict[str, Path | None] = {
    "bgzip": None,
    "tabix": None,
    "gffcompare": None,
    "gtfToGenePred": None,
    "genePredSingleCover": None,
}


def _validate_tool(path: Path) -> None:
    if not path.exists() or path.is_dir():
        raise FileNotFoundError(
            f"Path {path} does not point to an existing executable"
        )


def configure(**tools) -> None:
    """Configure paths for external tools used in bricks2marble."""
    global TOOLS
    for tool, path in tools.items():
        if tool not in TOOLS:
            raise ValueError(f"{tool!r} is not a recognized tool name")
        path = Path(path).expanduser()
        _validate_tool(path)
        TOOLS[tool] = path


def get_tool_path(tool: str) -> Path:
    if tool not in TOOLS:
        raise ValueError(f"{tool!r} is not a recognized tool name")
    tool_ = TOOLS.get(tool)
    if tool_ is None:
        tool_ = shutil.which(tool)
        if tool_ is not None: return Path(tool_)
        raise ValueError(
            f"{tool!r} is requested but not found. Either add it to your PATH "
            f"or call bricks2marble.tools.configure({tool}='path/to/tool') "
            "at the start of your script."
        )
    return tool_
