import subprocess
from collections.abc import Generator
from pathlib import Path

from .external import get_tool_path
from .types import Converter


def compress_fasta(
    fasta: Path | str,
    create_copy: bool = False,
    threads: int = 1,
) -> None:
    """Calls `bgzip` to compress the given fasta path.

    Args:
        fasta (Path or str): Path to the fasta to compress.
        create_copy (bool, optional): If True, writes the compressed
            file to the same location with the added suffix `.gz`.
            Defaults to False, which means the fasta itself will be
            overridden by the `.gz` file.
    """
    bgzip = str(get_tool_path("bgzip"))
    command = [bgzip, "--threads", str(threads), str(fasta)]
    if create_copy:
        with open(f"{fasta}.gz", "wb") as out:
            subprocess.run(
                command+["-c"],
                stdout=out,
                check=True,
            )
    else:
        subprocess.run(command, check=True)


def create_genepred_index(
    gtf: Path | str,
    single_cover: bool = False,
    threads: int = 1,
) -> None:
    """For a given gtf file, creates an index of the genepred format
    version of this file.

    First, creates a temporary `.gp` file (extended genepred format).
    Then, sorts the file on transcript names and start positions and
    compresses it using `bgzip`. This `gp.gz` file will then be indexed
    using `tabix`, which creates a `.gp.gz.tbi` file.

    Requires the UCSC tools `gtfToGenePred` and optionally
    `genePredSingleCover`.
    Link: https://hgdownload.soe.ucsc.edu/admin/exe/linux.x86_64/

    Args:
        gtf (Path or str): Path to an annotation file in gtf format.
        single_cover (bool, optional): Whether to call also call
            `genePredSingleCover` for choosing only the longest isoforms
            of each transcript from the annotation. Defaults to False.
        threads (int, optional): Number of threads to use for `bgzip`.
            Defaults to 1.
    """
    gtf = Path(gtf)
    gp_gz = gtf.parent / (gtf.stem + ".gp.gz")
    bgzip = str(get_tool_path("bgzip"))
    tabix = str(get_tool_path("tabix"))
    if single_cover:
        genePredSingleCover = str(get_tool_path("genePredSingleCover"))

    with Converter(
        gtf, "gp",
        extended=True,
        ignore_groups_without_exons=True,
    ) as gp:
        if single_cover:
            subprocess.run([
                genePredSingleCover, str(gp), str(gp),
            ], check=True, stderr=subprocess.DEVNULL)
        with open(gp_gz, "wb") as gp_gz_out:
            sort_out = subprocess.run(
                ["sort", "-k2,2", "-k4,4n", str(gp)],
                stdout=subprocess.PIPE,
                check=True,
            )
            subprocess.run(
                [bgzip, "--threads", str(threads)],
                input=sort_out.stdout,
                stdout=gp_gz_out,
                check=True,
            )
        subprocess.run(
            [tabix, "-s", "2", "-b", "4", "-e", "5", "-c", "#", gp_gz],
            check=True,
        )


def tabix_query(
    file: Path | str,
    sequence: str,
    start: int,
    end: int,
) -> Generator[list[str], None, None]:
    """Call tabix and generate an array of strings for each line it
    returns. The indexing matches Python conventions (0-based,
    end-exclusive).
    """
    query = "{}:{}-{}".format(sequence, start, end)
    process = subprocess.run(
        ["tabix", "-f", file, query],
        capture_output=True,
        check=True,
        text=True,
    )
    for line in process.stdout.splitlines():
        yield line.strip().split()
