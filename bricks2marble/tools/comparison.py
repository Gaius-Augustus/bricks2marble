import re
import shutil
import subprocess
import tempfile
from bisect import bisect_right
from pathlib import Path
from typing import overload

from pydantic import BaseModel, ValidationError

from ..struct import Annotation
from .external import get_tool_path
from .types import Converter

Bin = tuple[float, float, int]
BinCounts = dict[str, list[Bin]]

_TMAP_ALIASES = {
    "Length": "len",
    "Number of exons": "num_exons",
    "FPKM": "FPKM",
    "TPM": "TPM",
    "Coverage": "cov",
    "Reference match length": "ref_match_len",
}


class CompareMetrics(BaseModel):

    sensitivity: float
    precision: float

    missed: float | None = None
    novel: float | None = None

    @property
    def F1(self) -> float:
        if self.precision == 0 or self.sensitivity == 0:
            return 0
        return (
            2 * self.precision * self.sensitivity
            / (self.precision + self.sensitivity)
        )


class AnnotationComparison(BaseModel):

    base: CompareMetrics
    exon: CompareMetrics
    intron: CompareMetrics
    intron_chain: CompareMetrics
    transcript: CompareMetrics
    locus: CompareMetrics

    annotation_loci: int = 0
    reference_loci: int = 0

    # Histogram of tmap entries per gffcompare class code, filled in
    # only when ``compare`` is called with ``track_bins``. For a single
    # binning column it is a :data:`BinCounts` (``{class_code: [(lo, hi,
    # n), ...]}``) and for several columns a dict of those keyed by
    # column name (``{"Length": {...}, "Number of exons": {...}}``).
    track: dict | None = None


def _place_query(src: Path, dst: Path) -> None:
    """Make ``src`` available at ``dst`` for gffcompare, so its
    ``.tmap``/``.refmap`` output lands next to ``dst``.

    Prefers a symlink to avoid copying the query, but falls back to a
    real copy on filesystems that do not support symlinks (some cluster
    mounts raise ``OSError`` on :meth:`~pathlib.Path.symlink_to`).
    """
    try:
        dst.symlink_to(src.resolve())
    except OSError:
        shutil.copyfile(src, dst)


def _bin_column(pairs: list[tuple[str, float]], bins: int) -> BinCounts:
    """Histogram ``(class_code, value)`` pairs into ``bins`` equal-width
    bins spanning the minimum to maximum value.

    Bins are left-closed/right-open, except the last bin which also
    includes the maximum. When every value is an integer, the reported
    edges are integers too.
    """
    values = [v for _, v in pairs]
    if not values: return {}

    lo, hi = min(values), max(values)
    span = hi - lo
    integral = all(float(v).is_integer() for v in values)
    edges: list[float] = [lo + i * span / bins for i in range(bins + 1)]
    if integral: edges = [int(round(edge)) for edge in edges]

    counts: dict[str, list[int]] = {}
    for code, value in pairs:
        idx = min(max(bisect_right(edges, value) - 1, 0), bins - 1)
        counts.setdefault(code, [0] * bins)[idx] += 1

    return {
        code: [(edges[i], edges[i+1], code_counts[i]) for i in range(bins)]
        for code, code_counts in counts.items()
    }


def _track_class_codes(
    tmap: Path,
    bin_col: list[str],
    bins: int,
) -> dict:
    """Read a gffcompare ``.tmap`` file and bin the number of entries
    per class code, for every requested column in ``bin_col``.

    Returns a :data:`BinCounts` for a single column, or a dict of those
    keyed by column name for several columns (see
    :attr:`AnnotationComparison.track`).
    """
    lines = tmap.read_text().splitlines()
    if not lines: return {}
    header = lines[0].split("\t")
    index = {name: i for i, name in enumerate(header)}
    code_col = index["class_code"]

    resolved = {}
    for col in bin_col:
        name = _TMAP_ALIASES.get(col, col)
        if name not in index:
            raise KeyError(
                f"Column {col!r} (tmap column {name!r}) not found in the "
                f"gffcompare tmap header {header}."
            )
        resolved[col] = index[name]

    rows = [line.split("\t") for line in lines[1:] if line]
    per_column = {
        col: _bin_column(
            [(row[code_col], float(row[i])) for row in rows], bins,
        )
        for col, i in resolved.items()
    }
    if len(bin_col) == 1:
        return per_column[bin_col[0]]
    return per_column


@overload
def compare(
    annotation: Annotation | Path | str,
    reference: Annotation | Path | str,
    e: int = ...,
    track_bins: int | None = ...,
    bin_col: list[str] = ...,
) -> AnnotationComparison:
    ...
@overload
def compare(
    annotation: list[Annotation | Path | str],
    reference: Annotation | Path | str,
    e: int = ...,
    track_bins: int | None = ...,
    bin_col: list[str] = ...,
) -> list[AnnotationComparison]:
    ...
def compare(
    annotation: Annotation | Path | str | list[Annotation | Path | str],
    reference: Annotation | Path | str,
    e: int = 0,
    track_bins: int | None = None,
    bin_col: list[str] = ["Length"],
) -> AnnotationComparison | list[AnnotationComparison]:
    """Compare two annotations with the tool gffcompare:
    https://github.com/gpertea/gffcompare

    Tell bricks2marble the path to this executable using
    ```
    bricks2marble.tools.configure(gffcompare="your/path")
    ```
    or add it to your system PATH.

    The result is an easily accessible collection of metrics returned by
    this program. The tool is called with additional arguments, equal to
    ```
    gffcompare --strict-match -e {e} -T -o {cache dir} \\
        -r {reference} {annotation}
    ```
    The temporary `cache dir` is created and deleted automatically. The
    `-T` flag (which suppresses the per-input `.tmap`/`.refmap` files)
    is dropped when `track_bins` is given, so the `.tmap` file can be
    read.

    Args:
        annotation (Annotation or Path): The annotation you want to
            compare to the reference. If an
            :class:`~bricks2marble.struct.annotation.Annotation` is
            given, this will be converted to a gtf file first with the
            `.to_gtf()` method. You can also supply a sequence of
            annotations, each of which are then compared to the
            reference and metrics for all of these comparisons are
            returned.
        reference (Annotation or Path): The reference annotation. If an
            :class:`~bricks2marble.struct.annotation.Annotation` is
            given, this will be converted to a gtf file first with the
            `.to_gtf()` method.
        gffcompare (Path, optional): Path to the executable gffcompare.
            Defaults to just 'gffcompare', if it is properly added to
            the path.
        e (int, optional): Option `e` of `gffcompare`. Maximum allowed
            range of terminal exons in reference transcripts. Defaults
            to 0.
        track_bins (int, optional): If given, additionally read the
            gffcompare `.tmap` file and count, per class code, how many
            transcripts fall into each of `track_bins` equal-width bins.
            The bins span the minimum to maximum value of the binning
            column(s). The result is stored on
            :attr:`AnnotationComparison.track`. Defaults to None (no
            tracking).
        bin_col (list[str], optional): The `.tmap` column(s) to bin by.
            Each entry is either a friendly name ("Length", "Number of
            exons", "FPKM", "TPM", "Coverage", "Reference match length")
            or a raw tmap header ("len", "num_exons", ...). With a
            single column, `track` is
            `{class_code: [(lo, hi, n), ...]}`; with several, it is a
            dict of those keyed by column name. Only used when
            `track_bins` is given. Defaults to `["Length"]`.

    Returns:
        AnnotationComparison: See
            :class:`bricks2marble.tools.gtf.AnnotationComparison`. If a
            sequence of annotations is given, instead is a sequence of
            comparisons.
    """
    seq_given = True
    if not isinstance(annotation, list):
        annotation = [annotation]
        seq_given = False

    results = []
    with Converter(reference, "gtf", ignore=["gff3"]) as reference_file:

        for i in range(len(annotation)):

            with tempfile.TemporaryDirectory() as cache_dir_str:
                cache_dir = Path(cache_dir_str)

                with Converter(
                    annotation[i], "gtf", ignore=["gff3"],
                ) as cache_file:
                    query = Path(cache_file)
                    if track_bins is not None:
                        # tmap file is written next to query annotation,
                        # so we copy/symlink it into the temporary path
                        local = cache_dir / query.name
                        _place_query(query, local)
                        query = local

                    args = [
                        f"{get_tool_path('gffcompare')}",
                        "--strict-match",
                        f"-e {e}",
                    ]
                    if track_bins is None: args.append("-T")
                    args += [
                        "-o",
                        cache_dir_str.rstrip("/") + "/",
                        "-r",
                        str(reference_file),
                        str(query),
                    ]
                    subprocess.run(
                        args, check=True, stderr=subprocess.DEVNULL,
                    )

                pattern = re.compile(
                    r'^\s*(.+?) level:\s+([\d.]+|-nan)\s+\|\s+([\d.]+|-nan)'
                )
                pattern_mn = re.compile(
                    r'^\s*(Missed|Novel) (.+?):[\s/\d.]+\(\s*([\d.]+)%\)'
                )
                # "#     Query mRNAs :  N in  M loci" and the analogous
                # "# Reference mRNAs :  N in  K loci" report how many
                # loci gffcompare loaded from each input.
                pattern_loci = re.compile(
                    r'(Query|Reference) mRNAs\s*:\s*\d+\s+in\s+(\d+)\s+loci'
                )

                results.append({
                    "base": {"sensitivity": 0, "precision": 0},
                    "exon": {"sensitivity": 0, "precision": 0},
                    "intron": {"sensitivity": 0, "precision": 0},
                    "intron_chain": {"sensitivity": 0, "precision": 0},
                    "transcript": {"sensitivity": 0, "precision": 0},
                    "locus": {"sensitivity": 0, "precision": 0},
                    "annotation_loci": 0,
                    "reference_loci": 0,
                })
                with open(cache_dir / ".stats", 'r', encoding='utf-8') as file:
                    for line in file.readlines():
                        match = pattern.match(line)
                        match_mn = pattern_mn.match(line)
                        if match is not None:
                            level = match.group(1).strip().lower()
                            level = level.replace(" ", "_")
                            sensitivity = float(match.group(2))
                            if match.group(3) == "-nan":
                                precision = 0
                            else:
                                precision = float(match.group(3))
                            results[-1][level] = {
                                'sensitivity': sensitivity / 100,
                                'precision': precision / 100,
                            }
                        if match_mn is not None:
                            level = match_mn.group(2).strip()[:-1]
                            if level == "loc": level = "locus"
                            mn = match_mn.group(1).lower()
                            perc = float(match_mn.group(3))
                            results[-1][level] = results[-1][level] | {
                                mn: round(perc / 100, 3),
                            }
                        match_loci = pattern_loci.search(line)
                        if match_loci is not None:
                            n_loci = int(match_loci.group(2))
                            if match_loci.group(1) == "Query":
                                results[-1]["annotation_loci"] = n_loci
                            else:
                                results[-1]["reference_loci"] = n_loci

                if track_bins is not None:
                    # gffcompare writes one ".<query>.tmap" per input.
                    tmaps = list(cache_dir.glob("*.tmap"))
                    if not tmaps:
                        raise FileNotFoundError(
                            "gffcompare did not produce a .tmap file in "
                            f"{cache_dir}."
                        )
                    results[-1]["track"] = _track_class_codes(
                        tmaps[0], bin_col, track_bins,
                    )

    try:
        if seq_given:
            return [AnnotationComparison(**result) for result in results]
        else:
            return AnnotationComparison(**results[0])
    except ValidationError:
        return AnnotationComparison(
            base=CompareMetrics(sensitivity=0, precision=0),
            exon=CompareMetrics(sensitivity=0, precision=0),
            intron=CompareMetrics(sensitivity=0, precision=0),
            intron_chain=CompareMetrics(sensitivity=0, precision=0),
            transcript=CompareMetrics(sensitivity=0, precision=0),
            locus=CompareMetrics(sensitivity=0, precision=0),
        )


def annotation_diff(a: Annotation, b: Annotation) -> Annotation:
    """Returns all transcripts in `a` that are not in `b`. Transcript
    names are ignored.
    """
    b_transcripts = set(b.transcripts())
    result = Annotation()
    for tx in a.transcripts():
        if tx not in b_transcripts:
            result.add(tx)
    return result


def annotation_and(a: Annotation, b: Annotation) -> Annotation:
    """Returns all transcripts present in both `a` and `b`. Transcript
    names are ignored.
    """
    b_transcripts = set(b.transcripts())
    result = Annotation()
    for tx in a.transcripts():
        if tx in b_transcripts:
            result.add(tx)
    return result


def annotation_or(a: Annotation, b: Annotation) -> Annotation:
    """Returns all transcripts present in either `a` or `b`, with
    duplicates removed. Transcript names are ignored.
    """
    result = Annotation()
    seen = set()
    for ann in (a, b):
        for tx in ann.transcripts():
            if tx not in seen:
                seen.add(tx)
                result.add(tx)
    return result
