import re
import subprocess
import tempfile
from pathlib import Path
from typing import overload

from pydantic import BaseModel, ValidationError

from ..struct import Annotation
from .external import get_tool_path
from .types import Converter


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


@overload
def compare(
    annotation: Annotation | Path | str,
    reference: Annotation | Path | str,
    e: int = ...,
) -> AnnotationComparison:
    ...
@overload
def compare(
    annotation: list[Annotation | Path | str],
    reference: Annotation | Path | str,
    e: int = ...,
) -> list[AnnotationComparison]:
    ...
def compare(
    annotation: Annotation | Path | str | list[Annotation | Path | str],
    reference: Annotation | Path | str,
    e: int = 0,
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
    The temporary `cache dir` is created and deleted automatically.

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
                    subprocess.run([
                        f"{get_tool_path('gffcompare')}",
                        "--strict-match",
                        f"-e {e}",
                        "-T",
                        "-o",
                        cache_dir_str.rstrip("/") + "/",
                        "-r",
                        str(reference_file),
                        str(cache_file),
                    ], check=True, stderr=subprocess.DEVNULL)

                pattern = re.compile(
                    r'^\s*(.+?) level:\s+([\d.]+|-nan)\s+\|\s+([\d.]+|-nan)'
                )
                pattern_mn = re.compile(
                    r'^\s*(Missed|Novel) (.+?):[\s/\d.]+\(\s*([\d.]+)%\)'
                )

                results.append({})
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
    b_transcripts = {tx for chrom_ann in b for tx in chrom_ann}
    result = Annotation()
    for chrom_ann in a:
        for tx in chrom_ann:
            if tx not in b_transcripts:
                result.add(tx)
    return result


def annotation_and(a: Annotation, b: Annotation) -> Annotation:
    """Returns all transcripts present in both `a` and `b`. Transcript
    names are ignored.
    """
    b_transcripts = {tx for chrom_ann in b for tx in chrom_ann}
    result = Annotation()
    for chrom_ann in a:
        for tx in chrom_ann:
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
        for chrom_ann in ann:
            for tx in chrom_ann:
                if tx not in seen:
                    seen.add(tx)
                    result.add(tx)
    return result
