import bisect
import re
import warnings
from enum import Enum
from typing import Callable, Literal

from .fasta import Region


class FeatureType(Enum):

    CDS = "CDS"
    Exon = "exon"
    Gene = "gene"
    Intron = "intron"
    Transcript = "transcript"
    StartCodon = "start_codon"
    StopCodon = "stop_codon"
    Unknown = ""


class GTFEntry(Region):

    source: str
    feature: FeatureType
    score: int | None = None
    frame: Literal[0, 1, 2] | None = None
    attributes: str

    @staticmethod
    def from_list(line: list[str]) -> "GTFEntry":
        return GTFEntry(
            name=line[0],
            source=line[1],
            feature=FeatureType(line[2]),
            start=int(line[3]),
            end=int(line[4]),
            score=int(line[5]) if line[5] != "." else None,
            strand=line[6],  # type: ignore
            frame=int(line[7]) if line[7] != "." else None,  # type: ignore
            attributes=line[8],
        )

    def to_list(self) -> list[str | int]:
        return [
            self.name,
            self.source,
            self.feature.value,
            self.start,
            self.end,
            self.score if self.score is not None else ".",
            self.strand,
            self.frame if self.frame is not None else ".",
            self.attributes,
        ]

    def attribute(self, key: str) -> str:
        pattern = rf'\b{re.escape(key)}\s+"([^"]*)";'
        match = re.search(pattern, self.attributes)
        if match:
            return match.group(1).strip()
        raise AttributeError(
            f"Attribute {key!r} not found in {self.feature!r}"
        )

    def same_as(
        self,
        entry: "GTFEntry",
        borders: bool = True,
        attributes: bool = True,
    ) -> bool:
        comp = (self.name == entry.name and self.source == entry.source
                and self.feature == entry.feature and self.score == entry.score
                and self.strand == entry.strand and self.frame == entry.frame)
        if attributes:
            comp = comp and (self.attributes == entry.attributes)
        if borders:
            comp = comp and (
                self.start == entry.start and self.end == entry.end
            )
        return comp

    def __eq__(self, other) -> bool:
        if isinstance(other, GTFEntry):
            return super().__eq__(other)
        raise NotImplementedError("Can only compare GTFEntry to GTFEntry type")


class Transcript:
    """Class representing a transcript of a genome.

    Args:
        id (str): The name of the transcript.
    """

    def __init__(self, id: str) -> None:
        self.id = id
        self.seqname = ""
        self.gene_id = ""
        self.source = ""
        self.strand: Literal["+", "-"] = "+"
        self.entries: list[GTFEntry] = []
        self.features: dict[FeatureType, int] = {
            FeatureType.CDS: 0,
            FeatureType.Exon: 0,
            FeatureType.Intron: 0,
            FeatureType.Transcript: 0,
            FeatureType.StartCodon: 0,
            FeatureType.StopCodon: 0,
        }
        self.start = -1
        self.end = -1
        self.cds_len = -1

    def add(self, entry: GTFEntry) -> None:
        """Adds a :class:`GTFEntry` object to the transcript."""
        if len(self.entries) == 0:
            self.seqname = entry.name
            self.gene_id = entry.attribute("gene_id")
            self.strand = entry.strand
            self.source = entry.source

        if not (
            entry.name == self.seqname
            and entry.strand == self.strand
            and self.gene_id == entry.attribute("gene_id")
        ):
            raise RuntimeError(
                f"Entry {entry.feature}[{entry.start}:{entry.end}] does not "
                "match current transcript description"
            )

        if self.start < 0 or entry.start < self.start:
            self.start = entry.start
        if self.end < 0 or entry.end > self.end:
            self.end = entry.end

        if entry.feature == FeatureType.Transcript:
            return

        insort_index = bisect.bisect(
            self.entries,
            (entry.start, entry.end),
            key=lambda x: (x.start, x.end),
        )

        if insort_index >= len(self.entries):
            self.entries.insert(insort_index, entry)
        elif entry.same_as(
            self.entries[insort_index],
            borders=False,
        ) and entry.start - 1 == self.entries[insort_index].end:
            self.entries[insort_index].end = entry.end
        else:
            self.entries.insert(insort_index, entry)

        self.features[entry.feature] += 1

    def rename(self, name: str | Callable[[str], str]) -> None:
        """Changes the name of the sequence this transcript is located
        in. Also changes the sequence names in all contained GTF
        entries.

        Args:
            name (str | callable): If a string is given, changes the
                sequence name to that string. If a callable is given,
                this callable is applied to the sequence name, which is
                then set to the returned string.
        """
        if isinstance(name, str):
            rename = lambda _: name
        else:
            rename = name

        for entry in self.entries:
            entry.name = rename(entry.name)
        self.seqname = rename(self.seqname)

    def coords(self, feature: FeatureType) -> list[tuple[int, int]]:
        """Get the coordinates of the regions of the given
        :class:`FeatureType`.

        Args:
            feature (FeatureType): The type of feature to extract from
                the transcript.

        Returns:
            list[tuple[int,int]]: List of integer ranges ``(start,
                end)`` of the entries in the transcript with the given
                type.
        """
        if self.features[feature] == 0:
            return []

        coords = []
        for entry in self.entries:
            if entry.feature == feature:
                coords.append((entry.start, entry.end))
        return coords

    def get_cds_len(self):
        cds = self.coords(FeatureType.CDS)
        return sum([c[1] - c[0] + 1 for c in cds])

    def finalize(self, min_cds_length: int | None = None) -> bool:
        """Cleans up the entries in the Transcript by adding missing
        :class:`GTFEntry` objects.

        Args:
            min_cds_length (int, optional): Minimal length of coding
                regions. If this transcript has a shorter coding region,
                the method returns false. Defaults to no checks.
        """
        if self.features[FeatureType.CDS]==self.features[FeatureType.Exon]==0:
            warnings.warn(
                f"There are no CDS or exon entries in transcript {self.id}."
            )
        if min_cds_length is not None and self.get_cds_len() < min_cds_length:
            return False
        self.find_introns()
        self.fix_cds_frames()
        self.find_start_stop_codon()
        return True

    def find_introns(self) -> None:
        """If no intron entries were given to the transcript, they will
        be inferred by surrounding CDS or exon entries.
        """
        if self.features[FeatureType.Intron] > 0:
            return

        key = None
        if self.features[FeatureType.CDS] > 0:
            key = FeatureType.CDS
        elif self.features[FeatureType.Exon] > 0:
            key = FeatureType.Exon
        if key is None:
            return

        exon_lst: list[GTFEntry] = []
        for entry in self.entries:
            if entry.feature == key: exon_lst.append(entry)
        for i in range(1, len(exon_lst)):
            intron = GTFEntry(**(exon_lst[i].model_dump() | {
                "feature": FeatureType.Intron,
                "start": exon_lst[i-1].end,
            }))
            self.add(intron)

    def find_start_stop_codon(self) -> None:
        """Add start- and stop-codon entries if they are missing."""
        if (self.features[FeatureType.StartCodon] > 0
                and self.features[FeatureType.StopCodon] > 0):
            return

        key = None
        if self.features[FeatureType.CDS] > 0:
            key = FeatureType.CDS
        elif self.features[FeatureType.Exon] > 0:
            key = FeatureType.Exon
        if key is None:
            return

        line1 = GTFEntry(
            name=self.seqname,
            source=self.source,
            feature=FeatureType.Unknown,
            start=self.start,
            end=self.start + 3,
            score=None,
            strand=self.strand,
            frame=0,
            attributes=f"gene_id \"{self.gene_id}\"; "
                        f"transcript_id \"{self.id}\";",
        )
        line2 = GTFEntry(
            name=self.seqname,
            source=self.source,
            feature=FeatureType.Unknown,
            start=self.end - 3,
            end=self.end,
            score=None,
            strand=self.strand,
            frame=0,
            attributes=f"gene_id \"{self.gene_id}\"; "
                        f"transcript_id \"{self.id}\";",
        )

        fragmented = True
        if self.strand == '+':
            line1.feature = FeatureType.StartCodon
            line2.feature = FeatureType.StopCodon
            for entry in self.entries:
                if entry.feature == key:
                    if entry.frame == 0:
                        fragmented = False
                    break
            start = line1
            stop = line2
        else:
            line1.feature = FeatureType.StopCodon
            line2.feature = FeatureType.StartCodon
            for entry in reversed(self.entries):
                if entry.feature == key:
                    if entry.frame == 0:
                        fragmented = False
                    break
            stop = line1
            start = line2

        if self.features[FeatureType.StartCodon] == 0 and not fragmented:
            self.add(start)
        if self.features[FeatureType.StopCodon] == 0:
            self.add(stop)

    def fix_cds_frames(self) -> None:
        """Forces frames of the CDS entries in the transcript to be
        3-periodic.
        """
        if self.features[FeatureType.CDS] == 0:
            return

        phase = 0
        iter_ = self.entries if self.strand == "+" else reversed(self.entries)
        for entry in iter_:
            if entry.feature == FeatureType.CDS:
                entry.frame = phase  # type: ignore
                phase = (3 - (entry.end - entry.start - phase) % 3) % 3

    def to_list(self) -> list[GTFEntry]:
        """Creates GTF entries from the transcript. Here, coordinates of
        the inner objects will be transformed to match GTF indexing
        conventions, i.e. start at 1 and last index is inclusive.

        Returns:
            list[GTFEntry]: List of :class:`GTFEntry` objects.
        """
        gtf: list[GTFEntry] = []
        idx_cds = 0
        for entry in self.entries:
            if entry.feature == FeatureType.CDS:
                cds_type = 'internal'
                if self.features[FeatureType.CDS] == 1:
                    cds_type = 'single'
                elif ((idx_cds == 0 and self.strand == '+')
                        or (idx_cds == self.features[FeatureType.CDS] - 1
                                and self.strand == '-')
                ):
                    cds_type = 'initial'
                elif ((idx_cds == self.features[FeatureType.CDS] - 1
                            and self.strand == '+')
                        or (idx_cds == 0 and self.strand == '-')
                ):
                    cds_type = 'terminal'
                entry.attributes = (
                    f"gene_id \"{self.gene_id}\"; "
                    f"transcript_id \"{self.id}\"; "
                    f"cds_type \"{cds_type}\";"
                )
                idx_cds += 1
            else:
                entry.attributes = (
                    f"gene_id \"{self.gene_id}\"; "
                    f"transcript_id \"{self.id}\";"
                )

            entry_copy = entry.model_copy()
            entry_copy.start += 1
            gtf.append(entry_copy)

        if self.features[FeatureType.Exon] == 0:
            for entry in self.entries:
                if entry.feature == FeatureType.CDS:
                    exon = GTFEntry(**(
                        entry.model_dump() | {"feature": FeatureType.Exon}
                    ))
                    exon.start += 1
                    gtf.append(exon)

        tx_entry = GTFEntry(
            name=self.seqname,
            source=self.source,
            feature=FeatureType.Transcript,
            start=self.start+1,
            end=self.end,
            score=None,
            strand=self.strand,
            frame=None,
            attributes=f"gene_id \"{self.gene_id}\"; "
                        f"transcript_id \"{self.id}\";",
        )
        gtf.insert(0, tx_entry)
        return gtf
