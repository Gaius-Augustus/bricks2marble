import bisect
import re
from enum import Enum
from typing import Literal

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
        attributes: bool = True
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
        id (str): The identifier of the transcript.
        gene_id (str): The identifier of the gene the transcript is
            synthesized from.
        seqname (str): The name of the sequence or chromosome the
            transcript is coming from.
        strand (str, optional): Strand (+/-) on which the transctipt is
            located. Defaults to +.
    """

    def __init__(
        self,
        id: str,
        gene_id: str,
        seqname: str,
        strand: Literal["+", "-"] = "+",
    ) -> None:
        self.id = id
        self.seqname = seqname
        self.gene_id = gene_id
        # TODO change to:
        # self.entries: list[GTFEntry] = []
        # self.features: dict[FeatureType, list[int]] = {
        #     t: [] for t in FeatureType
        # }
        self.entries: dict[FeatureType, list[GTFEntry]] = {}
        self.start = -1
        self.end = -1
        self.cds_len = -1
        self.strand: Literal["+", "-"] = strand
        self.source = ""

    def add(self, entry: GTFEntry | list[str]) -> None:
        """Adds a :class:`GTFEntry` object to the transcript."""
        entry = GTFEntry.from_list(entry) if isinstance(entry, list) else entry
        if not (
            entry.name == self.seqname
            and entry.strand == self.strand
            and self.gene_id == entry.attribute("gene_id")
        ):
            raise RuntimeError(
                f"Entry {entry} does not match current transcript description"
            )

        if entry.feature not in self.entries:
            self.entries[entry.feature] = []
        insort_index = bisect.bisect(
            self.entries[entry.feature],
            (entry.start, entry.end),
            key=lambda x: (x.start, x.end),
        )
        if insort_index >= len(self.entries[entry.feature]):
            self.entries[entry.feature].insert(insort_index, entry)
        elif entry.same_as(
            self.entries[entry.feature][insort_index],
            borders=False,
        ) and entry.start - 1 == self.entries[entry.feature][insort_index].end:
            self.entries[entry.feature][insort_index].end = entry.end
        else:
            self.entries[entry.feature].insert(insort_index, entry)

        if self.start < 0 or entry.start < self.start:
            self.start = entry.start
        if self.end < 0 or entry.end > self.end:
            self.end = entry.end
        self.source = entry.source

    def coords_per_frame(
        self,
        type: FeatureType,
    ) -> dict[str, list[tuple[int, int]]]:
        """Get the coordinates of the regions of given type per reading
        frame.

        Args:
            type (FeatureType): The type of feature to extract from the
                transcript.
        Returns:
            dict[str,list[tuple[int,int]]]: Dictionary mapping frame
                phases (0, 1, 2) to a list of integer ranges ``(start,
                end)`` of the entries in the transcript with the given
                type.
        """
        coords = {'0' : [], '1' : [], '2' : [], '.' : []}

        if type == FeatureType.CDS and type not in self.entries:
            type = FeatureType.Exon
        if type not in self.entries:
            return coords

        for entry in self.entries[type]:
            coords["." if entry.frame is None else str(entry.frame)].append(
                (entry.start, entry.end)
            )
        if type == FeatureType.CDS:
            coords['0'] += coords['.']
            del coords['.']

        return coords

    def coords(self, type: FeatureType) -> list[tuple[int, int]]:
        """Get the coordinates of the regions of given type.

        Args:
            type (FeatureType): The type of feature to extract from the
                transcript.
        Returns:
            list[tuple[int,int]]: List of integer ranges ``(start,
                end)`` of the entries in the transcript with the given
                type.
        """
        coords = []
        if type == FeatureType.CDS and type not in self.entries.keys():
            type = FeatureType.Exon
        if type not in self.entries:
            return coords

        for entry in self.entries[type]:
            coords.append((entry.start, entry.end))
        return coords

    def get_cds_len(self):
        cds = self.coords(FeatureType.CDS)
        return sum([c[1] - c[0] + 1 for c in cds])

    def get_cds_coords(self) -> dict[str, list[tuple[int, int]]]:
        """Get the coordinates and reading frame of the coding regions

        Returns:
            dict[str,list[tuple[int,int]]]: Dictionary with list of CDS
                coords for each each frame phase (0, 1, 2).
        """
        cds_coords = {'0' : [], '1' : [], '2' : []}

        if FeatureType.CDS in self.entries:
            key = FeatureType.CDS
        else:
            key = FeatureType.Exon

        for entry in self.entries[key]:
            frame = "." if entry.frame is None else str(entry.frame)
            cds_coords[frame].append((entry.start, entry.end))
        return cds_coords

    def finalize(self) -> bool:
        """Add transcript, intron, CDS, exon coordinates if they were
        not included in the gtf file.

        Returns:
            bool: False if no cds were found for the tx, True otherwise.
        """
        if (FeatureType.CDS not in self.entries
                and FeatureType.Exon not in self.entries):
            return False
        self.find_introns()
        self.find_transcript()
        self.find_start_stop_codon()
        self.fix_cds_frames()
        return True

    def find_introns(self) -> None:
        """Add intron lines if none were given to this transcript."""
        if FeatureType.Intron not in self.entries:
            self.entries[FeatureType.Intron] = []

            key = None
            if FeatureType.CDS in self.entries.keys():
                key = FeatureType.CDS
            elif FeatureType.Exon in self.entries.keys():
                key = FeatureType.Exon

            if key is not None:
                exon_lst: list[GTFEntry] = []
                for line in self.entries[key]:
                    exon_lst.append(line)
                for i in range(1, len(exon_lst)):
                    intron = GTFEntry(
                        name=exon_lst[i].name,
                        source=exon_lst[i].source,
                        feature=FeatureType.Intron,
                        start=exon_lst[i-1].end + 1,
                        end=exon_lst[i].start - 1,
                        score=exon_lst[i].score,
                        strand=exon_lst[i].strand,
                        frame=exon_lst[i].frame,
                        attributes=f"gene_id \"{self.gene_id}\"; "
                                   f"transcript_id \"{self.id}\";",
                    )
                    self.add(intron)

    def find_transcript(self) -> None:
        """Add an entry for the transcript itself if missing."""
        if FeatureType.Transcript not in self.entries:
            for key in self.entries.keys():
                for entry in self.entries[key]:
                    if entry.start < self.start or self.start < 0:
                        self.start = entry.start
                    if entry.end > self.end:
                        self.end = entry.end
            entry = GTFEntry(
                name=self.seqname,
                source=entry.source,
                feature=FeatureType.Transcript,
                start=self.start,
                end=self.end,
                score=None,
                strand=entry.strand,
                frame=None,
                attributes=f"gene_id \"{self.gene_id}\"; "
                            f"transcript_id \"{self.id}\";",
            )
            self.add(entry)

    def find_start_stop_codon(self) -> None:
        """Add start- and stop-codon entries if they are missing."""
        if (FeatureType.StartCodon in self.entries
                and FeatureType.StopCodon in self.entries):
            return

        key = None
        if FeatureType.CDS in self.entries:
            key = FeatureType.CDS
        elif FeatureType.Exon in self.entries:
            key = FeatureType.Exon
        if key is None:
            return

        tx = self.entries[key][0]
        line1 = GTFEntry(
            name=self.seqname,
            source=tx.source,
            feature=FeatureType.Unknown,
            start=tx.start,
            end=tx.start + 2,
            score=None,
            strand=self.strand,
            frame=0,
            attributes=f"gene_id \"{self.gene_id}\"; "
                        f"transcript_id \"{self.id}\";",
        )
        tx = self.entries[key][-1]
        line2 = GTFEntry(
            name=self.seqname,
            source=tx.source,
            feature=FeatureType.Unknown,
            start=tx.end - 2,
            end=tx.end,
            score=None,
            strand=self.strand,
            frame=0,
            attributes=f"gene_id \"{self.gene_id}\"; "
                        f"transcript_id \"{self.id}\";",
        )

        fragmented = True
        if tx.strand == '+':
            line1.feature = FeatureType.StartCodon
            line2.feature = FeatureType.StopCodon
            if self.entries[key][0].frame == 0:
                fragmented = False
            start = line1
            stop = line2
        else:
            line1.feature = FeatureType.StopCodon
            line2.feature = FeatureType.StartCodon
            if self.entries[key][-1].frame == 0:
                fragmented = False
            stop = line1
            start = line2
        if FeatureType.StartCodon not in self.entries and not fragmented:
            self.add(start)
        if FeatureType.StopCodon not in self.entries:
            self.add(stop)

    def fix_cds_frames(self) -> None:
        """Forces frames of the CDS entries in the transcript to be
        3-periodic.
        """
        if FeatureType.CDS in self.entries:
            phase = 0
            for line in self.entries[FeatureType.CDS]:
                line.frame = phase  # type: ignore
                phase = (3 - (line.end - line.start + 1 - phase) % 3) % 3

    def _check_splits(self) -> None:
        # method obsolete, will be checked at insert
        for k in self.entries:
            new_list = [self.entries[k][0]]
            for i in range(1, len(self.entries[k])):
                # TODO what if two exons are in different phases?
                # -> they will still be merged...
                if new_list[-1].end == self.entries[k][i].start-1:
                    new_list[-1].end = self.entries[k][i].end
                else:
                    new_list.append(self.entries[k][i])
            self.entries[k] = new_list

    def to_list(self, prefix: str = "") -> list[GTFEntry]:
        """Creates gtf output for the transcript.

        Returns:
            list[GTFEntry]: List of :class:`GTFEntry` objects.
        """
        gtf: list[GTFEntry] = []
        if prefix:
            prefix += '.'
        tx_line: GTFEntry | None = None
        for k in self.entries:
            for i, entry in enumerate(self.entries[k]):

                if k == FeatureType.Transcript:
                    tx_line = entry
                    tx_line.attributes = (
                        f"gene_id \"{self.gene_id}\"; "
                        f"transcript_id \"{prefix + self.id}\";"
                    )
                    continue

                elif k == FeatureType.CDS:
                    cds_type = 'internal'
                    if len(self.entries[k]) == 1:
                        cds_type = 'single'
                    elif ((i == 0 and self.strand == '+')
                            or (i == len(self.entries[k]) - 1
                                    and self.strand == '-')
                    ):
                        cds_type = 'initial'
                    elif ((i == len(self.entries[k]) - 1
                                and self.strand == '+')
                            or (i == 0 and self.strand == '-')
                    ):
                        cds_type = 'terminal'
                    # TODO why is cds_type formatted differently?
                    entry.attributes = (
                        f"gene_id \"{self.gene_id}\"; "
                        f"transcript_id \"{prefix + self.id}\"; "
                        f"cds_type={cds_type};"
                    )

                elif k not in [FeatureType.Transcript, FeatureType.Gene]:
                    entry.attributes = (
                        f"gene_id \"{self.gene_id}\"; "
                        f"transcript_id \"{prefix + self.id}\";"
                    )

                gtf.append(entry)

        if FeatureType.Exon not in self.entries:
            # TODO is exon and CDS interchangable? or is only one of
            # them needed for output?
            for entry in self.entries[FeatureType.CDS]:
                gtf.append(GTFEntry(
                    **(entry.model_dump() | {"feature": FeatureType.Exon})
                ))

        gtf.sort(key=lambda x: (x.start, x.end))
        if tx_line is not None:
            gtf = [tx_line] + gtf
        return gtf
