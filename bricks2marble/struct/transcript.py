from typing import Literal

from .fasta import Region


class GTFEntry(Region):

    source: str
    feature: str
    score: int | Literal["."]
    frame: Literal["0", "1", "2", "."]
    attributes: str

    @staticmethod
    def from_list(line: list[str]) -> "GTFEntry":
        return GTFEntry(
            name=line[0],
            source=line[1],
            feature=line[2],
            start=int(line[3]),
            end=int(line[4]),
            score=int(line[5]) if line[5] != "." else ".",
            strand=line[6],  # type: ignore
            frame=line[7],  # type: ignore
            attributes=line[8],
        )

    def to_list(self) -> list[str | int]:
        return [
            self.name,
            self.source,
            self.feature,
            self.start,
            self.end,
            self.score,
            self.strand,
            self.frame,
            self.attributes,
        ]

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
        chr (str): The name of the sequence or chromosome the transcript
            is coming from.
        strand (str, optional): Strand (+/-) on which the transctipt is
            located. Defaults to +.
    """

    def __init__(
        self,
        id: str,
        gene_id: str,
        chr: str,
        strand: Literal["+", "-"] = "+",
    ) -> None:
        self.id = id
        self.chr = chr
        self.gene_id = gene_id
        self.entries: dict[str, list[GTFEntry]] = {}
        self.start = -1
        self.end = -1
        self.cds_len = -1
        self.strand: Literal["+", "-"] = strand
        self.source = ""

    def add(self, entry: GTFEntry | list[str]) -> None:
        """Add a :class:`GTFEntry` object to the transcript."""
        entry = GTFEntry.from_list(entry) if isinstance(entry, list) else entry
        if not (entry.name == self.chr or entry.strand == self.strand):
            raise RuntimeError(
                "File is not in gtf format. "
                f"Error in line {entry}\n"
                "Transcript ID is not unique"
            )

        if entry.feature not in self.entries:
            self.entries[entry.feature] = []

        if self.start < 0 or entry.start < self.start:
            self.start = entry.start
        if self.end < 0 or entry.end > self.end:
            self.end = entry.end

        if self.gene_id == "" and entry.feature != 'transcript':
            self.gene_id = entry.attributes.split(
                'gene_id "'
            )[1].split('";')[0]

        self.entries[entry.feature].append(entry)
        self.source = entry.source

    def coords_per_frame(self, type: str) -> dict[str, list[tuple[int, int]]]:
        """Get the coordinates of the regions of given type per reading
        frame.

        Args:
            type (str): The type of feature to extract from the
                transcript.
        Returns:
            dict[str,list[tuple[int,int]]]: Dictionary mapping frame
                phases (0, 1, 2) to a list of integer ranges ``(start,
                end)`` of the entries in the transcript with the given
                type.
        """
        coords = {'0' : [], '1' : [], '2' : [], '.' : []}

        if type == 'CDS' and type not in self.entries.keys():
            type = 'exon'
        if type not in self.entries.keys():
            return coords

        for entry in self.entries[type]:
            coords[entry.frame].append([entry.start, entry.end])

        for k in coords.keys():
            coords[k].sort(key=lambda c: (c[0], c[1]))
        if type == 'CDS':
            coords['0'] += coords['.']
            del coords['.']

        return coords

    def coords(self, type: str) -> list[tuple[int, int]]:
        """Get the coordinates of the regions of given type.

        Args:
            type (str): The type of feature to extract from the
                transcript.
        Returns:
            list[tuple[int,int]]: List of integer ranges ``(start,
                end)`` of the entries in the transcript with the given
                type.
        """
        coords = []
        if type == 'CDS' and type not in self.entries.keys():
            type = 'exon'
        if type not in self.entries.keys():
            return coords

        for entry in self.entries[type]:
            coords.append([entry.start, entry.end])

        coords.sort(key=lambda c: (c[0], c[1]))
        return coords

    def get_cds_len(self):
        cds = self.coords('CDS')
        return sum([c[1] - c[0] + 1 for c in cds])

    def get_cds_coords(self) -> dict[str, list[list[int]]]:
        """Get the coordinates and reading frame of the coding regions

        Returns:
            dict[str,list[list[int]]]: Dictionary with list of CDS
                coords for each each frame phase (0, 1, 2).
        """
        cds_coords = {'0' : [], '1' : [], '2' : []}

        if 'CDS' in self.entries.keys():
            key = 'CDS'
        else:
            key = 'exon'

        for entry in self.entries[key]:
            cds_coords[entry.frame].append([entry.start, entry.end])
        for k in cds_coords.keys():
            cds_coords[k].sort(key=lambda c: (c[0], c[1]))

        return cds_coords

    def add_missing_lines(self) -> bool:
        """Add transcript, intron, CDS, exon coordinates if they were
        not included in the gtf file.

        Returns:
            bool: False if no cds were found for the tx, True otherwise.
        """
        self.find_introns()
        if not self.check_cds_exons():
            return False
        self.find_transcript()
        self.find_start_stop_codon()
        return True

    def check_cds_exons(self) -> bool:
        """Check if the transcript has CDS or exons."""
        if ('CDS' not in self.entries and 'exon' not in self.entries):
            print(f'Skipping transcript {self.id}, no CDS nor exons')
            return False
        return True

    def find_introns(self) -> None:
        """Add intron lines."""
        if 'intron' not in self.entries:
            self.entries.update({'intron' : []})
            key = ''

            if 'CDS' in self.entries.keys():
                key = 'CDS'
            elif 'exon' in self.entries.keys():
                key = 'exon'

            if key:
                exon_lst: list[GTFEntry] = []
                for line in self.entries[key]:
                    exon_lst.append(line)
                exon_lst = sorted(exon_lst, key=lambda e: e.name)
                for i in range(1, len(exon_lst)):
                    intron = GTFEntry(
                        name=exon_lst[i].name,
                        source=exon_lst[i].source,
                        feature="intron",
                        start=exon_lst[i-1].end + 1,
                        end=exon_lst[i].start - 1,
                        score=exon_lst[i].score,
                        strand=exon_lst[i].strand,
                        frame=exon_lst[i].frame,
                        attributes=f"gene_id \"{self.gene_id}\"; "
                                   f"transcript_id \"{self.id}\";",
                    )
                    self.entries['intron'].append(intron)

    def find_transcript(self) -> None:
        """Add transcript lines."""
        if not 'transcript' in self.entries.keys():
            for key in self.entries.keys():
                for entry in self.entries[key]:
                    if entry.start < self.start or self.start < 0:
                        self.start = entry.start
                    if entry.end > self.end:
                        self.end = entry.end
            entry = GTFEntry(
                name=self.chr,
                source=entry.source,
                feature='transcript',
                start=self.start,
                end=self.end,
                score='.',
                strand=entry.strand,
                frame='.',
                attributes=self.id,
            )
            self.add(entry)

    def find_start_stop_codon(self) -> None:
        """Add start/stop codon lines."""

        if not 'start_codon' in self.entries:
            self.entries.update({'start_codon' : []})
        if not 'stop_codon' in self.entries:
            self.entries.update({'stop_codon' : []})

        key = None
        if 'CDS' in self.entries:
            key = 'CDS'
        elif 'exon' in self.entries:
            key = 'exon'
        if key is None:
            return

        self.entries[key].sort(key = lambda x: x.start)
        tx = self.entries[key][0]
        line1 = GTFEntry(
            name=self.chr,
            source=tx.source,
            feature="",
            start=tx.start,
            end=tx.start + 2,
            score='.',
            strand=self.strand,
            frame='0',
            attributes=f"gene_id \"{self.gene_id}\"; "
                        f"transcript_id \"{self.id}\";",
        )
        tx = self.entries[key][-1]
        line2 = GTFEntry(
            name=self.chr,
            source=tx.source,
            feature='',
            start=tx.end - 2,
            end=tx.end,
            score='.',
            strand=self.strand,
            frame='0',
            attributes=f"gene_id \"{self.gene_id}\"; "
                        f"transcript_id \"{self.id}\";",
        )

        fragmented_transcript = True
        if tx.strand == '+':
            line1.feature = 'start_codon'
            line2.feature = 'stop_codon'
            if self.entries[key][0].frame == 0:
                fragmented_transcript = False
            start = line1
            stop = line2
        else:
            line1.feature = 'stop_codon'
            line2.feature = 'start_codon'
            if self.entries[key][-1].frame == 0:
                fragmented_transcript = False
            stop = line1
            start = line2
        if ('start_codon' not in self.entries.keys()
                and not fragmented_transcript):
            if not fragmented_transcript:
                self.add(start)
            else:
                self.entries.update({'start_codon' : []})
        if 'stop_codon' not in self.entries.keys():
            self.add(stop)

    def redo_phase(self) -> None:
        if 'CDS' in self.entries:
            self.entries['CDS'] = sorted(
                self.entries['CDS'],
                key=lambda x: x.start,
                reverse=(self.strand == '-'),
            )
            phase = 0
            for line in self.entries['CDS']:
                line.frame = str(phase)  # type: ignore
                phase = (3 - (line.end - line.start + 1 - phase) % 3) % 3

    def check_splits(self) -> None:
        for k in self.entries.keys():
            self.entries[k] = sorted(
                self.entries[k],
                key=lambda x: x.start,
            )
            new_list = [self.entries[k][0]]
            for i in range(1, len(self.entries[k])):
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
        for k in self.entries.keys():
            for i, entry in enumerate(self.entries[k]):

                if k == 'transcript':
                    tx_line = entry
                    tx_line.attributes = prefix + self.id
                    continue

                elif k == 'CDS':
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
                    entry.attributes = (
                        f"transcript_id \"{prefix + self.id}\"; "
                        f"gene_id \"{self.gene_id}\"; "
                        f"cds_type={cds_type};"
                    )

                elif k not in ['transcript', 'gene']:
                    entry.attributes = (
                        f"transcript_id \"{prefix + self.id}\"; "
                        f"gene_id \"{self.gene_id}\"; "
                    )

                gtf.append(entry)

        if 'exon' not in self.entries.keys():
            for entry in self.entries['CDS']:
                gtf.append(
                    GTFEntry(**(entry.model_dump() | {"feature": "exon"}))
                )

        gtf = sorted(gtf, key=lambda entry: (entry.start, entry.end))
        if tx_line is not None:
            gtf = [tx_line] + gtf

        return gtf
