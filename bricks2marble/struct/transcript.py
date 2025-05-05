import sys
from typing import Literal

from pydantic import BaseModel


class NotGTFFormat(Exception):

    ...


class GTFEntry(BaseModel):

    seqname: str
    source: str
    feature: str
    start: int
    end: int
    score: int | Literal["."]
    strand: Literal["+", "-"]
    frame: Literal["0", "1", "2", "."]
    attributes: str

    @staticmethod
    def from_list(line: list[str]) -> "GTFEntry":
        return GTFEntry(
            seqname=line[0],
            source=line[1],
            feature=line[2],
            start=int(line[3]),
            end=int(line[4]),
            score=int(line[5]) if line[5] != "." else ".",
            strand=line[6],  # type: ignore
            frame=line[7],  # type: ignore
            attributes=line[8],
        )


class Transcript:
    """Class handling the data structures and methods for a transcript.

    Args:
        id (str): Transcript ID
        gene_id (str): Gene ID
        chr (str): Chromosome or sequence name where the transcript is
            located.
        annotation (str): ID of the source annotation.
        strand (str): Strand (+/-) on which the transctipt is located.
            Defaults to +.
    """

    def __init__(
        self,
        id: str,
        gene_id: str,
        chr: str,
        annotation: str,
        strand: Literal["+", "-"] = "+",
    ) -> None:
        self.id = id
        self.chr = chr
        self.gene_id = gene_id
        self.transcript_lines: dict[str, list[GTFEntry]] = {}
        self.gtf = []
        self.source_anno = annotation
        self.start = -1
        self.end = -1
        self.cds_len = -1
        self.cds_coords = {}
        self.strand: Literal["+", "-"] = strand
        self.source_method = ''

    def add_line(self, line: GTFEntry | list[str]) -> None:
        """Add a single line from the gtf file to the transcript data
        structure.

        Args:
            line (list): List of all elements of a line from a gtf file.
        """
        entry = GTFEntry.from_list(line) if isinstance(line, list) else line
        if not (entry.seqname == self.chr or entry.strand == self.strand):
            raise NotGTFFormat(
                "File is not in gtf format. "
                f"Error in line {entry}\n"
                "Transcript ID is not unique"
            )

        if entry.feature not in self.transcript_lines.keys():
            self.transcript_lines.update({entry.feature : []})

        self.source_method = entry.source

        if self.start < 0 or entry.start < self.start:
            self.start = entry.start
        if self.end < 0 or entry.end > self.end:
            self.end = entry.end

        if self.gene_id == "" and entry.feature != 'transcript':
            self.gene_id = entry.attributes.split(
                'gene_id "'
            )[1].split('";')[0]

        self.transcript_lines[entry.feature].append(entry)

    def coords_per_frame(
        self,
        type: str,
    ) -> dict[str, list[tuple[int, int]]]:
        """Get the coordinates and reading frame of the coding regions.

        Returns:
            dict[list[list[int]]]: Dictionary with list of type coords
                for each frame phase (0, 1, 2).
        """

        coords = {'0' : [], '1' : [], '2' : [], '.' : []}

        if type == 'CDS' and type not in self.transcript_lines.keys():
            type = 'exon'
        if type not in self.transcript_lines.keys():
            return coords

        for entry in self.transcript_lines[type]:
            coords[entry.frame].append([entry.start, entry.end])

        for k in coords.keys():
            coords[k].sort(key=lambda c: (c[0], c[1]))
        if type == 'CDS':
            coords['0'] += coords['.']
            del coords['.']

        return coords

    def coords(self, type: str) -> list[tuple[int, int]]:
        coords = []

        if type == 'CDS' and type not in self.transcript_lines.keys():
            type = 'exon'
        if type not in self.transcript_lines.keys():
            return coords

        for entry in self.transcript_lines[type]:
            coords.append([entry.start, entry.end])

        coords.sort(key=lambda c: (c[0], c[1]))
        return coords

    def get_cds_len(self):
        cds = self.coords('CDS')
        return sum([c[1] - c[0] + 1 for c in cds])

    def get_cds_coords(self) -> dict[str, list[list[int]]]:
        """Get the coordinates and reading frame of the coding regions

        Returns:
            dict[str, list[list[int]]]: Dictionary with list of CDS
                coords for each each frame phase (0, 1, 2).
        """
        if not self.cds_coords.keys():
            self.cds_coords = {'0' : [], '1' : [], '2' : []}

            if 'CDS' in self.transcript_lines.keys():
                key = 'CDS'
            else:
                key = 'exon'

            for entry in self.transcript_lines[key]:
                self.cds_coords[entry.frame].append([entry.start, entry.end])
            for k in self.cds_coords.keys():
                self.cds_coords[k].sort(key=lambda c: (c[0], c[1]))

        return self.cds_coords

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
        if ('CDS' not in self.transcript_lines.keys()
                and 'exon' not in self.transcript_lines.keys()):
            sys.stderr.write(
                f'Skipping transcript {self.id}, no CDS nor exons'
            )
            return False
        return True

    def find_introns(self) -> None:
        """Add intron lines."""
        if not 'intron' in self.transcript_lines.keys():
            self.transcript_lines.update({'intron' : []})
            key = ''

            if 'CDS' in self.transcript_lines.keys():
                key = 'CDS'
            elif 'exon' in self.transcript_lines.keys():
                key = 'exon'

            if key:
                exon_lst: list[GTFEntry] = []
                for line in self.transcript_lines[key]:
                    exon_lst.append(line)
                exon_lst = sorted(exon_lst, key=lambda e: e.seqname)
                for i in range(1, len(exon_lst)):
                    intron = GTFEntry(
                        seqname=exon_lst[i].seqname,
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
                    self.transcript_lines['intron'].append(intron)

    def find_transcript(self) -> None:
        """Add transcript lines."""
        if not 'transcript' in self.transcript_lines.keys():
            for key in self.transcript_lines.keys():
                for entry in self.transcript_lines[key]:
                    if entry.start < self.start or self.start < 0:
                        self.start = entry.start
                    if entry.end > self.end:
                        self.end = entry.end
            tx_line = [
                self.chr,
                entry.source,
                'transcript',
                self.start,
                self.end,
                '.',
                entry.strand,
                '.',
                self.id,
            ]
            self.add_line(tx_line)

    def find_start_stop_codon(self) -> None:
        """Add start/stop codon lines."""

        if not 'start_codon' in self.transcript_lines.keys():
            self.transcript_lines.update({'start_codon' : []})
        if not 'stop_codon' in self.transcript_lines.keys():
            self.transcript_lines.update({'stop_codon' : []})

        key = ''
        if 'CDS' in self.transcript_lines.keys():
            key = 'CDS'
        elif 'exon' in self.transcript_lines.keys():
            key = 'exon'

        if key:
            self.transcript_lines[key].sort(key = lambda x: x.start)
            tx = self.transcript_lines[key][0]
            line1 = GTFEntry(
                seqname=self.chr,
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
            tx = self.transcript_lines[key][-1]
            line2 = GTFEntry(
                seqname=self.chr,
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
                if self.transcript_lines[key][0].frame == 0:
                    fragmented_transcript = False
                start = line1
                stop = line2
            else:
                line1.feature = 'stop_codon'
                line2.feature = 'start_codon'
                if self.transcript_lines[key][-1].frame == 0:
                    fragmented_transcript = False
                stop = line1
                start = line2
            if ('start_codon' not in self.transcript_lines.keys()
                    and not fragmented_transcript):
                if not fragmented_transcript:
                    self.add_line(start)
                else:
                    self.transcript_lines.update({'start_codon' : []})
            if 'stop_codon' not in self.transcript_lines.keys():
                self.add_line(stop)

    def redo_phase(self):
        if 'CDS' in self.transcript_lines:
            self.transcript_lines['CDS'] = sorted(
                self.transcript_lines['CDS'],
                key=lambda x: x.start,
                reverse=(self.strand == '-'),
            )
            phase = 0
            for line in self.transcript_lines['CDS']:
                line.frame = str(phase)  # type: ignore
                phase = (3 - (line.end - line.start + 1 - phase) % 3) % 3

    def check_splits(self) -> None:
        for k in self.transcript_lines.keys():
            self.transcript_lines[k] = sorted(
                self.transcript_lines[k],
                key=lambda x: x.start,
            )
            new_list = [self.transcript_lines[k][0]]
            for i in range(1, len(self.transcript_lines[k])):
                if new_list[-1].end == self.transcript_lines[k][i].start-1:
                    new_list[-1].end = self.transcript_lines[k][i].end
                else:
                    new_list.append(self.transcript_lines[k][i])
            self.transcript_lines[k] = new_list

    def get_gtf(self, prefix: str = "") -> list[GTFEntry]:
        """Creates gtf output for the transcript.

        Returns:
            list[GTFEntry]: List of :class:`GTFEntry` objects.
        """
        gtf: list[GTFEntry] = []
        if prefix:
            prefix += '.'
        tx_line: GTFEntry | None = None
        for k in self.transcript_lines.keys():
            for i, entry in enumerate(self.transcript_lines[k]):

                if k == 'transcript':
                    tx_line = entry
                    tx_line.attributes = prefix + self.id
                    continue

                elif k == 'CDS':
                    cds_type = 'internal'
                    if len(self.transcript_lines[k]) == 1:
                        cds_type = 'single'
                    elif ((i == 0 and self.strand == '+')
                            or (i == len(self.transcript_lines[k]) - 1
                                    and self.strand == '-')
                    ):
                        cds_type = 'initial'
                    elif ((i == len(self.transcript_lines[k]) - 1
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

        if 'exon' not in self.transcript_lines.keys():
            for entry in self.transcript_lines['CDS']:
                gtf.append(
                    GTFEntry(**(entry.model_dump() and {"feature": "exon"}))
                )

        gtf = sorted(gtf, key=lambda entry: (entry.start, entry.end))
        if tx_line is not None:
            gtf = [tx_line] + gtf

        return gtf
