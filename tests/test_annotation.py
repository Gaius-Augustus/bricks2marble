from pathlib import Path

import bricks2marble as b2m


def test_load(gtf_path: Path) -> None:
    annotation = b2m.load_gtf(gtf_path)

    assert annotation.genes["g1"].transcripts["g1.t1"].entries[
        b2m.struct.FeatureType.Transcript
    ] == [
        b2m.struct.GTFEntry(
            name="chr1",
            source="source",
            feature=b2m.struct.FeatureType.Transcript,
            start=1,
            end=250,
            strand="+",
            frame=None,
            attributes="gene_id \"g1\"; transcript_id \"g1.t1\";",
        ),
    ]

    assert set(annotation.genes.keys()) == {"g1"}
    assert set(annotation.genes["g1"].transcripts.keys()) == {"g1.t1", "g1.t2"}

    assert annotation.genes["g1"].transcripts["g1.t1"].entries[
        b2m.struct.FeatureType.Exon
    ] == [
        b2m.struct.GTFEntry(
            name="chr1",
            source="source",
            feature=b2m.struct.FeatureType.Exon,
            start=1,
            end=120,
            strand="+",
            frame=0,
            attributes="gene_id \"g1\"; transcript_id \"g1.t1\";",
        ),
        b2m.struct.GTFEntry(
            name="chr1",
            source="source",
            feature=b2m.struct.FeatureType.Exon,
            start=181,
            end=240,
            strand="+",
            frame=0,
            attributes="gene_id \"g1\"; transcript_id \"g1.t1\";",
        ),
    ]
    assert annotation.genes["g1"].transcripts["g1.t1"].entries[
        b2m.struct.FeatureType.Intron
    ] == [
        b2m.struct.GTFEntry(
            name="chr1",
            source="source",
            feature=b2m.struct.FeatureType.Intron,
            start=121,
            end=180,
            strand="+",
            frame=None,
            attributes="gene_id \"g1\"; transcript_id \"g1.t1\";",
        ),
        b2m.struct.GTFEntry(
            name="chr1",
            source="source",
            feature=b2m.struct.FeatureType.Intron,
            start=241,
            end=250,
            strand="+",
            frame=None,
            attributes="gene_id \"g1\"; transcript_id \"g1.t1\";",
        ),
    ]

    assert (b2m.struct.FeatureType.StartCodon
            not in annotation.genes["g1"].transcripts["g1.t1"].entries)

    annotation.finalize()

    assert annotation.genes["g1"].transcripts["g1.t1"].entries[
        b2m.struct.FeatureType.StartCodon
    ] == [
        b2m.struct.GTFEntry(
            name="chr1",
            source="source",
            feature=b2m.struct.FeatureType.StartCodon,
            start=1,
            end=3,
            strand="+",
            frame=0,
            attributes="gene_id \"g1\"; transcript_id \"g1.t1\";",
        ),
    ]
