import bricks2marble as b2m


def test_add() -> None:
    t1 = b2m.struct.Transcript(
        "t_1",
        "g_1",
        "sequence_1",
        "+",
    )
    entry = b2m.struct.GTFEntry(
        name="sequence_1",
        source="source",
        feature=b2m.struct.FeatureType.Intron,
        start=100,
        end=150,
        strand="+",
        frame=0,
        attributes="gene_id \"g_1\"; transcript_id \"t_1\";",
    )
    assert entry.attribute("gene_id") == "g_1"
    assert entry.attribute("transcript_id") == "t_1"
    t1.add(entry)
    t1.add(b2m.struct.GTFEntry(
        name="sequence_1",
        source="source",
        feature=b2m.struct.FeatureType.Exon,
        start=160,
        end=190,
        strand="+",
        frame=0,
        attributes="gene_id \"g_1\"; transcript_id \"t_1\";",
    ))
    t1.add(b2m.struct.GTFEntry(
        name="sequence_1",
        source="source",
        feature=b2m.struct.FeatureType.Exon,
        start=0,
        end=100,
        strand="+",
        frame=0,
        attributes="gene_id \"g_1\"; transcript_id \"t_1\";",
    ))
    t1.add(b2m.struct.GTFEntry(
        name="sequence_1",
        source="source",
        feature=b2m.struct.FeatureType.Exon,
        start=50,
        end=120,
        strand="+",
        frame=1,
        attributes="gene_id \"g_1\"; transcript_id \"t_1\";",
    ))

    assert t1.start == 0 and t1.end == 190

    assert t1.coords_per_frame(b2m.struct.FeatureType.Exon) == {
        "0": [(0, 100), (160, 190)],
        "1": [(50, 120)],
        "2": [],
        ".": [],
    }

    assert t1.coords(b2m.struct.FeatureType.Intron) == [(100, 150)]
    assert t1.get_cds_coords() == {
        "0": [(0, 100), (160, 190)],
        "1": [(50, 120)],
        "2": [],
    }

    t1.find_introns()
    t1.find_transcript()
    assert t1.coords(b2m.struct.FeatureType.Transcript) == [(0, 190)]
    t1.find_start_stop_codon()
