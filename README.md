# bricks2marble

- Python structures for nucleotide sequences and genome annotations.
- Tensorflow implementation of an HMM used for finding genes.
- Pre- and postprocessing tools for deep learning genome annotation models.
- Python interfaces for common bioinformatics tools and file format converters.

## Installation

Download `bricks2marble` via pip.

    $ python -m pip install bricks2marble

For development purposes, clone the repository and install it locally inside your virtual
environment.

    $ git clone https://github.com/gaius-augustus/bricks2marble
    $ cd bricks2marble
    $ python -m pip install -e .

If access to the tensorflow part of bricks2marble is needed, specify this as an optional dependency
when installing. This will install [hidten](https://github.com/Gaius-Augustus/hidten).

    $ python -m pip install bricks2marble[tf]
    # or
    $ python -m pip install -e .[tf]

When plotting is required, `pip install bricks2marble[plot]`.

## Overview

Below are some use cases of `bricks2marble`. All methods have docstrings that explain their
behaviour and several optional arguments in detail.

### Reading and writing

Loading large fasta files is implemented efficiently using bytearray translation tables (~11 seconds
for the human genome).
Additionally, `mmap` is used for indexing large fasta files (~4 seconds for the human genome).

```python
import bricks2marble as b2m

fasta = b2m.io.load_fasta("genome.fa.gz") # load everything into memory
sequence = fasta["chr1"].positions(0, 100)

fasta = b2m.io.indexed_fasta("genome.fa") # build a sequence index
sequence = fasta.fetch("chr1", (0, 100)) # load only required parts
```

For some specific cases, external tools are used. For example, indexing `.fa.gz` files requires
[pyfaidx](https://pypi.org/project/pyfaidx/).

    $ python -m pip install pyfaidx

```python
fasta = b2m.io.indexed_fasta("genome.fa.gz") # build a sequence index for compressed files
sequence = fasta.fetch("chr1", (0, 100)) # load only required parts
```

Additionally, `.gp` (`.genepred`) files can be loaded and are internally sorted for optimized
access.

```python
anno = b2m.io.load_annotation("reference.gp")
anno.classify(1062, "chr1") # labels per strand: ("intergenic", "CDS")
```

### Tools

The subpackage `bricks2marble.tools` contains a number of interfaces to common external tools
related to genome file formats. Download the external tools yourself and tell `bricks2marble` where
they can be found locally. Optionally, you can add them to your system path, so `bricks2marble` can
find them automatically.

#### Example: Comparing genome annotations

Download [gffcompare](https://ccb.jhu.edu/software/stringtie/gffcompare.shtml) and use the
`bricks2marble` interface for extracting metrics.

```python
import bricks2marble as b2m

b2m.tools.configure(gffcompare="path/to/gffcompare")
comparison = b2m.tools.compare(
    ["my_annotation.gp", "other_annotation.gtf"],
    "reference.gff",
    e=3,
)
print(comparison[0].locus.sensitivity)
fig = b2m.tools.plot_comparison(
    comparison,
    labels=["My", "Other"],
    table=True,
)
fig.show()
```

#### Example: Converting files

Convert various file formats for genome annotations. The internal `bricks2marble` representation of
annotation files is closely related to the `genepred` format. Conversions to `gtf` and `gff3` are
implemented directly. Conversions from these formats to `genepred` are handled by the corresponding
[external tools from UCSC](https://hgdownload.soe.ucsc.edu/admin/exe/linux.x86_64/), like `gtfToGenePred`.

```python
import bricks2marble as b2m

b2m.tools.configure(gtfToGenePred="path/to/gtfToGenePred")

with b2m.tools.Converter("my_annotation.gtf", "gp") as tmp_file_path:
    # gp file created using Python's tempfile
    annotation = b2m.io.load_annotation(tmp_file_path)
# gp file deleted, annotation loaded into memory

b2m.tools.convert(annotation, "my_annotation.gff", source="MyTool")
```

## License

This project is licensed under the [MIT license](/LICENSE).

## Projects using bricks2marble

- [Tiberius: End-to-End Deep Learning with an HMM for Gene
   Prediction](https://github.com/Gaius-Augustus/Tiberius)
