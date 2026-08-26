# Author-layout Figure 3 source

`gwb-ceilings-final-prepatch.pdf` is the immutable author-finished PDF before
the AGN--IMRI benchmark rate was updated from 30 to
30.4 Gpc^-3 yr^-1.  It is retained only because the editable Keynote source
is not available in this repository.

`tools/patch_author_figure3.py` accepts this file only at its pinned SHA-256
hash and writes a separate corrected PDF.  It does not overwrite this source.
The fully code-generated scientific figure is produced independently by
`make_figure3_multiband.py` and is the preferred public reproduction.
