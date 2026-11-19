# Read-only probe of a PDF: page size, extracted text, embedded image sizes; renders page 1 to PNG if PyMuPDF is present. 1 thread.
import sys
from pypdf import PdfReader
p, out_png = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else None)
r = PdfReader(p); pg = r.pages[0]
print("mediabox", [float(x) for x in pg.mediabox]); print("TEXT:", repr(pg.extract_text()))
try:
    import fitz
    d = fitz.open(p); print("fitz", fitz.__doc__.split()[1] if fitz.__doc__ else "?")
    for img in d[0].get_images(full=True):
        print("image xref", img[0], "size", img[2], "x", img[3])
    if out_png:
        d[0].get_pixmap(dpi=200).save(out_png); print("png", out_png)
except ImportError:
    print("no fitz")
