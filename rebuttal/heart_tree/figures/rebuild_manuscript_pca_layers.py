"""Rebuild the current manuscript PCA figures from fixed point-cloud/vector layers.

The author approved these current PDF layers as the source of truth. This route
preserves their point clouds and final layout and never loads experimental logs,
checkpoints, summaries, or cell arrays. It is distinct from the raw-count plotting
entry points, whose layout predates the current manuscript.

Requires pypdf. Example:
    python rebuild_manuscript_pca_layers.py --figure 6 --output figure6.pdf
"""
from pathlib import Path
import argparse
import hashlib
import json
import logging

from pypdf import PdfReader, PdfWriter
from pypdf.generic import ContentStream, ByteStringObject

SOURCE = Path(__file__).resolve().parent / "manuscript_layers"
EXPECTED = {
    6: ("scVI", "CFGen", "scANVI", "MDLM", "CRNDiff", "Real target cells", "Generated cells"),
    7: ("(a) Test cells", "Endothelial", "Myeloid", "Neuronal", "Mesothelial"),
    8: ("Marginal tilt", "FK steering", "Tilted FK steering", "All test cells (n=45,145)", "Real target cells", "Generated cells"),
}
LABEL_EDITS = {
    7: ("(a) Real cells", "(a) Test cells"),
    8: ("All real cells (n=45,145)", "All test cells (n=45,145)"),
}
FORBIDDEN = ("Ours", "Delivered cells", "Real cells of the requested type", "Marginal tilt only", "Tilted FK (ours)")


def objects(page):
    yield page
    for ref in page.get("/Resources", {}).get("/XObject", {}).values():
        obj = ref.get_object()
        if obj.get("/Subtype") == "/Form":
            yield from objects(obj)


def image_hashes(reader):
    values = []
    for obj in objects(reader.pages[0]):
        for ref in obj.get("/Resources", {}).get("/XObject", {}).values():
            image = ref.get_object()
            if image.get("/Subtype") == "/Image":
                values.append(hashlib.sha256(image.get_data()).hexdigest())
    return sorted(values)


def label_bytes(text):
    # The existing embedded DejaVu fonts map printable ASCII U+0020..U+007E
    # to CIDs 0x0003..0x0061. Reuse these glyphs without altering the font,
    # text matrix, size, colour, or any point-cloud/vector geometry.
    assert all(32 <= ord(char) <= 126 for char in text)
    return b"".join((ord(char) - 29).to_bytes(2, "big") for char in text)


def replace_label(operations, obj, old, new):
    replaced = 0
    font_key = None
    for args, op in operations:
        if op == b"Tf":
            font_key = args[0]
        if op != b"TJ":
            continue
        for index, item in enumerate(args[0]):
            if not isinstance(item, (str, bytes)):
                continue
            raw = item.original_bytes if hasattr(item, "original_bytes") else bytes(item)
            if raw != label_bytes(old):
                continue
            font = obj["/Resources"]["/Font"][font_key].get_object()
            cmap = font["/ToUnicode"].get_object().get_data()
            assert b"<0003> <0061> <0020>" in cmap
            args[0][index] = ByteStringObject(label_bytes(new))
            replaced += 1
    return replaced


def rebuild(number, destination):
    logging.getLogger("pypdf").setLevel(logging.ERROR)
    manifest = json.loads((SOURCE / "manifest.json").read_text(encoding="utf-8"))[str(number)]
    source = SOURCE / manifest["file"]
    assert hashlib.sha256(source.read_bytes()).hexdigest() == manifest["sha256"]
    destination = Path(destination)
    if source.resolve() == destination.resolve():
        raise ValueError("The fixed source layer must not be overwritten.")
    reader = PdfReader(source)
    text = reader.pages[0].extract_text()
    final_text = text
    if number in LABEL_EDITS:
        old, new = LABEL_EDITS[number]
        assert text.count(old) == 1
        final_text = text.replace(old, new)
    assert all(label in final_text for label in EXPECTED[number])
    assert not any(label in text for label in FORBIDDEN)
    writer = PdfWriter()
    writer.clone_document_from_reader(reader)
    # Reconstruct each vector/text command stream from parsed PDF operators;
    # retain the source coordinate transforms, font objects and raster layers.
    streams = 0
    replacements = 0
    for obj in objects(writer.pages[0]):
        contents = obj.get_contents() if hasattr(obj, "get_contents") else obj
        stream = ContentStream(contents, writer)
        parsed = list(stream.operations)
        if number in LABEL_EDITS:
            replacements += replace_label(parsed, obj, *LABEL_EDITS[number])
        stream.operations = parsed
        if hasattr(obj, "replace_contents"):
            obj.replace_contents(stream)
        else:
            obj.set_data(stream.get_data())
        streams += 1
    assert replacements == int(number in LABEL_EDITS)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as file:
        writer.write(file)
    rebuilt = PdfReader(destination)
    assert list(reader.pages[0].mediabox) == list(rebuilt.pages[0].mediabox)
    assert image_hashes(reader) == image_hashes(rebuilt)
    assert final_text == rebuilt.pages[0].extract_text()
    print(f"Figure {number}: rebuilt {streams} streams; {replacements} label updated; point-cloud layers preserved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure", type=int, choices=(6, 7, 8), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rebuild(args.figure, args.output)
