#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional


# ── Utilities ─────────────────────────────────────────────────────────────────

def _log(msg: str) -> None:
    print(f"  {msg}", flush=True)


def _check_java() -> bool:
    return shutil.which("java") is not None


def _find_audiveris_jar() -> Path | None:
    """Look for Audiveris JAR in env var, cwd, and common install locations."""
    # 1. Environment variable
    env = os.environ.get("AUDIVERIS_JAR")
    if env and Path(env).exists():
        return Path(env)

    # 2. Current directory and script directory
    for candidate_name in ("Audiveris.jar", "audiveris.jar", "audiveris-5.3.jar"):
        for search_dir in (Path.cwd(), Path(__file__).parent):
            p = search_dir / candidate_name
            if p.exists():
                return p

    # 3. ~/.audiveris/
    home_jar = Path.home() / ".audiveris" / "Audiveris.jar"
    if home_jar.exists():
        return home_jar

    return None


def _validate_musicxml(xml_path: Path) -> int:
    """
    Parse the MusicXML file with music21 and return the number of notes.
    Raises RuntimeError if parsing fails.
    """
    try:
        from music21 import converter, note, chord
        score = converter.parse(str(xml_path))
        n_notes = sum(
            1 for el in score.flatten().notesAndRests
            if isinstance(el, (note.Note, chord.Chord))
        )
        return n_notes
    except Exception as exc:
        raise RuntimeError(f"music21 failed to parse output: {exc}") from exc


# ── PDF → images ──────────────────────────────────────────────────────────────

def pdf_to_images(pdf_path: Path, out_dir: Path, dpi: int = 300,
                  pages: list[int] | None = None) -> list[Path]:
    """
    Convert PDF pages to PNG images using pdf2image (requires poppler).

    Parameters
    ----------
    pdf_path : Path to the PDF file.
    out_dir  : Directory where PNG images will be written.
    dpi      : Resolution (300 is good for OMR; higher = slower).
    pages    : 1-based list of pages to convert.  None → all pages.

    Returns
    -------
    Sorted list of Path objects for the generated PNG files.
    """
    try:
        from pdf2image import convert_from_path
    except ImportError:
        sys.exit(
            "\n  ERROR: pdf2image is not installed.\n"
            "  Install it with:  pip install pdf2image\n"
            "  Also install poppler:\n"
            "    macOS:  brew install poppler\n"
            "    Ubuntu: apt install poppler-utils\n"
        )

    _log(f"Converting PDF → images at {dpi} DPI …")
    kwargs: dict = {"dpi": dpi, "fmt": "png", "output_folder": str(out_dir),
                    "output_file": "page", "paths_only": True}
    if pages:
        kwargs["first_page"] = pages[0]
        kwargs["last_page"]  = pages[-1]

    try:
        image_paths = convert_from_path(str(pdf_path), **kwargs)
    except Exception as exc:
        sys.exit(f"\n  ERROR during PDF conversion: {exc}\n"
                 "  Is poppler installed and on PATH?\n")

    result = sorted(Path(p) for p in image_paths)
    _log(f"  Generated {len(result)} page image(s).")
    return result


#  OMR engine: oemer

def run_oemer(image_path: Path, out_dir: Path) -> Path:
    """
    Run oemer on a single image and return the path to the output MusicXML.

    oemer CLI: oemer <img_path> -o <output_dir>
    Output:    <output_dir>/<img_stem>.musicxml

    Install: pip install oemer opencv-python-headless
    """
    try:
        import importlib.util
        if importlib.util.find_spec("oemer") is None:
            raise ImportError("oemer not found")
    except ImportError:
        _log("oemer is not installed.  Run:  pip install oemer opencv-python-headless")
        raise RuntimeError("oemer unavailable")

    # Locate the oemer binary installed alongside this Python interpreter
    oemer_bin = Path(sys.executable).parent / "oemer"
    if not oemer_bin.exists():
        import shutil
        found = shutil.which("oemer")
        if not found:
            raise RuntimeError(
                "oemer binary not found. Re-install with:  pip install oemer")
        oemer_bin = Path(found)

    _log(f"Running oemer on {image_path.name} …")
    _log(f"  (first run downloads ~200 MB of model checkpoints — please wait)")

    # oemer <img_path> -o <output_dir>
    # Output file: <output_dir>/<img_stem>.musicxml
    cmd = [str(oemer_bin), str(image_path.resolve()), "-o", str(out_dir)]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        _log("oemer stderr (last 20 lines):")
        for line in (result.stderr + result.stdout).splitlines()[-20:]:
            _log(f"  {line}")
        raise RuntimeError(f"oemer exited with code {result.returncode}")

    # oemer outputs <stem>.musicxml in the output directory
    xml_candidates = (list(out_dir.rglob("*.musicxml"))
                      + list(out_dir.rglob("*.xml")))
    if not xml_candidates:
        raise RuntimeError("oemer ran but produced no MusicXML output")

    xml_candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return xml_candidates[0]


# OMR engine: Audiveris

def run_audiveris(input_path: Path, out_dir: Path) -> Path:
    """
    Run Audiveris in batch mode and return the path to the output MusicXML.

    Audiveris outputs a .mxl (compressed MusicXML) file in out_dir.

    Requirements
    ------------
      - Java 11+ on PATH
      - Audiveris JAR at AUDIVERIS_JAR env var, cwd, or ~/.audiveris/
        Download from: https://github.com/Audiveris/audiveris/releases
    """
    if not _check_java():
        raise RuntimeError("Java is not installed or not on PATH. "
                           "Install Java 11+:  brew install openjdk@11")

    jar = _find_audiveris_jar()
    if jar is None:
        raise RuntimeError(
            "Audiveris JAR not found.\n"
            "  Download from: https://github.com/Audiveris/audiveris/releases\n"
            "  Then set: export AUDIVERIS_JAR=/path/to/Audiveris.jar"
        )

    _log(f"Running Audiveris ({jar.name}) on {input_path.name} …")
    cmd = [
        "java", "-jar", str(jar),
        "-batch",
        "-export",
        "-output", str(out_dir),
        "--",
        str(input_path.resolve()),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        _log("Audiveris stderr (last 20 lines):")
        for line in result.stderr.splitlines()[-20:]:
            _log(f"  {line}")
        raise RuntimeError(f"Audiveris exited with code {result.returncode}")

    # Audiveris outputs .mxl files named after the input
    mxl_candidates = list(out_dir.rglob("*.mxl")) + list(out_dir.rglob("*.xml"))
    if not mxl_candidates:
        raise RuntimeError("Audiveris ran but produced no MXL output")

    mxl_candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return mxl_candidates[0]



def merge_musicxml_files(xml_paths: list[Path], merged_path: Path) -> Path:
    """
    Naively concatenate multiple single-page MusicXML files into one
    by appending measures from pages 2+ into the part of page 1.

    Uses music21 so the result is a well-formed MusicXML file.
    """
    try:
        from music21 import converter, stream
    except ImportError:
        raise RuntimeError("music21 is required for merging. pip install music21")

    _log(f"Merging {len(xml_paths)} MusicXML files …")

    scores = [converter.parse(str(p)) for p in xml_paths]
    if len(scores) == 1:
        return xml_paths[0]

    # Simple merge: append all parts from page 2+ to page 1's corresponding parts
    base = scores[0]
    for extra in scores[1:]:
        for part_idx, extra_part in enumerate(extra.parts):
            if part_idx < len(base.parts):
                base_part = base.parts[part_idx]
                for measure in extra_part.getElementsByClass("Measure"):
                    base_part.append(measure)
            else:
                base.append(extra_part)

    base.write("musicxml", fp=str(merged_path))
    _log(f"  Merged score saved to {merged_path}")
    return merged_path


# ── Main pipeline ─────────────────────────────────────────────────────────────

ENGINE_FUNCS = {
    "oemer":     run_oemer,
    "audiveris": run_audiveris,
}

ENGINE_ORDER = ["oemer", "audiveris"]


def run_pipeline(
    input_path: Path,
    output_path: Path,
    engine: str | None = None,
    dpi: int = 300,
    all_pages: bool = False,
    validate_only: bool = False,
) -> Path:
    """
    Full OMR pipeline: input (PDF or image) → validated MusicXML output.
    """
    print()
    print("  ╔══════════════════════════════════════════════════╗")
    print("  ║       Smarter Music — OMR Pipeline               ║")
    print("  ╚══════════════════════════════════════════════════╝")
    print()
    _log(f"Input   : {input_path}")
    _log(f"Output  : {output_path}")
    _log(f"Engine  : {engine or 'auto'}")
    print()

    if not input_path.exists():
        sys.exit(f"  ERROR: Input file not found: {input_path}")

    suffix = input_path.suffix.lower()
    is_pdf = suffix == ".pdf"
    is_image = suffix in (".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp")

    if not is_pdf and not is_image:
        sys.exit(f"  ERROR: Unsupported input format '{suffix}'. "
                 "Use PDF, PNG, JPG, TIFF, BMP, or WEBP.")

    with tempfile.TemporaryDirectory(prefix="omr_") as tmp:
        tmp_dir = Path(tmp)

        # ── Step 1: PDF → images (if needed) ──────────────────────────────
        if is_pdf:
            image_dir = tmp_dir / "pages"
            image_dir.mkdir()
            image_paths = pdf_to_images(input_path, image_dir, dpi=dpi)
            if not all_pages:
                _log("Using page 1 only (pass --all-pages to process all pages).")
                image_paths = image_paths[:1]
        else:
            image_paths = [input_path]

        # ── Step 2: Run OMR on each image ─────────────────────────────────
        xml_results: list[Path] = []
        engines_to_try = [engine] if engine else ENGINE_ORDER

        for img_path in image_paths:
            _log(f"Processing: {img_path.name}")
            xml_out: Path | None = None

            for eng_name in engines_to_try:
                fn = ENGINE_FUNCS.get(eng_name)
                if fn is None:
                    _log(f"  Unknown engine '{eng_name}', skipping.")
                    continue
                try:
                    eng_out_dir = tmp_dir / f"{eng_name}_{img_path.stem}"
                    eng_out_dir.mkdir(exist_ok=True)
                    xml_out = fn(img_path, eng_out_dir)
                    _log(f"  [{eng_name}] ✓  Output: {xml_out}")
                    break
                except RuntimeError as exc:
                    _log(f"  [{eng_name}] ✗  {exc}")

            if xml_out is None:
                sys.exit(
                    "\n  ERROR: All OMR engines failed.\n"
                    "  Check the log above for details, then install one of:\n"
                    "    pip install oemer          # deep-learning OMR (easiest)\n"
                    "    export AUDIVERIS_JAR=...   # Audiveris Java engine\n"
                )
            xml_results.append(xml_out)

        # ── Step 3: Merge multi-page results ──────────────────────────────
        if len(xml_results) > 1:
            merged = tmp_dir / "merged.xml"
            merge_musicxml_files(xml_results, merged)
            xml_final = merged
        else:
            xml_final = xml_results[0]

        # ── Step 4: Validate with music21 ─────────────────────────────────
        print()
        _log("Validating output with music21 …")
        try:
            n_notes = _validate_musicxml(xml_final)
            _log(f"  Parsed OK — {n_notes} note/chord events found.")
        except RuntimeError as exc:
            _log(f"  WARNING: Validation failed — {exc}")
            _log("  The file may still be usable; check it manually.")

        if validate_only:
            _log("--validate-only: stopping before copying output.")
            return xml_final

        # ── Step 5: Copy to output path ────────────────────────────────────
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(xml_final, output_path)

    print()
    _log(f"Done.  MusicXML saved to:  {output_path}")
    print()
    print("  Next steps")
    print("  ----------")
    print(f"  1. Open score.py and change the parse() line to:")
    print(f"       score = converter.parse(\"{output_path.name}\")")
    print( "  2. Check note count and measure numbers with:")
    print( "       python score.py")
    print( "  3. Adjust any missing/mis-numbered measures manually (see existing")
    print( "       _insert_after() calls in score.py for reference).")
    print( "  4. Run the score follower:")
    print( "       python note_detector.py")
    print()

    return output_path


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optical Music Recognition: PDF/image → MusicXML",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("input",
                        help="PDF or image file to process")
    parser.add_argument("-o", "--output", default=None,
                        help="Output MusicXML path (default: <input_stem>.mxl)")
    parser.add_argument("-e", "--engine", choices=list(ENGINE_FUNCS.keys()),
                        default=None,
                        help="OMR engine to use (default: auto)")
    parser.add_argument("--dpi", type=int, default=300,
                        help="DPI for PDF→image conversion (default: 300)")
    parser.add_argument("--all-pages", action="store_true",
                        help="Process all pages of a PDF (default: page 1 only)")
    parser.add_argument("--validate-only", action="store_true",
                        help="Run OMR and validate but don't write output file")

    args = parser.parse_args()

    input_path = Path(args.input)
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / (input_path.stem + ".mxl")

    run_pipeline(
        input_path=input_path,
        output_path=output_path,
        engine=args.engine,
        dpi=args.dpi,
        all_pages=args.all_pages,
        validate_only=args.validate_only,
    )


if __name__ == "__main__":
    main()
