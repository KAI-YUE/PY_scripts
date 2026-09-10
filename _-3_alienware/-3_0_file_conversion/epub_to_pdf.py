#!/usr/bin/env python3
"""Convert an EPUB to PDF locally using Calibre's ebook-convert command."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile


# %% CONFIG (edit these, then run the file)
# ----------------------------
root_dir = "/home/kyue/.var/app/org.telegram.desktop/data/TelegramDesktop/tdata/temp_data#2/{:s}"

INPUT_EPUB = r"5.epub"                  # Full path to your .epub file
OUTPUT_PDF = r"/mnt/ssd/YUE/Ebook/manga/11.pdf"                  # Blank = PDF beside the EPUB
CONVERTER  = "ebook-convert"      # Or full path to Calibre's executable
OVERWRITE  = False                # True = replace an existing PDF


INPUT_EPUB = os.path.join(root_dir.format(INPUT_EPUB))

# %% CONVERSION
# ----------------------------
def convert_epub(source, output, converter, overwrite=False):
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output}. Set OVERWRITE = True to replace it.")

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="epub_to_pdf_", dir=output.parent) as folder:
        temporary = Path(folder) / "converted.pdf"
        env = os.environ.copy()
        env.setdefault("QT_QPA_PLATFORM", "offscreen")
        subprocess.run([converter, str(source), str(temporary)], env=env, check=True)

        with temporary.open("rb") as result:
            if result.read(5) != b"%PDF-":
                raise ValueError("The converter did not produce a PDF.")

        if overwrite:
            os.replace(temporary, output)
        else:
            # Exclusive creation also protects a file created during conversion.
            with output.open("xb") as result:
                try:
                    with temporary.open("rb") as converted:
                        shutil.copyfileobj(converted, result)
                except BaseException:
                    result.close()
                    output.unlink()
                    raise


# %% RUN
# ----------------------------
def main():
    if not INPUT_EPUB.strip():
        raise ValueError("Set INPUT_EPUB in the CONFIG section before running.")

    source = Path(INPUT_EPUB).expanduser().resolve()
    if not source.is_file() or source.suffix.lower() != ".epub":
        raise ValueError(f"Expected an existing .epub file: {source}")

    output = Path(OUTPUT_PDF).expanduser().absolute() if OUTPUT_PDF.strip() else source.with_suffix(".pdf")
    if output.suffix.lower() != ".pdf":
        raise ValueError("OUTPUT_PDF must have a .pdf extension.")
    if output.resolve() == source:
        raise ValueError("Input and output must be different files.")

    converter = shutil.which(str(Path(CONVERTER).expanduser()))
    if not converter:
        raise FileNotFoundError("Calibre's ebook-convert was not found. Install Calibre from https://calibre-ebook.com/download or set CONVERTER to its full executable path.")

    convert_epub(source, output, converter, OVERWRITE)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
