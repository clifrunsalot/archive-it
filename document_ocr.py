"""Bounded, local PDF page extraction and image OCR in an isolated process."""

import json
import math
import os
import shutil
import signal
import subprocess
import sys
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

from app_logging import LOGGER

MAX_PIXELS = 16_000_000
MAX_SOURCE_PIXELS = 40_000_000
MAX_OCR_PAGES = 50
MAX_TEXT_CHARS = 2_000_000
MAX_RESULT_BYTES = 16_000_000
WORKER_TIMEOUT = 300
PAGE_TIMEOUT = 60
ERROR_CODES = {
    "ocr_engine_missing", "ocr_engine_failed", "ocr_timeout", "ocr_page_limit",
    "image_pixel_limit", "image_frames_not_supported", "render_failed",
    "extracted_text_limit", "worker_failed", "worker_output_invalid",
}


class LocalExtractionError(ValueError):
    """Fixed reason code; parser/engine output never escapes into logs."""


def engine_version() -> str:
    engine = shutil.which("tesseract")
    if engine is None:
        return "missing"
    try:
        result = subprocess.run([engine, "--version"], capture_output=True, text=True,
                                timeout=5, check=False)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise LocalExtractionError("ocr_engine_failed") from error
    first_line = result.stdout.splitlines()[0] if result.stdout else ""
    parts = first_line.split()
    if result.returncode or len(parts) != 2 or parts[0] != "tesseract":
        raise LocalExtractionError("ocr_engine_failed")
    return parts[1]


def extract_local(data: bytes, suffix: str, *, native_pdf: bool = False) -> dict:
    started = perf_counter()
    LOGGER.debug("Local extraction worker started format=%s bytes=%d native_pdf=%s "
                 "timeout_seconds=%d", suffix, len(data), native_pdf, WORKER_TIMEOUT)
    with TemporaryDirectory(prefix="archive-it-") as directory:
        folder = Path(directory)
        source, result = folder / ("input" + suffix), folder / "result.json"
        source.write_bytes(data)
        command = [sys.executable, str(Path(__file__).resolve()), str(source), str(result),
                   "native" if native_pdf else "ocr"]
        with (folder / "stderr").open("wb") as diagnostics:
            process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=diagnostics,
                                       start_new_session=True)
            try:
                process.wait(timeout=WORKER_TIMEOUT)
            except subprocess.TimeoutExpired as error:
                # The worker owns its Tesseract children; stop only this process group.
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
                process.wait()
                raise LocalExtractionError("ocr_timeout") from error
        if not result.exists() or result.stat().st_size > MAX_RESULT_BYTES:
            raise LocalExtractionError("worker_failed")
        try:
            payload = json.loads(result.read_text(encoding="utf-8"))
        except (ValueError, UnicodeError) as error:
            raise LocalExtractionError("worker_output_invalid") from error
        if not isinstance(payload, dict):
            raise LocalExtractionError("worker_output_invalid")
        if process.returncode:
            reason = payload.get("error")
            raise LocalExtractionError(
                reason if isinstance(reason, str) and reason in ERROR_CODES else "worker_failed")
        pages = payload.get("pages")
        if not isinstance(pages, list) or not pages or any(
            not isinstance(page, dict) or set(page) != {"text", "ocr"}
            or not isinstance(page["text"], str) or type(page["ocr"]) is not bool
            for page in pages
        ):
            raise LocalExtractionError("worker_output_invalid")
        if sum(len(page["text"]) for page in pages) > MAX_TEXT_CHARS:
            raise LocalExtractionError("extracted_text_limit")
        for number, page in enumerate(pages, 1):
            LOGGER.debug("Local extraction page=%d text_chars=%d ocr=%s",
                         number, len(page["text"]), page["ocr"])
        LOGGER.debug("Local extraction worker finished pages=%d elapsed_ms=%.2f",
                     len(pages), (perf_counter() - started) * 1000)
        return payload


def ocr_image(image, folder: Path, number: int) -> str:
    # Uniform backgrounds contain no visible text, regardless of paper color.
    with image.convert("L") as gray:
        low, high = gray.getextrema()
        if low == high:
            return ""
    engine = shutil.which("tesseract")
    if engine is None:
        raise LocalExtractionError("ocr_engine_missing")
    source, output = folder / f"page-{number}.png", folder / f"text-{number}"
    image.save(source)
    try:
        with (folder / "engine-stderr").open("wb") as diagnostics:
            try:
                completed = subprocess.run(
                    [engine, str(source), str(output), "-l", "eng", "--psm", "3"],
                    stdout=subprocess.DEVNULL, stderr=diagnostics, timeout=PAGE_TIMEOUT,
                    env=dict(os.environ, OMP_THREAD_LIMIT="1"), check=False,
                )
            except subprocess.TimeoutExpired as error:
                raise LocalExtractionError("ocr_timeout") from error
        text_file = output.with_suffix(".txt")
        if completed.returncode or not text_file.exists():
            raise LocalExtractionError("ocr_engine_failed")
        if text_file.stat().st_size > MAX_TEXT_CHARS * 4:
            raise LocalExtractionError("extracted_text_limit")
        return text_file.read_text(encoding="utf-8")
    finally:
        source.unlink(missing_ok=True)
        output.with_suffix(".txt").unlink(missing_ok=True)


def worker(source: Path, native_pdf: bool) -> dict:
    from PIL import Image, ImageOps
    import pypdfium2 as pdfium

    pages = []
    total = 0

    def append(text: str, ocr: bool) -> None:
        nonlocal total
        total += len(text)
        if total > MAX_TEXT_CHARS:
            raise LocalExtractionError("extracted_text_limit")
        pages.append({"text": text, "ocr": ocr})

    if source.suffix == ".pdf":
        with pdfium.PdfDocument(source) as document:
            ocr_pages = 0
            for number in range(len(document)):
                with closing(document[number]) as page:
                    text = ""
                    if native_pdf:
                        with closing(page.get_textpage()) as textpage:
                            text = textpage.get_text_bounded(errors="strict")
                    used_ocr = not text.strip()
                    if used_ocr:
                        ocr_pages += 1
                        if ocr_pages > MAX_OCR_PAGES:
                            raise LocalExtractionError("ocr_page_limit")
                        width, height = page.get_size()
                        if not all(math.isfinite(x) and x > 0 for x in (width, height)):
                            raise LocalExtractionError("render_failed")
                        scale = min(200 / 72, math.sqrt(MAX_PIXELS / (width * height)),
                                    6000 / max(width, height))
                        if math.ceil(width * scale) * math.ceil(height * scale) > MAX_PIXELS:
                            scale *= 0.99
                        # PDFium documents float scale; its untyped default is inferred as int.
                        with closing(page.render(scale=scale)) as bitmap:  # pyright: ignore[reportArgumentType]
                            with bitmap.to_pil() as image:
                                text = ocr_image(image, source.parent, number + 1)
                    append(text, used_ocr and bool(text.strip()))
    else:
        try:
            original = Image.open(source)
        except Image.DecompressionBombError as error:
            raise LocalExtractionError("image_pixel_limit") from error
        with original:
            if original.width * original.height > MAX_SOURCE_PIXELS:
                raise LocalExtractionError("image_pixel_limit")
            if getattr(original, "n_frames", 1) != 1:
                raise LocalExtractionError("image_frames_not_supported")
            with ImageOps.exif_transpose(original) as upright, upright.convert("RGB") as image:
                scale = min(1.0, math.sqrt(MAX_PIXELS / (image.width * image.height)),
                            6000 / max(image.size))
                image.thumbnail((max(1, int(image.width * scale)),
                                 max(1, int(image.height * scale))))
                append(ocr_image(image, source.parent, 1), True)
    return {"pages": pages}


def main() -> int:
    source, destination = Path(sys.argv[1]), Path(sys.argv[2])
    try:
        result = worker(source, sys.argv[3] == "native")
    except LocalExtractionError as error:
        destination.write_text(json.dumps({"error": str(error)}), encoding="utf-8")
        return 1
    except (OSError, ValueError, RuntimeError, OverflowError):
        # Process boundary: report a fixed reason, never native parser messages.
        destination.write_text(json.dumps({"error": "render_failed"}), encoding="utf-8")
        return 1
    destination.write_text(json.dumps(result), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
