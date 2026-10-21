import io
import json
import shutil
import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest
from PIL import Image, ImageDraw, ImageFont

import document_ocr as ocr


def image_bytes():
    with Image.new("RGB", (1200, 300), "white") as image:
        draw = ImageDraw.Draw(image)
        draw.text((50, 50), "Receipt R-777 Total $123.45",
                  font=ImageFont.load_default(size=48), fill="black")
        output = io.BytesIO()
        image.save(output, format="PNG")
        return output.getvalue()


def test_blank_image_does_not_need_engine(tmp_path, monkeypatch):
    monkeypatch.setattr(ocr.shutil, "which", lambda _: None)
    with Image.new("RGB", (100, 100), "white") as image:
        assert ocr.ocr_image(image, tmp_path, 1) == ""


def test_missing_engine_is_explicit(tmp_path, monkeypatch):
    monkeypatch.setattr(ocr.shutil, "which", lambda _: None)
    with Image.open(io.BytesIO(image_bytes())) as image:
        with pytest.raises(ocr.LocalExtractionError, match="ocr_engine_missing"):
            ocr.ocr_image(image, tmp_path, 1)


@pytest.mark.parametrize("failure,reason", [
    (Mock(returncode=1), "ocr_engine_failed"),
    (subprocess.TimeoutExpired("tesseract", 60), "ocr_timeout"),
])
def test_engine_errors_are_redacted_and_images_cleaned(tmp_path, monkeypatch, failure, reason):
    monkeypatch.setattr(ocr.shutil, "which", lambda _: "/synthetic/tesseract")
    run = Mock(side_effect=failure) if isinstance(failure, Exception) else Mock(return_value=failure)
    monkeypatch.setattr(ocr.subprocess, "run", run)
    with Image.open(io.BytesIO(image_bytes())) as image:
        with pytest.raises(ocr.LocalExtractionError, match=reason):
            ocr.ocr_image(image, tmp_path, 1)
    assert not (tmp_path / "page-1.png").exists()
    assert not (tmp_path / "text-1.txt").exists()
    assert run.call_args.kwargs["timeout"] == ocr.PAGE_TIMEOUT


def test_image_pixel_and_frame_limits(tmp_path, monkeypatch):
    source = tmp_path / "input.png"
    source.write_bytes(image_bytes())
    monkeypatch.setattr(ocr, "MAX_SOURCE_PIXELS", 100)
    with pytest.raises(ocr.LocalExtractionError, match="image_pixel_limit"):
        ocr.worker(source, False)


def test_multiframe_image_rejected(tmp_path):
    source = tmp_path / "input.tiff"
    with Image.new("RGB", (10, 10), "white") as first, \
            Image.new("RGB", (10, 10), "black") as second:
        first.save(source, save_all=True, append_images=[second])
    with pytest.raises(ocr.LocalExtractionError, match="image_frames_not_supported"):
        ocr.worker(source, False)


def test_worker_missing_engine_and_temporary_cleanup(monkeypatch, tmp_path):
    temporary = ocr.TemporaryDirectory
    folders = []
    def create(**kwargs):
        directory = temporary(dir=tmp_path, **kwargs)
        folders.append(Path(directory.name))
        return directory
    monkeypatch.setattr(ocr, "TemporaryDirectory", create)
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(ocr.LocalExtractionError, match="ocr_engine_missing"):
        ocr.extract_local(image_bytes(), ".png")
    assert folders and all(not folder.exists() for folder in folders)


def test_engine_version_tracks_installation(monkeypatch):
    monkeypatch.setattr(ocr.shutil, "which", lambda _: None)
    assert ocr.engine_version() == "missing"
    monkeypatch.setattr(ocr.shutil, "which", lambda _: "/synthetic/tesseract")
    monkeypatch.setattr(ocr.subprocess, "run",
                        Mock(return_value=Mock(returncode=0, stdout="tesseract 5.5.3\n")))
    assert ocr.engine_version() == "5.5.3"


def test_corrupt_worker_input_has_fixed_error():
    with pytest.raises(ocr.LocalExtractionError, match="render_failed"):
        ocr.extract_local(b"SECRET invalid image", ".png")


@pytest.mark.skipif(shutil.which("tesseract") is None, reason="Local Tesseract not installed")
def test_worker_debug_logs_counts_not_document_text(caplog):
    import logging

    with caplog.at_level(logging.DEBUG, logger="archive_it"):
        result = ocr.extract_local(image_bytes(), ".png")
    assert result["pages"][0]["text"].strip()
    assert "Local extraction worker started" in caplog.text
    assert "Local extraction page=1 text_chars=" in caplog.text
    assert "Local extraction worker finished pages=1" in caplog.text
    assert "R-777" not in caplog.text and "$123.45" not in caplog.text


def test_ocr_page_limit(tmp_path, monkeypatch):
    from test_document_library import make_pdf

    source = tmp_path / "input.pdf"
    source.write_bytes(make_pdf("", ""))
    monkeypatch.setattr(ocr, "MAX_OCR_PAGES", 1)
    with pytest.raises(ocr.LocalExtractionError, match="ocr_page_limit"):
        ocr.worker(source, False)


def test_render_pixel_bound(tmp_path, monkeypatch):
    from test_document_library import make_pdf

    source = tmp_path / "input.pdf"
    source.write_bytes(make_pdf(""))
    monkeypatch.setattr(ocr, "MAX_PIXELS", 10000)
    sizes = []
    def recognize(image, *_):
        sizes.append(image.size)
        return "Synthetic text"
    monkeypatch.setattr(ocr, "ocr_image", recognize)
    result = ocr.worker(source, False)
    assert all(width * height <= 10000 for width, height in sizes)
    assert result["pages"][0]["ocr"]


def test_worker_timeout_stops_own_process_group(tmp_path, monkeypatch):
    process = Mock(pid=12345)
    process.wait.side_effect = [subprocess.TimeoutExpired("worker", 300), 0]
    monkeypatch.setattr(ocr.subprocess, "Popen", Mock(return_value=process))
    kill = Mock()
    monkeypatch.setattr(ocr.os, "killpg", kill)
    with pytest.raises(ocr.LocalExtractionError, match="ocr_timeout"):
        ocr.extract_local(b"synthetic", ".png")
    kill.assert_called_once_with(12345, ocr.signal.SIGKILL)
    assert process.wait.call_count == 2


@pytest.mark.parametrize("payload,returncode,reason", [
    ({"error": "SECRET parser text"}, 1, "worker_failed"),
    ({"error": ["invalid"]}, 1, "worker_failed"),
    ({"pages": [{"text": 7, "ocr": True}]}, 0, "worker_output_invalid"),
    ({"pages": []}, 0, "worker_output_invalid"),
])
def test_worker_result_validation(monkeypatch, payload, returncode, reason):
    def launch(command, **_kwargs):
        from pathlib import Path
        Path(command[3]).write_text(json.dumps(payload))
        return Mock(returncode=returncode)
    monkeypatch.setattr(ocr.subprocess, "Popen", launch)
    with pytest.raises(ocr.LocalExtractionError, match=reason):
        ocr.extract_local(b"synthetic", ".png")


def test_text_limit(tmp_path, monkeypatch):
    source = tmp_path / "input.png"
    source.write_bytes(image_bytes())
    monkeypatch.setattr(ocr, "MAX_TEXT_CHARS", 3)
    monkeypatch.setattr(ocr, "ocr_image", lambda *_: "1234")
    with pytest.raises(ocr.LocalExtractionError, match="extracted_text_limit"):
        ocr.worker(source, False)


@pytest.mark.skipif(shutil.which("tesseract") is None, reason="Local Tesseract not installed")
def test_local_ocr_image_and_scanned_pdf():
    image = image_bytes()
    result = ocr.extract_local(image, ".png")
    assert "R-777" in result["pages"][0]["text"]
    assert "$123.45" in result["pages"][0]["text"]
    assert result["pages"][0]["ocr"]
    with Image.open(io.BytesIO(image)) as source:
        output = io.BytesIO()
        source.save(output, format="PDF", resolution=150)
    result = ocr.extract_local(output.getvalue(), ".pdf", native_pdf=True)
    assert "R-777" in result["pages"][0]["text"]
    assert "$123.45" in result["pages"][0]["text"]
    assert result["pages"][0]["ocr"]
