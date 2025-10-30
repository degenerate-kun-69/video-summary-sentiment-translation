"""Application entry point and Flask app factory."""

from __future__ import annotations

import importlib
import shutil
from pathlib import Path
from typing import Dict, List, Optional

from flask import (
	Blueprint,
	Flask,
	current_app,
	flash,
	redirect,
	render_template,
	request,
	url_for,
)

from extract_audio.extractor import extract_audio

# Local import with graceful fallback if the dependency is misconfigured.
try:
	from transcribe.transcribe import HindiTranscriber
except Exception as exc:  # pragma: no cover - defensive
	HindiTranscriber = Optional[object]  # type: ignore
	_transcriber_import_error = exc
else:
	_transcriber_import_error = None


DEFAULT_CONFIG: Dict[str, object] = {
	"SECRET_KEY": "dev-secret-key",  # Replace in production.
	"VIDEO_DIR": Path("videos"),
	"AUDIO_DIR": Path("audios"),
	"DEFAULT_AUDIO_FORMAT": "wav",
}


def create_app(config: Optional[Dict[str, object]] = None) -> Flask:
	"""Application factory to create Flask app instances."""

	app = Flask(__name__, template_folder="templates")
	_load_config(app, config)
	_ensure_storage_dirs(app)
	_register_blueprints(app)
	_register_cli(app)
	return app


def _load_config(app: Flask, config: Optional[Dict[str, object]]) -> None:
	"""Load default and user provided configuration."""

	for key, value in DEFAULT_CONFIG.items():
		app.config.setdefault(key, value)

	if config:
		app.config.update(config)


def _ensure_storage_dirs(app: Flask) -> None:
	"""Create storage directories if they do not exist."""

	for key in ("VIDEO_DIR", "AUDIO_DIR"):
		directory = Path(app.config[key])
		directory.mkdir(parents=True, exist_ok=True)


def _register_blueprints(app: Flask) -> None:
	"""Attach blueprints."""

	app.register_blueprint(_create_main_blueprint())


def _register_cli(app: Flask) -> None:
	"""Placeholder for future CLI commands."""

	pass


def _create_main_blueprint() -> Blueprint:
	"""Create the main blueprint serving the web UI."""

	bp = Blueprint("main", __name__)

	@bp.route("/", methods=["GET", "POST"])
	def index():
		videos = _available_videos()
		transcript_text: Optional[str] = None
		audio_filename: Optional[str] = None

		if request.method == "POST":
			selected_video = request.form.get("video_name")
			if not selected_video:
				flash("Select a video before submitting.", "warning")
				return redirect(url_for("main.index"))

			video_path = _resolve_video_path(selected_video)
			if not video_path.exists():
				flash(f"Video not found: {selected_video}", "danger")
				return redirect(url_for("main.index"))

			extraction_result = extract_audio(str(video_path), output_format=current_app.config["DEFAULT_AUDIO_FORMAT"])

			if not extraction_result.get("success"):
				flash(extraction_result.get("error", "Audio extraction failed."), "danger")
				return redirect(url_for("main.index"))

			audio_path = _move_audio_to_library(extraction_result["audio_path"])

			try:
				transcriber = _get_transcriber()
				transcript_text = transcriber.transcribe(str(audio_path)) if transcriber else None
			except Exception as exc:  # pragma: no cover - defensive logging
				flash(f"Transcription failed: {exc}", "danger")
				transcript_text = None

			audio_filename = audio_path.name

		return render_template(
			"index.html",
			videos=videos,
			transcript=transcript_text,
			audio_filename=audio_filename,
		)

	@bp.route("/health", methods=["GET"])
	def healthcheck():
		"""Simple healthcheck endpoint for monitoring."""

		return {"status": "ok"}

	return bp


def _available_videos() -> List[str]:
	"""Return a list of available video filenames."""

	video_dir = Path(current_app.config["VIDEO_DIR"])
	files = [file.name for file in video_dir.glob("*") if file.is_file()]
	files.sort()
	return files


def _resolve_video_path(filename: str) -> Path:
	"""Resolve a video filename against the configured directory."""

	safe_name = Path(filename).name
	return Path(current_app.config["VIDEO_DIR"]) / safe_name


def _move_audio_to_library(source_path: str) -> Path:
	"""Move extracted audio into the configured audio directory."""

	audio_dir = Path(current_app.config["AUDIO_DIR"])
	audio_dir.mkdir(parents=True, exist_ok=True)

	source = Path(source_path)
	destination = audio_dir / source.name

	if source.resolve() == destination.resolve():
		return destination

	if destination.exists():
		destination.unlink()

	shutil.move(str(source), str(destination))
	return destination


def _get_transcriber():
	"""Create or fetch a cached transcriber instance."""

	if _transcriber_import_error is not None:
		raise RuntimeError(f"Failed to import HindiTranscriber: {_transcriber_import_error}")

	transcriber = current_app.extensions.get("hindi_transcriber")
	if transcriber is None:
		transcriber = _init_transcriber()
		current_app.extensions["hindi_transcriber"] = transcriber
	return transcriber


def _init_transcriber():
	"""Instantiate HindiTranscriber with light patching for missing keys."""

	try:
		return HindiTranscriber()
	except KeyError as exc:
		if exc.args and exc.args[0] == "faster_whisper_available":
			module = importlib.import_module("transcribe.transcribe")
			original_detect = module.detect_environment

			def patched_detect_environment():  # type: ignore[assignment]
				env = original_detect()
				env.setdefault("faster_whisper_available", False)
				return env

			module.detect_environment = patched_detect_environment  # type: ignore[assignment]
			return HindiTranscriber()
		raise


if __name__ == "__main__":  # pragma: no cover - manual invocation
	application = create_app()
	application.run(debug=True)
