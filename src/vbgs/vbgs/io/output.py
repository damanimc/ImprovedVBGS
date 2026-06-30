import json
import os
import re
from pathlib import Path

from vbgs.io.paths import add_gaussian_splatting_to_syspath
from vbgs.model.utils import store_model


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
add_gaussian_splatting_to_syspath()


def default_output_root() -> Path:
    return Path(os.environ.get("VBGS_OUTPUT_DIR", PACKAGE_ROOT / "output"))


MAX_RUN_NAME_LEN = 64


def clean_run_name(value: str | None, fallback: str = "run") -> str:
    text = (value or fallback).strip().lower()
    text = re.sub(r"[^a-z0-9._-]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-._")
    text = text or fallback
    if len(text) <= MAX_RUN_NAME_LEN:
        return text
    return text[:MAX_RUN_NAME_LEN].strip("-._") or fallback


def validate_output_path(path: Path) -> Path:
    resolved = Path(path).resolve()
    forbidden = (PACKAGE_ROOT / "scripts" / "data").resolve()
    try:
        resolved.relative_to(forbidden)
    except ValueError:
        parts = resolved.parts
        for idx in range(len(parts) - 1):
            if parts[idx] == "scripts" and parts[idx + 1] == "data":
                break
        else:
            return resolved
    raise ValueError(
        "Refusing to write generated outputs under src/vbgs/scripts/data. "
        f"Use {default_output_root()} or pass --output-root/--output-dir."
    )


def unique_run_dir(output_root: Path, run_name: str) -> Path:
    run_dir = output_root / clean_run_name(run_name)
    if not run_dir.exists():
        return run_dir
    for idx in range(1, 1000):
        candidate = output_root / f"{run_dir.name}-{idx:02d}"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not allocate output directory for {run_name}")


def export_ply(model_path: Path, ply_path: Path) -> str | None:
    try:
        from vbgs.render.volume import save_inria_splat_ply

        save_inria_splat_ply(model_path, ply_path)
        return None
    except Exception as exc:
        return str(exc)


class RunOutput:
    def __init__(self, path: Path):
        self.path = validate_output_path(Path(path))
        self.path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def create(
        cls,
        run_name: str | None = None,
        output_root: Path | None = None,
        output_dir: Path | None = None,
        unique: bool = False,
    ) -> "RunOutput":
        if output_dir is not None:
            return cls(Path(output_dir))
        root = validate_output_path(
            Path(output_root) if output_root is not None else default_output_root()
        )
        root.mkdir(parents=True, exist_ok=True)
        name = clean_run_name(run_name)
        return cls(unique_run_dir(root, name) if unique else root / name)

    def checkpoint(self, model, data_params, name: str) -> Path:
        model_path = self.path / name
        store_model(model, data_params, model_path)
        return model_path

    def final_model(self, model, data_params, metrics: dict, stem: str = "model_final") -> dict:
        model_path = self.checkpoint(model, data_params, f"{stem}.json")
        ply_path = self.path / f"{stem}.ply"
        ply_error = export_ply(model_path, ply_path)
        metrics["output_dir"] = str(self.path)
        metrics["final_model"] = str(model_path)
        metrics["final_ply"] = str(ply_path) if ply_error is None else None
        if ply_error is not None:
            metrics["ply_error"] = ply_error
            print(f"Skipping PLY export via original exporter: {ply_error}")
        return metrics

    def metrics(self, metrics: dict, filename: str = "metrics.json") -> Path:
        metrics_path = self.path / filename
        with metrics_path.open("w") as f:
            json.dump(metrics, f, indent=2)
        return metrics_path

    def image_dir(self, name: str) -> Path:
        path = self.path / name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def ensure_dir(self, path: Path) -> Path:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        return path
