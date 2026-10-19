"""Install pinned official Qdrant Windows binaries, Web UI and Chinese BGE weights."""

import hashlib
import json
import zipfile

from setup_resources import ROOT, download


def checked(url, target, digest):
    download(url, target)
    with target.open("rb") as source:
        actual = hashlib.file_digest(source, "sha256").hexdigest()
    if actual != digest:
        raise RuntimeError(f"SHA256 mismatch: {target.name}; remove the file and retry.")


def main():
    version = "1.19.2"
    filename = "qdrant-x86_64-pc-windows-msvc.zip"
    archive = ROOT / "runtime/downloads" / filename
    checked(
        f"https://github.com/qdrant/qdrant/releases/download/v{version}/{filename}",
        archive,
        "7d86596f16c6e85d45a50312f5e16ccb51e059b62e3d7308110cb65b4c799a4d",
    )
    target = ROOT / "tools/qdrant"
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zipped:
        zipped.extractall(target)
    ui = ROOT / "runtime/downloads/qdrant-ui-0.2.19.zip"
    checked(
        "https://github.com/qdrant/qdrant-web-ui/releases/download/v0.2.19/dist-qdrant.zip",
        ui,
        "be50e32d84aa41c249cad5d2b109d7538cf283fbdab71dee5f5bfa40ca29fd53",
    )
    with zipfile.ZipFile(ui) as zipped:
        zipped.extractall(target / "static")
    revision = "7999e1d3359715c523056ef9478215996d62a620"
    base = f"https://huggingface.co/BAAI/bge-small-zh-v1.5/resolve/{revision}"
    for name in (
        "config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.txt",
    ):
        path = ROOT / "runtime/models/bge-small-zh-v1.5" / name
        if name == "model.safetensors":
            checked(
                f"{base}/{name}",
                path,
                "354763b9b1357bc9c44f62c6be2276321081ed2567773608c0d0785b61d5a026",
            )
        else:
            download(f"{base}/{name}", path)
    # JSON is also valid YAML; absolute paths allow start.ps1 to use the project cwd.
    static = target / "static"
    if (static / "dist/index.html").is_file():
        static /= "dist"
    configuration = {
        "log_level": "WARN",
        "telemetry_disabled": True,
        "storage": {
            "storage_path": (ROOT / "runtime/qdrant/storage").as_posix(),
            "snapshots_path": (ROOT / "runtime/qdrant/snapshots").as_posix(),
        },
        "service": {
            "host": "127.0.0.1",
            "http_port": 6333,
            "grpc_port": 6334,
            "enable_cors": False,
            "static_content_dir": static.as_posix(),
        },
    }
    (target / "config.yaml").write_text(json.dumps(configuration), encoding="utf-8")
    print("Qdrant + dashboard + BGE Chinese embedding ready (checksums verified).")


if __name__ == "__main__":
    main()
