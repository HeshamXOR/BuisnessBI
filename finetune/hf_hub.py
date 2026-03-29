"""Utilities to upload/download fine-tuned artifacts from Hugging Face Hub."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


DEFAULT_LORA_REPO = "HeshamXOR/business-analyst-phi3-mini-lora"
DEFAULT_MERGED_REPO = "HeshamXOR/business-analyst-phi3-mini-merged"
DEFAULT_LORA_DIR = "finetune/output/lora_adapter"
DEFAULT_MERGED_DIR = "finetune/output/merged_model"


def _resolve_token(cli_token: str | None) -> str:
    token = cli_token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if not token:
        raise ValueError(
            "Missing Hugging Face token. Provide --token or set HF_TOKEN env var."
        )
    return token


def upload_folder_to_hub(
    repo_id: str,
    folder_path: str,
    token: str,
    private: bool = False,
) -> None:
    api = HfApi(token=token)
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=private)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(folder),
        commit_message=f"Upload artifacts from {folder}",
    )


def download_model_from_hub(repo_id: str, local_dir: str, token: str) -> str:
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    return snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        token=token,
        resume_download=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload/download fine-tuned model artifacts on HF Hub")
    sub = parser.add_subparsers(dest="command", required=True)

    upload = sub.add_parser("upload", help="Upload a local artifact folder to HF Hub")
    upload.add_argument("--repo-id", required=True, help="Target model repo, e.g. user/name")
    upload.add_argument("--folder", required=True, help="Local folder path to upload")
    upload.add_argument("--token", default=None, help="HF token (or use HF_TOKEN env var)")
    upload.add_argument("--private", action="store_true", help="Create private repository")

    download = sub.add_parser("download", help="Download an HF model repo snapshot locally")
    download.add_argument("--repo-id", required=True, help="Source model repo, e.g. user/name")
    download.add_argument("--local-dir", required=True, help="Destination local directory")
    download.add_argument("--token", default=None, help="HF token (or use HF_TOKEN env var)")

    quick = sub.add_parser("quick-download", help="Download one of the default project repos")
    quick.add_argument(
        "--which",
        choices=["lora", "merged"],
        default="merged",
        help="Which default repo to download",
    )
    quick.add_argument("--token", default=None, help="HF token (or use HF_TOKEN env var)")

    args = parser.parse_args()
    token = _resolve_token(getattr(args, "token", None))

    if args.command == "upload":
        upload_folder_to_hub(
            repo_id=args.repo_id,
            folder_path=args.folder,
            token=token,
            private=args.private,
        )
        print(f"Uploaded {args.folder} -> https://huggingface.co/{args.repo_id}")

    elif args.command == "download":
        local = download_model_from_hub(
            repo_id=args.repo_id,
            local_dir=args.local_dir,
            token=token,
        )
        print(f"Downloaded {args.repo_id} to {local}")

    elif args.command == "quick-download":
        repo_id = DEFAULT_MERGED_REPO if args.which == "merged" else DEFAULT_LORA_REPO
        local_dir = DEFAULT_MERGED_DIR if args.which == "merged" else DEFAULT_LORA_DIR
        local = download_model_from_hub(repo_id=repo_id, local_dir=local_dir, token=token)
        print(f"Downloaded {args.which} model from {repo_id} to {local}")


if __name__ == "__main__":
    main()
