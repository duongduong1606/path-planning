"""Extract selected comma2k19 chunks safely on Windows/NTFS."""

from __future__ import annotations

import shutil
from pathlib import Path
from zipfile import ZipFile


BASE = Path(__file__).resolve().parent / "data" / "comma2k19"
ARCHIVE_DIR = BASE / "archives"
OUTPUT_DIR = BASE / "extracted"
ARCHIVES = ("Chunk_3.zip", "Chunk_4.zip")


def safe_target(member_name: str) -> Path:
    safe_name = member_name.replace("|", "_")
    root = OUTPUT_DIR.resolve()
    target = (OUTPUT_DIR / safe_name).resolve()
    if target != root and root not in target.parents:
        raise RuntimeError(f"Unsafe ZIP member path: {member_name}")
    return target


def extract_archive(path: Path) -> tuple[int, int]:
    extracted_files = 0
    extracted_bytes = 0
    with ZipFile(path) as archive:
        members = archive.infolist()
        print(f"Extracting {path.name}: {len(members)} entries", flush=True)
        for index, member in enumerate(members, start=1):
            target = safe_target(member.filename)
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue

            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and target.stat().st_size == member.file_size:
                extracted_files += 1
                extracted_bytes += member.file_size
                continue

            partial = target.with_name(target.name + ".part")
            if partial.exists():
                partial.unlink()
            with archive.open(member) as source, partial.open("wb") as destination:
                shutil.copyfileobj(source, destination, length=1024 * 1024)
            if partial.stat().st_size != member.file_size:
                raise RuntimeError(f"Size mismatch while extracting {member.filename}")
            partial.replace(target)
            extracted_files += 1
            extracted_bytes += member.file_size

            if index % 1000 == 0:
                print(f"  {index}/{len(members)} entries", flush=True)

    return extracted_files, extracted_bytes


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    total_files = 0
    total_bytes = 0
    for archive_name in ARCHIVES:
        archive_path = ARCHIVE_DIR / archive_name
        if not archive_path.is_file():
            raise FileNotFoundError(archive_path)
        files, size = extract_archive(archive_path)
        total_files += files
        total_bytes += size
        print(
            f"Finished {archive_name}: {files} files, {size / 1024**3:.2f} GiB",
            flush=True,
        )

    print(
        f"Completed: {total_files} files, {total_bytes / 1024**3:.2f} GiB",
        flush=True,
    )


if __name__ == "__main__":
    main()
