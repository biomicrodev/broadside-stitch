from pathlib import Path


def read_paths(path: Path) -> list[Path]:
    with path.open("r") as file:
        lines = [l.strip() for l in file.readlines()]
        lines = [Path(l) for l in lines if l]
    return lines
