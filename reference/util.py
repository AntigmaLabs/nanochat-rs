from pathlib import Path


def print_tree(root: Path):
    """Print a simple tree view of a directory."""
    root = Path(root)
    print(root.resolve())

    def walk(dir_path: Path, prefix: str):
        try:
            entries = sorted(
                list(dir_path.iterdir()),
                key=lambda e: (not e.is_dir(), e.name.lower()),
            )
        except Exception:
            return
        total = len(entries)
        for idx, entry in enumerate(entries):
            is_last = idx == total - 1
            connector = "└── " if is_last else "├── "
            print(prefix + connector + entry.name)
            if entry.is_dir():
                extension = "    " if is_last else "│   "
                walk(entry, prefix + extension)

    walk(root, "")