def get_last_directory(path: str) -> str:
    """Extract name of the last directory in path.

    CIFAR10/CIFAR10-ACGAN-Mod/ -> CIFAR10-ACGAN-Mod
    CIFAR10/CIFAR10-ACGAN-Mod -> CIFAR10-ACGAN-Mod

    Args:
        path: Path to the directory with '/' as a splitting character.

    Returns:
        Name of the last directory in path.
    """
    return list(filter(lambda x: len(x) > 0, path.split("/")))[-1]
