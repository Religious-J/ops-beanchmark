"""ANSI colors for terminal output (disabled when stdout is not a TTY)."""

import sys


def use_ansi_color() -> bool:
    return sys.stdout.isatty()


def green(text: str) -> str:
    return f"\033[32m{text}\033[0m" if use_ansi_color() else text


def red(text: str) -> str:
    return f"\033[31m{text}\033[0m" if use_ansi_color() else text
