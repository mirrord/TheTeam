"""Source ingestion + chunking for the CraftAnalyzer tool.

Resolves a :class:`~pithos.tools.craft_analyzer.models.CraftAnalysisRequest`
(raw text, a single file, or a directory/glob collection of files) into
normalized source text, then splits long text into overlapping chunks so the
subagent only ever sees a bounded amount of context per call.
"""

from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import Optional

from .models import CraftAnalysisRequest


class SourceResolutionError(ValueError):
    """Raised when a :class:`CraftAnalysisRequest` cannot be resolved to text."""


def resolve_source(
    request: CraftAnalysisRequest,
    include: Optional[list[str]] = None,
    exclude: Optional[list[str]] = None,
    max_files: int = 200,
) -> tuple[str, str]:
    """Resolve a request into ``(text, title)``.

    Exactly one of ``request.text``, ``request.file_path``, or
    ``request.roots`` must be set.

    Args:
        request: The analysis request.
        include: Glob patterns used when ``request.roots`` is a directory
            collection. Defaults to ``["**/*.txt", "**/*.md"]``.
        exclude: Glob patterns to exclude from a directory collection.
        max_files: Safety cap on the number of files read from ``roots``.

    Returns:
        A ``(text, title)`` tuple. ``title`` falls back to
        ``request.title``, then a derived name, then ``"untitled"``.

    Raises:
        SourceResolutionError: If zero or more than one source is set, or if
            no text could be resolved.
    """
    provided = [
        name
        for name, value in (
            ("text", request.text),
            ("file_path", request.file_path),
            ("roots", request.roots),
        )
        if value
    ]
    if len(provided) == 0:
        raise SourceResolutionError("one of text, file_path, or roots must be provided")
    if len(provided) > 1:
        raise SourceResolutionError(
            f"only one of text, file_path, or roots may be provided (got: {provided})"
        )

    if request.text:
        title = request.title or "untitled"
        return request.text, title

    if request.file_path:
        path = Path(request.file_path).expanduser()
        if not path.exists():
            raise SourceResolutionError(f"file not found: {path}")
        if not path.is_file():
            raise SourceResolutionError(f"not a file: {path}")
        text = path.read_text(encoding="utf-8")
        title = request.title or path.stem
        return text, title

    # request.roots: directory/glob collection.
    files = _collect_files(request.roots or [], include, exclude, max_files)
    if not files:
        raise SourceResolutionError(f"no files matched under roots: {request.roots}")
    parts = []
    for path in files:
        try:
            parts.append(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, OSError):
            continue
    if not parts:
        raise SourceResolutionError(
            f"no readable text files found under roots: {request.roots}"
        )
    text = "\n\n".join(parts)
    title = request.title or (
        files[0].stem if len(files) == 1 else f"{len(files)}_files"
    )
    return text, title


def _collect_files(
    roots: list[str],
    include: Optional[list[str]],
    exclude: Optional[list[str]],
    max_files: int,
) -> list[Path]:
    """Enumerate files beneath ``roots`` using glob include/exclude patterns.

    Mirrors :class:`pithos.coding_nodes.ListFilesNode`'s enumeration logic.
    """
    include_patterns = list(include or ["**/*.txt", "**/*.md"])
    exclude_patterns = list(exclude or [])
    results: list[Path] = []

    for root_str in roots:
        root = Path(root_str).expanduser()
        if root.is_file():
            results.append(root)
            continue
        if not root.exists():
            raise SourceResolutionError(f"root not found: {root}")
        if not root.is_dir():
            raise SourceResolutionError(f"root is not a directory: {root}")

        seen: set[str] = set()
        for pattern in include_patterns:
            for path in sorted(root.glob(pattern)):
                if not path.is_file():
                    continue
                rel = path.as_posix()
                if rel in seen:
                    continue
                if _is_excluded(path, root, exclude_patterns):
                    continue
                seen.add(rel)
                results.append(path)
                if len(results) >= max_files:
                    break
            if len(results) >= max_files:
                break

    results.sort(key=lambda p: p.as_posix())
    return results


def _is_excluded(path: Path, root: Path, exclude_patterns: list[str]) -> bool:
    try:
        rel = path.relative_to(root).as_posix()
    except ValueError:
        rel = path.as_posix()
    full = path.as_posix()
    for pattern in exclude_patterns:
        if fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(full, pattern):
            return True
    return False


def chunk_text(
    text: str,
    char_cap: int = 6000,
    overlap: int = 300,
    max_chunks: int = 20,
) -> list[str]:
    """Split ``text`` into overlapping chunks of at most ``char_cap`` chars.

    Args:
        text: Source text to split.
        char_cap: Maximum characters per chunk.
        overlap: Number of trailing characters from the previous chunk
            repeated at the start of the next, to preserve cross-boundary
            context.
        max_chunks: Hard cap on the number of chunks returned; remaining text
            beyond this cap is dropped.

    Returns:
        A list of chunk strings. Returns an empty list for empty/blank text.
    """
    body = (text or "").strip()
    if not body:
        return []
    if char_cap <= 0:
        raise ValueError("char_cap must be positive")
    if overlap < 0 or overlap >= char_cap:
        raise ValueError("overlap must be >= 0 and less than char_cap")

    if len(body) <= char_cap:
        return [body]

    chunks: list[str] = []
    start = 0
    step = char_cap - overlap
    length = len(body)
    while start < length and len(chunks) < max_chunks:
        end = min(start + char_cap, length)
        chunks.append(body[start:end])
        if end >= length:
            break
        start += step
    return chunks
