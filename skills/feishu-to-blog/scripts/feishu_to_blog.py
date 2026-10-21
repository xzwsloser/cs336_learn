#!/usr/bin/env python3
"""Convert a Feishu/Lark document into a CS336 static-blog note."""

from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import mimetypes
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse


TEMP_DIR_NAME = ".feishu-to-blog-tmp"
FEISHU_HOST_SUFFIXES = (".feishu.cn", ".larksuite.com")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".bmp"}
FEISHU_MEDIA_PREFIXES = (
    "callout",
    "grid",
    "column",
    "whiteboard",
    "source",
    "sheet",
    "bitable",
    "cite",
    "synced_reference",
)

ATTR_RE = re.compile(
    r"""([A-Za-z_:][-A-Za-z0-9_:.]*)\s*=\s*(?:"([^"]*)"|'([^']*)')"""
)
MD_IMAGE_RE = re.compile(r"!\[([^\]\n]*)\]\((https?://[^)\s]+)\)")
MD_IMAGE_ANY_RE = re.compile(r'!\[[^\]\n]*\]\(([^)\s]+)(?:\s+"[^"]*")?\)')
HTML_IMAGE_RE = re.compile(r"<img\b([^>]*?)/?>", re.IGNORECASE | re.DOTALL)
WHITEBOARD_RE = re.compile(r"<whiteboard\b([^>]*?)/?>", re.IGNORECASE | re.DOTALL)
CALLOUT_RE = re.compile(r"<callout\b([^>]*)>(.*?)</callout\s*>", re.IGNORECASE | re.DOTALL)
BLOCKQUOTE_RE = re.compile(r"<blockquote\b[^>]*>(.*?)</blockquote\s*>", re.IGNORECASE | re.DOTALL)
CITE_RE = re.compile(r"<cite\b([^>]*)>.*?</cite\s*>", re.IGNORECASE | re.DOTALL)
SOURCE_RE = re.compile(r"<source\b([^>]*?)/?>", re.IGNORECASE | re.DOTALL)
SHEET_RE = re.compile(r"<sheet\b([^>]*?)/?>", re.IGNORECASE | re.DOTALL)
BITABLE_RE = re.compile(r"<bitable\b([^>]*?)/?>", re.IGNORECASE | re.DOTALL)
SYNCED_REFERENCE_RE = re.compile(r"<synced_reference\b([^>]*?)/?>", re.IGNORECASE | re.DOTALL)
HEADING_BOLD_RE = re.compile(r"^(#{1,6})[ \t]+\*\*(.+?)\*\*[ \t]*$", re.MULTILINE)
GRID_OPEN_RE = re.compile(r"<grid\b([^>]*)>", re.IGNORECASE)
COLUMN_OPEN_RE = re.compile(r"<column\b([^>]*)>", re.IGNORECASE)
CSS_COLOR_RE = re.compile(
    r"^(?:#[0-9a-fA-F]{3,8}|rgba?\([0-9.,%\s]+\)|[A-Za-z][A-Za-z-]*)$"
)

FEISHU_CALLOUT_COLORS: dict[str, tuple[str, str]] = {
    "red": ("#fdecec", "#d83931"),
    "orange": ("#fff3e8", "#de7802"),
    "yellow": ("#fff8e1", "#dbb100"),
    "green": ("#e8f7ee", "#2ea121"),
    "blue": ("#eaf3ff", "#245bdb"),
    "purple": ("#f3edff", "#6425d0"),
    "gray": ("#f2f3f5", "#646a73"),
    "light-red": ("#fdecec", "#f54a45"),
    "light-orange": ("#fff3e8", "#ff8800"),
    "light-yellow": ("#fff8e1", "#f5c400"),
    "light-green": ("#e8f7ee", "#34c759"),
    "light-blue": ("#eaf3ff", "#3370ff"),
    "light-purple": ("#f3edff", "#7f3bf5"),
    "light-gray": ("#f5f6f7", "#8f959e"),
    "medium-red": ("#fbdcdc", "#d83931"),
    "medium-orange": ("#ffe2c7", "#de7802"),
    "medium-yellow": ("#fff0b3", "#dbb100"),
    "medium-green": ("#d4f2df", "#2ea121"),
    "medium-blue": ("#d9e7ff", "#245bdb"),
    "medium-purple": ("#e7dcff", "#6425d0"),
    "medium-gray": ("#e4e5e8", "#646a73"),
}

FEISHU_CALLOUT_EMOJI: dict[str, str] = {
    "info": "ℹ️",
    "tip": "💡",
    "note": "📝",
    "warning": "⚠️",
    "error": "❌",
    "success": "✅",
    "bulb": "💡",
    "memo": "📝",
    "exclamation": "❗",
    "question": "❓",
}


class ConversionError(RuntimeError):
    pass


@dataclass
class MediaRef:
    token: str
    kind: str
    saved_path: Path | None = None
    output_rel: str | None = None


@dataclass
class MediaSlot:
    ref: MediaRef
    alt: str
    placeholder: str


def parse_attrs(source: str) -> dict[str, str]:
    attrs: dict[str, str] = {}
    for match in ATTR_RE.finditer(source or ""):
        value = match.group(2) if match.group(2) is not None else match.group(3)
        attrs[match.group(1).lower()] = html.unescape(value or "")
    return attrs


def is_feishu_host(host: str) -> bool:
    normalized = (host or "").lower().rstrip(".")
    return any(
        normalized == suffix.lstrip(".") or normalized.endswith(suffix)
        for suffix in FEISHU_HOST_SUFFIXES
    )


def validate_feishu_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise ConversionError("URL must use https")
    if parsed.username or parsed.password:
        raise ConversionError("URL must not contain credentials")
    if not is_feishu_host(parsed.hostname or ""):
        raise ConversionError("URL host must be feishu.cn or larksuite.com")
    parts = [part for part in parsed.path.split("/") if part]
    if not parts or parts[0] not in {"wiki", "docx", "docs", "doc"}:
        raise ConversionError("URL must point to a Feishu wiki or docx document")


def parse_json_output(stdout: str, stderr: str, returncode: int) -> dict[str, Any]:
    text = (stdout or "").strip()
    candidates = [text]
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    detail = (stderr or stdout or "").strip()
    if len(detail) > 1200:
        detail = detail[:1200] + "..."
    raise ConversionError(f"lark-cli returned non-JSON output (exit {returncode}): {detail}")


def run_lark(
    lark_bin: str,
    root: Path,
    identity: str,
    arguments: list[str],
) -> dict[str, Any]:
    command = [lark_bin, *arguments, "--as", identity, "--format", "json"]
    try:
        process = subprocess.run(
            command,
            cwd=root,
            text=True,
            capture_output=True,
            env={**os.environ, "NO_COLOR": "1"},
            check=False,
        )
    except FileNotFoundError as exc:
        raise ConversionError(f"cannot find {lark_bin!r}; install or pass --lark-bin") from exc

    payload = parse_json_output(process.stdout, process.stderr, process.returncode)
    if process.returncode != 0 or payload.get("ok") is False:
        error = payload.get("error") or {}
        message = error.get("message") or (process.stderr or process.stdout).strip()
        hint = error.get("hint")
        if hint:
            message = f"{message} ({hint})"
        raise ConversionError(str(message or "lark-cli command failed"))
    return payload


def detect_slug(title: str, date: str, doc_token: str) -> str:
    lecture = re.search(r"(?i)\b(?:lecture|lec)\s*[-_:#]?\s*0*(\d+)\b", title)
    if lecture:
        return f"lecture-{int(lecture.group(1)):02d}"
    words = re.findall(r"[A-Za-z0-9]+", title.lower())
    if words:
        slug = "-".join(words[:8])[:64].strip("-")
        if slug:
            return slug
    token_suffix = re.sub(r"[^A-Za-z0-9]", "", doc_token)[-6:].lower() or "note"
    return f"feishu-{date.replace('-', '')}-{token_suffix}"


def validate_slug(slug: str) -> None:
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", slug or ""):
        raise ConversionError("slug must contain only lowercase letters, digits, and hyphens")
    if len(slug) > 80:
        raise ConversionError("slug must be at most 80 characters")


def validate_date(value: str) -> None:
    try:
        dt.date.fromisoformat(value)
    except ValueError as exc:
        raise ConversionError("date must use YYYY-MM-DD") from exc


def require_repo(root: Path) -> None:
    required = [root / "notes.json", root / "note.html", root / "notes", root / "assets" / "images"]
    missing = [str(path.relative_to(root)) for path in required if not path.exists()]
    if missing:
        raise ConversionError(f"not a CS336 blog repository; missing: {', '.join(missing)}")


def load_index(root: Path) -> list[dict[str, Any]]:
    path = root / "notes.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConversionError(f"cannot read notes.json: {exc}") from exc
    if not isinstance(data, list) or any(not isinstance(item, dict) for item in data):
        raise ConversionError("notes.json must contain a JSON array of objects")
    return data


def find_index_entry(entries: list[dict[str, Any]], slug: str) -> dict[str, Any] | None:
    return next((entry for entry in entries if entry.get("slug") == slug), None)


def atomic_write(path: Path, content: str) -> None:
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(content, encoding="utf-8")
    os.replace(temp, path)


def update_index(root: Path, entry: dict[str, Any], overwrite: bool) -> None:
    entries = load_index(root)
    matches = [item for item in entries if item.get("slug") == entry["slug"]]
    if matches and not overwrite:
        raise ConversionError(f"notes.json already contains slug {entry['slug']!r}")
    entries = [item for item in entries if item.get("slug") != entry["slug"]]
    entries.append(entry)
    entries.sort(key=lambda item: (str(item.get("date", "")), str(item.get("slug", ""))))
    atomic_write(
        root / "notes.json",
        json.dumps(entries, ensure_ascii=False, indent=2) + "\n",
    )


def _as_blockquote(body: str, prefix: str = "") -> str:
    lines = body.strip("\n").splitlines()
    if not lines:
        return "\n\n"
    output: list[str] = []
    prefix_pending = bool(prefix)
    for line in lines:
        stripped = line.strip()
        if not stripped:
            output.append(">")
            continue
        current = line.rstrip()
        if prefix_pending:
            current = prefix + current
            prefix_pending = False
        output.append("> " + current + "  ")
    if output:
        output[-1] = output[-1].rstrip()
    return "\n\n" + "\n".join(output) + "\n\n"


def _find_matching_tag(text: str, start: int, tag: str) -> tuple[int, int] | None:
    pattern = re.compile(rf"</?{re.escape(tag)}\b[^>]*>", re.IGNORECASE)
    depth = 1
    for match in pattern.finditer(text, start):
        if match.group(0).startswith("</"):
            depth -= 1
        else:
            depth += 1
        if depth == 0:
            return match.start(), match.end()
    return None


def _column_ratio(attrs: dict[str, str]) -> float:
    raw = attrs.get("width-ratio") or attrs.get("width_ratio") or ""
    try:
        ratio = float(raw)
    except (TypeError, ValueError):
        return 1.0
    if ratio <= 0 or ratio > 100 or ratio != ratio:
        return 1.0
    return ratio


def _extract_columns(body: str) -> list[tuple[float, str]]:
    columns: list[tuple[float, str]] = []
    cursor = 0
    while True:
        match = COLUMN_OPEN_RE.search(body, cursor)
        if not match:
            break
        close = _find_matching_tag(body, match.end(), "column")
        if close is None:
            break
        if not columns and body[: match.start()].strip():
            columns.append((1.0, body[: match.start()]))
        columns.append((_column_ratio(parse_attrs(match.group(1))), body[match.end() : close[0]]))
        cursor = close[1]
    trailing = body[cursor:].strip()
    if trailing:
        columns.append((1.0, trailing))
    return columns


def _grid_repl(body: str) -> str:
    columns = _extract_columns(body)
    if not columns:
        return body
    track_sizes = " ".join(f"{ratio:.6g}fr" for ratio, _ in columns)
    output = [f'\n\n<div class="feishu-grid" style="grid-template-columns: {track_sizes};">\n']
    for _, content in columns:
        cleaned = content.strip("\n")
        output.append(f'\n<div class="feishu-column">\n\n{cleaned}\n\n</div>\n')
    output.append("\n</div>\n\n")
    return "".join(output)


def _replace_grids(text: str) -> str:
    match = GRID_OPEN_RE.search(text)
    if not match:
        return text
    close = _find_matching_tag(text, match.end(), "grid")
    if close is None:
        return text
    body = _replace_grids(text[match.end() : close[0]])
    return text[: match.start()] + _grid_repl(body) + _replace_grids(text[close[1] :])


def _callout_color(value: str, role: str) -> str | None:
    key = (value or "").strip().lower()
    if key in FEISHU_CALLOUT_COLORS:
        index = 0 if role == "background" else 1
        return FEISHU_CALLOUT_COLORS[key][index]
    if key and CSS_COLOR_RE.fullmatch(key):
        return value.strip()
    return None


def _callout_style(attrs: dict[str, str]) -> str:
    background = _callout_color(
        attrs.get("background-color") or attrs.get("background_color") or "",
        "background",
    )
    border = _callout_color(
        attrs.get("border-color") or attrs.get("border_color") or "",
        "border",
    )
    text = _callout_color(
        attrs.get("text-color") or attrs.get("text_color") or "",
        "border",
    )
    if background and not border:
        key = (attrs.get("background-color") or attrs.get("background_color") or "").lower()
        border = FEISHU_CALLOUT_COLORS.get(key, (background, background))[1]
    declarations: list[str] = []
    if background:
        declarations.append(f"--feishu-callout-bg: {background}")
    if border:
        declarations.append(f"--feishu-callout-border: {border}")
    if text:
        declarations.append(f"--feishu-callout-text: {text}")
    if not declarations:
        return ""
    return f' style="{html.escape("; ".join(declarations), quote=True)}"'


def _callout_emoji(attrs: dict[str, str]) -> str:
    value = (attrs.get("emoji") or attrs.get("emoji-id") or attrs.get("type") or "").strip()
    if not value:
        return ""
    return FEISHU_CALLOUT_EMOJI.get(value.lower(), value)


def _callout_repl(match: re.Match[str]) -> str:
    attrs = parse_attrs(match.group(1))
    emoji = _callout_emoji(attrs)
    icon = (
        f'<div class="feishu-callout-icon" aria-hidden="true">{html.escape(emoji)}</div>\n'
        if emoji
        else ""
    )
    body = match.group(2).strip("\n")
    return (
        f'\n\n<div class="feishu-callout"{_callout_style(attrs)}>\n'
        f"{icon}"
        f'<div class="feishu-callout-body">\n\n{body}\n\n</div>\n'
        "</div>\n\n"
    )


def _blockquote_repl(match: re.Match[str]) -> str:
    body = re.sub(r"</?p\b[^>]*>", "", match.group(1), flags=re.IGNORECASE)
    return _as_blockquote(body)


def _document_url_from_attrs(attrs: dict[str, str], fallback_type: str = "docx") -> str | None:
    doc_id = (
        attrs.get("doc-id")
        or attrs.get("doc_id")
        or attrs.get("src-token")
        or attrs.get("src_token")
        or attrs.get("token")
    )
    if not doc_id:
        return None
    doc_type = (attrs.get("file-type") or attrs.get("file_type") or fallback_type).lower()
    route = {
        "doc": "docs",
        "docx": "docx",
        "wiki": "wiki",
        "sheet": "sheets",
        "sheets": "sheets",
        "bitable": "base",
        "base": "base",
        "slides": "slides",
        "mindnote": "mindnotes",
        "file": "file",
    }.get(doc_type, "docx")
    return f"https://feishu.cn/{route}/{doc_id}"


def _escape_link_text(value: str) -> str:
    return value.replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]")


def _link(title: str, url: str) -> str:
    return f"[{_escape_link_text(title)}]({url})"


def _cite_repl(match: re.Match[str]) -> str:
    attrs = parse_attrs(match.group(1))
    title = attrs.get("title") or attrs.get("name") or "飞书文档"
    url = _document_url_from_attrs(attrs)
    return _link(title, url) if url else _escape_link_text(title)


def _source_repl(match: re.Match[str]) -> str:
    attrs = parse_attrs(match.group(1))
    name = attrs.get("name") or attrs.get("title") or "附件"
    token = attrs.get("token")
    if token:
        return _link(f"📎 {name}", f"https://feishu.cn/file/{token}")
    return f"📎 {_escape_link_text(name)}"


def _sheet_repl(match: re.Match[str]) -> str:
    attrs = parse_attrs(match.group(1))
    token = attrs.get("token") or attrs.get("spreadsheet-token") or attrs.get("spreadsheet_token")
    return _link("📊 飞书表格", f"https://feishu.cn/sheets/{token}") if token else "📊 飞书表格"


def _bitable_repl(match: re.Match[str]) -> str:
    attrs = parse_attrs(match.group(1))
    token = attrs.get("token") or attrs.get("app-token") or attrs.get("app_token")
    return _link("📊 飞书多维表格", f"https://feishu.cn/base/{token}") if token else "📊 飞书多维表格"


def _synced_reference_repl(match: re.Match[str]) -> str:
    attrs = parse_attrs(match.group(1))
    title = attrs.get("title") or attrs.get("name") or "同步内容"
    url = _document_url_from_attrs(attrs)
    return _link(f"🔗 {title}", url) if url else f"🔗 {_escape_link_text(title)}"


def normalize_feishu_markup(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\ufeff", "").replace("\u200b", "")

    text = _replace_grids(text)
    text = CALLOUT_RE.sub(_callout_repl, text)
    text = BLOCKQUOTE_RE.sub(_blockquote_repl, text)
    text = CITE_RE.sub(_cite_repl, text)
    text = SOURCE_RE.sub(_source_repl, text)
    text = SHEET_RE.sub(_sheet_repl, text)
    text = BITABLE_RE.sub(_bitable_repl, text)
    text = SYNCED_REFERENCE_RE.sub(_synced_reference_repl, text)

    text = re.sub(r"</p\s*>", "\n\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<p\b[^>]*>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<br\s*/?>", "  \n", text, flags=re.IGNORECASE)
    text = re.sub(r"<hr\s*/?>", "\n\n---\n\n", text, flags=re.IGNORECASE)
    text = HEADING_BOLD_RE.sub(r"\1 \2", text)
    text = re.sub(r"(?m)^(#{1,6})[ \t]+", r"\1 ", text)
    return dedent_accidental_code(text)


def _looks_like_list_item(line: str) -> bool:
    return bool(re.match(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)", line))


def dedent_accidental_code(text: str) -> str:
    lines = text.splitlines()
    output: list[str] = []
    in_fence = False
    previous_nonempty = ""
    for line in lines:
        stripped_left = line.lstrip()
        if re.match(r"^(```|~~~)", stripped_left):
            in_fence = not in_fence
            output.append(line)
            if stripped_left:
                previous_nonempty = line
            continue
        if in_fence:
            output.append(line)
            if stripped_left:
                previous_nonempty = line
            continue
        leading = len(line) - len(line.lstrip(" \t"))
        if leading >= 4 and stripped_left:
            should_keep = _looks_like_list_item(line) or _looks_like_list_item(previous_nonempty)
            if not should_keep:
                line = stripped_left
        output.append(line)
        if line.strip():
            previous_nonempty = line
    return "\n".join(output)


def media_token_from_url(url: str) -> str | None:
    parsed = urlparse(url)
    if parsed.scheme != "https" or not is_feishu_host(parsed.hostname or ""):
        return None
    patterns = (
        r"/(?:file|media)/([A-Za-z0-9]+)",
        r"/download/(?:all/)?([A-Za-z0-9]+)",
        r"/([A-Za-z0-9]{12,})(?:/|$)",
    )
    for pattern in patterns:
        match = re.search(pattern, parsed.path)
        if match:
            return match.group(1)
    return None


def _clean_alt(value: str, fallback: str = "图片") -> str:
    cleaned = re.sub(r"\s+", " ", html.unescape(value or "")).strip()
    return cleaned[:400] or fallback


def stage_media(text: str) -> tuple[str, list[MediaRef], list[MediaSlot], list[str]]:
    refs_by_key: dict[tuple[str, str], MediaRef] = {}
    refs: list[MediaRef] = []
    slots: list[MediaSlot] = []
    warnings: list[str] = []

    def add_slot(token: str, kind: str, alt: str) -> str:
        key = (kind, token)
        ref = refs_by_key.get(key)
        if ref is None:
            ref = MediaRef(token=token, kind=kind)
            refs_by_key[key] = ref
            refs.append(ref)
        index = len(slots)
        placeholder = f"@@FEISHU_MEDIA_{index}@@"
        slots.append(MediaSlot(ref=ref, alt=_clean_alt(alt), placeholder=placeholder))
        return placeholder

    def whiteboard_repl(match: re.Match[str]) -> str:
        attrs = parse_attrs(match.group(1))
        token = attrs.get("token")
        if not token:
            warnings.append("A whiteboard without a token was skipped")
            return ""
        return "\n\n" + add_slot(token, "whiteboard", "飞书画板") + "\n\n"

    text = WHITEBOARD_RE.sub(whiteboard_repl, text)

    def html_image_repl(match: re.Match[str]) -> str:
        attrs = parse_attrs(match.group(1))
        source = attrs.get("src", "")
        token = attrs.get("token") or media_token_from_url(source)
        if token:
            return add_slot(token, "media", attrs.get("alt", "图片"))
        if source:
            warnings.append(f"External image kept as a remote URL: {source}")
            return match.group(0)
        return ""

    text = HTML_IMAGE_RE.sub(html_image_repl, text)

    def markdown_image_repl(match: re.Match[str]) -> str:
        alt, url = match.group(1), match.group(2)
        token = media_token_from_url(url)
        if token:
            return add_slot(token, "media", alt)
        warnings.append(f"External image kept as a remote URL: {url}")
        return match.group(0)

    text = MD_IMAGE_RE.sub(markdown_image_repl, text)
    return text, refs, slots, warnings


def _download_payload(ref: MediaRef, output_base: str, lark_bin: str, root: Path, identity: str) -> dict[str, Any]:
    arguments = ["docs", "+media-download", "--token", ref.token, "--output", output_base, "--overwrite"]
    if ref.kind == "whiteboard":
        arguments.extend(["--type", "whiteboard"])
    try:
        return run_lark(lark_bin, root, identity, arguments)
    except ConversionError as exc:
        if ref.kind == "media" and re.search(r"permission|forbidden|403", str(exc), re.IGNORECASE):
            return run_lark(
                lark_bin,
                root,
                identity,
                ["docs", "+media-preview", "--token", ref.token, "--output", output_base, "--overwrite"],
            )
        raise


def download_and_install_media(
    refs: list[MediaRef],
    root: Path,
    slug: str,
    overwrite: bool,
    lark_bin: str,
    identity: str,
    keep_temp: bool,
) -> None:
    if not refs:
        return
    temp_dir = root / TEMP_DIR_NAME
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True)
    moved: list[Path] = []
    try:
        for index, ref in enumerate(refs, start=1):
            output_base = f"{TEMP_DIR_NAME}/{slug}-img-{index:02d}"
            payload = _download_payload(ref, output_base, lark_bin, root, identity)
            data = payload.get("data") or {}
            saved_value = data.get("saved_path")
            saved_path = Path(saved_value) if saved_value else None
            if saved_path is None:
                matches = sorted(temp_dir.glob(f"{slug}-img-{index:02d}*"))
                if len(matches) == 1:
                    saved_path = matches[0]
            if saved_path is None or not saved_path.is_file() or saved_path.stat().st_size == 0:
                raise ConversionError(f"media token {ref.token!r} was not saved to a usable file")
            saved_path = saved_path.resolve()
            if saved_path != root and not saved_path.is_relative_to(root):
                raise ConversionError(f"lark-cli saved media outside the repository: {saved_path}")
            suffix = saved_path.suffix.lower()
            if suffix not in IMAGE_EXTENSIONS:
                guessed = mimetypes.guess_extension(str(data.get("content_type") or "")) or ""
                suffix = guessed.lower()
            if suffix not in IMAGE_EXTENSIONS:
                raise ConversionError(f"media token {ref.token!r} is not an image ({data.get('content_type')!r})")
            ref.saved_path = saved_path
            ref.output_rel = f"assets/images/{slug}-img-{index:02d}{suffix}"

        assets_dir = root / "assets" / "images"
        assets_dir.mkdir(parents=True, exist_ok=True)
        if overwrite:
            for old_asset in assets_dir.glob(f"{slug}-img-*"):
                old_asset.unlink()
        for ref in refs:
            assert ref.saved_path is not None and ref.output_rel is not None
            destination = root / ref.output_rel
            if destination.exists() and not overwrite:
                raise ConversionError(f"asset already exists: {destination.relative_to(root)}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(ref.saved_path), str(destination))
            moved.append(destination)
            os.chmod(destination, 0o644)
    except Exception:
        for path in moved:
            path.unlink(missing_ok=True)
        raise
    finally:
        if temp_dir.exists() and not keep_temp:
            shutil.rmtree(temp_dir)


def first_h1(text: str) -> str | None:
    match = re.search(r"(?m)^#\s+(.+?)\s*$", text)
    return clean_heading(match.group(1)) if match else None


def clean_heading(value: str) -> str:
    value = re.sub(r"^[#\s*_`]+|[#\s*_`]+$", "", value)
    return re.sub(r"\s+", " ", value).strip()


def ensure_h1(text: str, title: str, force_title: bool) -> tuple[str, str]:
    match = re.search(r"(?m)^#\s+.+?\s*$", text)
    if force_title:
        heading = f"# {title}"
        if match:
            text = text[: match.start()] + heading + text[match.end() :]
        else:
            text = heading + "\n\n" + text.lstrip()
        return text, title
    if match:
        return text, clean_heading(match.group(0)) or title
    return f"# {title}\n\n{text.lstrip()}", title


def derive_description(markdown: str) -> str:
    blocks = re.split(r"\n\s*\n", markdown)
    for block in blocks:
        if "@@FEISHU_MEDIA_" in block or block.lstrip().startswith(("#", ">", "```", "~~~", "|")):
            continue
        if re.match(r"^(?:[-*+]\s+|\d+[.)]\s+)", block.lstrip()):
            continue
        line = block.strip().splitlines()[0].strip()
        line = re.sub(r"^[-*+]\s+", "", line)
        line = re.sub(r"^\d+[.)]\s+", "", line)
        line = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", line)
        line = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", line)
        line = re.sub(r"<[^>]+>", "", line)
        line = re.sub(r"`([^`]*)`", r"\1", line)
        line = re.sub(r"\${1,2}([^$\n]+?)\${1,2}", r"\1", line)
        line = re.sub(r"[*_~>]+", "", line)
        line = re.sub(r"\s+", " ", html.unescape(line)).strip()
        if len(line) < 24 or line.endswith((":", "：")):
            continue
        return truncate_description(line)
    return "飞书文档转换的学习笔记。"


def truncate_description(value: str, limit: int = 120) -> str:
    if len(value) <= limit:
        return value
    candidate = value[:limit]
    cut = max(candidate.rfind(mark) for mark in ("。", "；", "！", "？", ".", ";", "!", "?"))
    if cut >= 60:
        return candidate[: cut + 1]
    return candidate.rstrip("，,；;：: ") + "…"


def sanitize_index_text(value: str, limit: int) -> str:
    cleaned = html.unescape(value or "")
    cleaned = re.sub(r"<[^>]+>", "", cleaned)
    cleaned = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", cleaned)
    cleaned = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", cleaned)
    cleaned = re.sub(r"[`*_~]+", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if len(cleaned) > limit:
        cleaned = cleaned[:limit].rstrip() + "…"
    return html.escape(cleaned, quote=False)


def normalize_spacing(text: str) -> str:
    lines = text.splitlines()
    output: list[str] = []
    in_fence = False
    for line in lines:
        stripped = line.lstrip()
        if re.match(r"^(```|~~~)", stripped):
            if not in_fence and output and output[-1].strip():
                output.append("")
            in_fence = not in_fence
            output.append(line)
            if not in_fence:
                output.append("")
            continue
        if in_fence:
            output.append(line)
            continue
        if re.match(r"^#{1,6}\s+", line):
            if output and output[-1].strip():
                output.append("")
            output.append(line)
            output.append("")
            continue
        if re.match(r"^!\[[^\]]*\]\([^)]+\)\s*$", line.strip()):
            if output and output[-1].strip():
                output.append("")
            output.append(line)
            output.append("")
            continue
        output.append(line)
    normalized = "\n".join(output)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip() + "\n"


def replace_media_slots(text: str, slots: list[MediaSlot]) -> str:
    for slot in slots:
        if not slot.ref.output_rel:
            raise ConversionError(f"media token {slot.ref.token!r} has no installed asset path")
        alt = slot.alt.replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]")
        text = text.replace(slot.placeholder, f"![{alt}]({slot.ref.output_rel})")
    return text


def validate_markdown_text(text: str, root: Path, slug: str) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    headings = re.findall(r"(?m)^#\s+.+$", text)
    if not headings:
        errors.append("missing H1 title")
    elif len(headings) > 1:
        warnings.append(f"found {len(headings)} H1 headings")
    first_content = next((line for line in text.splitlines() if line.strip()), "")
    if headings and not first_content.startswith("# "):
        warnings.append("H1 is not the first non-empty line")

    for tag in FEISHU_MEDIA_PREFIXES:
        if re.search(rf"<{tag}\b", text, re.IGNORECASE):
            errors.append(f"unconverted Feishu tag: <{tag}>")

    if re.search(r"!\[[^\]]*\]\([^)]*feishu\.cn/file/", text, re.IGNORECASE):
        errors.append("Markdown still references a remote Feishu file URL as an image")

    for match in MD_IMAGE_ANY_RE.finditer(text):
        target = unquote(match.group(1))
        if target.startswith(("http://", "https://", "data:", "#")):
            if "feishu.cn/file/" in target:
                errors.append(f"unresolved Feishu image: {target}")
            continue
        target_path = target.split("#", 1)[0].split("?", 1)[0]
        if not (root / target_path).is_file():
            errors.append(f"missing local image: {target_path}")

    for match in HTML_IMAGE_RE.finditer(text):
        attrs = parse_attrs(match.group(1))
        source = attrs.get("src", "")
        if not source or source.startswith(("http://", "https://", "data:", "#")):
            continue
        if not (root / unquote(source)).is_file():
            errors.append(f"missing local image: {unquote(source)}")

    fence_open = False
    for line in text.splitlines():
        cleaned = re.sub(r"^(?:\s*>\s*)+", "", line)
        if re.match(r"^\s*(```|~~~)", cleaned):
            fence_open = not fence_open
    if fence_open:
        errors.append("unbalanced code fence")

    without_fences = re.sub(r"```.*?```|~~~.*?~~~", "", text, flags=re.DOTALL)
    without_inline_code = re.sub(r"`[^`\n]*`", "", without_fences)
    if without_inline_code.count("$$") % 2:
        warnings.append("display math delimiter $$ appears unpaired")
    single_dollars = without_inline_code.replace("$$", "").replace("\\$", "")
    if single_dollars.count("$") % 2:
        warnings.append("inline math delimiter $ appears unpaired")

    if re.search(r"(?m)^\|.*\|[ \t]*$", text) and not re.search(r"(?m)^\|?[ \t]*:?-{3,}", text):
        warnings.append("table-like rows found without an obvious separator row")

    return errors, warnings


def validate_repository(root: Path, slug: str | None = None) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    entries = load_index(root)
    seen: set[str] = set()
    selected = [entry for entry in entries if slug is None or entry.get("slug") == slug]
    if slug is not None and not selected:
        errors.append(f"notes.json has no entry for slug {slug!r}")

    for entry in entries:
        entry_slug = str(entry.get("slug", ""))
        if entry_slug in seen:
            errors.append(f"duplicate notes.json slug: {entry_slug!r}")
        seen.add(entry_slug)
        try:
            validate_slug(entry_slug)
        except ConversionError as exc:
            errors.append(f"invalid slug {entry_slug!r}: {exc}")
        for field in ("title", "date", "description"):
            if not str(entry.get(field, "")).strip():
                errors.append(f"notes.json entry {entry_slug!r} is missing {field}")
        try:
            validate_date(str(entry.get("date", "")))
        except ConversionError as exc:
            errors.append(f"notes.json entry {entry_slug!r}: {exc}")

    for entry in selected:
        entry_slug = str(entry.get("slug", ""))
        path = root / "notes" / f"{entry_slug}.md"
        if not path.is_file():
            errors.append(f"missing note file: notes/{entry_slug}.md")
            continue
        text = path.read_text(encoding="utf-8")
        note_errors, note_warnings = validate_markdown_text(text, root, entry_slug)
        errors.extend(f"{entry_slug}: {item}" for item in note_errors)
        warnings.extend(f"{entry_slug}: {item}" for item in note_warnings)

    if slug is None:
        indexed = {str(entry.get("slug", "")) for entry in entries}
        for path in sorted((root / "notes").glob("*.md")):
            if path.stem not in indexed:
                warnings.append(f"note file is not indexed in notes.json: notes/{path.name}")
    return errors, warnings


def convert(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.repo).resolve()
    require_repo(root)
    validate_feishu_url(args.url)
    date = args.date or dt.date.today().isoformat()
    validate_date(date)

    inspect_payload = run_lark(
        args.lark_bin,
        root,
        args.identity,
        ["drive", "+inspect", "--url", args.url],
    )
    inspect_data = inspect_payload.get("data") or {}
    doc_token = str(inspect_data.get("token") or "")
    if not doc_token:
        raise ConversionError("lark-cli did not return a document token")
    doc_type = str(inspect_data.get("type") or "").lower()
    if doc_type and doc_type not in {"doc", "docx"}:
        raise ConversionError(f"unsupported Feishu document type {doc_type!r}; expected docx or doc")

    fetch_payload = run_lark(
        args.lark_bin,
        root,
        args.identity,
        [
            "docs",
            "+fetch",
            "--doc",
            args.url,
            "--doc-format",
            "markdown",
            "--detail",
            "simple",
        ],
    )
    document = (fetch_payload.get("data") or {}).get("document") or {}
    raw_content = document.get("content")
    if not isinstance(raw_content, str) or not raw_content.strip():
        raise ConversionError("lark-cli returned an empty document")

    content = normalize_feishu_markup(raw_content)
    title = args.title or first_h1(content) or str(inspect_data.get("title") or "飞书笔记")
    slug = args.slug or detect_slug(title, date, doc_token)
    validate_slug(slug)

    entries = load_index(root)
    if (find_index_entry(entries, slug) or (root / "notes" / f"{slug}.md").exists()) and not args.overwrite:
        raise ConversionError(f"slug {slug!r} already exists; pass --overwrite or choose --slug")

    content, title = ensure_h1(content, title, force_title=bool(args.title))
    content, refs, slots, media_warnings = stage_media(content)
    description = args.description or derive_description(content)
    description = sanitize_index_text(description, 150)
    if not description:
        description = sanitize_index_text(f"{title} 的飞书学习笔记。", 150)

    download_and_install_media(
        refs,
        root,
        slug,
        args.overwrite,
        args.lark_bin,
        args.identity,
        args.keep_temp,
    )
    content = replace_media_slots(content, slots)
    content = normalize_spacing(content)

    errors, warnings = validate_markdown_text(content, root, slug)
    warnings = media_warnings + warnings
    if errors:
        raise ConversionError("blog validation failed: " + "; ".join(errors))

    entry = {
        "slug": slug,
        "title": sanitize_index_text(title, 180),
        "date": date,
        "description": description,
    }
    note_path = root / "notes" / f"{slug}.md"
    atomic_write(note_path, content)
    update_index(root, entry, args.overwrite)

    final_errors, final_warnings = validate_repository(root, slug)
    warnings.extend(final_warnings)
    if final_errors:
        raise ConversionError("post-write validation failed: " + "; ".join(final_errors))

    return {
        "ok": True,
        "slug": slug,
        "title": entry["title"],
        "date": date,
        "markdown": f"notes/{slug}.md",
        "images": len(refs),
        "index": "notes.json",
        "warnings": warnings,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert a Feishu document URL into a CS336 blog note.")
    parser.add_argument("--url", help="Feishu wiki or docx URL")
    parser.add_argument("--repo", default=".", help="repository root (default: current directory)")
    parser.add_argument("--slug", help="target slug under notes/")
    parser.add_argument("--title", help="override the document title")
    parser.add_argument("--date", help="publication date (YYYY-MM-DD)")
    parser.add_argument("--description", help="override the notes.json description")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing slug and its images")
    parser.add_argument("--as", dest="identity", default="user", choices=("user", "bot"))
    parser.add_argument("--lark-bin", default="lark-cli")
    parser.add_argument("--keep-temp", action="store_true", help="keep downloaded staging files")
    parser.add_argument("--check", action="store_true", help="validate existing notes instead of converting")
    parser.add_argument("--check-slug", help="limit --check to one slug")
    parser.add_argument("--json", action="store_true", help="print a JSON summary")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.check:
            root = Path(args.repo).resolve()
            require_repo(root)
            errors, warnings = validate_repository(root, args.check_slug)
            result = {"ok": not errors, "errors": errors, "warnings": warnings}
            if args.json:
                print(json.dumps(result, ensure_ascii=False, indent=2))
            else:
                for warning in warnings:
                    print(f"warning: {warning}", file=sys.stderr)
                for error in errors:
                    print(f"error: {error}", file=sys.stderr)
                if not errors:
                    print("validation: OK")
            return 0 if not errors else 1

        if not args.url:
            parser.error("--url is required unless --check is used")
        result = convert(args)
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            for warning in result["warnings"]:
                print(f"warning: {warning}", file=sys.stderr)
            print(f"created {result['markdown']}")
            print(f"updated {result['index']} ({result['slug']}, {result['date']})")
            print(f"downloaded {result['images']} image(s)")
            print("validation: OK")
        return 0
    except ConversionError as exc:
        if args.json:
            print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False, indent=2))
        else:
            print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
