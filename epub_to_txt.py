#!/usr/bin/env python3
"""
Utility script for converting EPUB ebooks into plain text.

The script extracts the spine-defined documents from the EPUB package,
strips HTML markup, and concatenates the resulting text into a single file.
"""

from __future__ import annotations

import argparse
import posixpath
import re
import zipfile
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, Iterable, List, Optional
import xml.etree.ElementTree as ET


class SimpleHTMLTextExtractor(HTMLParser):
    """Lightweight HTML-to-text converter suitable for well-formed XHTML."""

    BLOCK_TAGS = {
        "p",
        "div",
        "section",
        "article",
        "header",
        "footer",
        "aside",
        "nav",
        "li",
        "blockquote",
        "pre",
        "table",
        "tr",
        "td",
        "th",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
    }
    LINE_BREAK_TAGS = {"br", "hr"}

    def __init__(self) -> None:
        super().__init__()
        self._chunks: List[str] = []

    def handle_starttag(self, tag: str, attrs: List[tuple[str, Optional[str]]]) -> None:
        tag = tag.lower()
        if tag in self.LINE_BREAK_TAGS:
            self._append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in self.BLOCK_TAGS:
            self._append("\n")

    def handle_data(self, data: str) -> None:
        text = unescape(data.replace("\xa0", " "))
        collapsed = " ".join(text.split())
        if not collapsed:
            return
        if self._chunks and not self._chunks[-1].endswith(("\n", " ")):
            self._chunks.append(" ")
        self._chunks.append(collapsed)

    def get_text(self) -> str:
        raw = "".join(self._chunks)
        lines = raw.splitlines()
        cleaned_lines: List[str] = []
        previous_blank = True
        for line in lines:
            stripped = line.strip()
            if not stripped:
                if not previous_blank:
                    cleaned_lines.append("")
                previous_blank = True
            else:
                cleaned_lines.append(stripped)
                previous_blank = False
        return "\n".join(cleaned_lines).strip()

    def _append(self, fragment: str) -> None:
        if not self._chunks:
            self._chunks.append(fragment)
            return
        if fragment == "\n":
            if not self._chunks[-1].endswith("\n"):
                self._chunks.append(fragment)
            return
        self._chunks.append(fragment)


def detect_encoding(payload: bytes, default: str = "utf-8") -> str:
    """Best-effort detection of the declared XML/HTML encoding."""
    if payload.startswith(b"\xef\xbb\xbf"):
        return "utf-8"
    head = payload[:512]
    match = re.search(rb'encoding=["\']([^"\']+)["\']', head, flags=re.IGNORECASE)
    if match:
        candidate = match.group(1).decode("ascii", errors="ignore").strip()
        if candidate:
            return candidate
    return default


def decode_bytes(payload: bytes) -> str:
    encoding = detect_encoding(payload)
    try:
        return payload.decode(encoding, errors="ignore")
    except LookupError:
        return payload.decode("utf-8", errors="ignore")


def extract_text_from_markup(payload: bytes) -> str:
    text = decode_bytes(payload)
    extractor = SimpleHTMLTextExtractor()
    extractor.feed(text)
    extractor.close()
    return extractor.get_text()


def locate_opf(zip_file: zipfile.ZipFile) -> str:
    try:
        with zip_file.open("META-INF/container.xml") as container:
            tree = ET.parse(container)
    except KeyError as exc:
        raise ValueError("Container definition META-INF/container.xml not found.") from exc

    root = tree.getroot()
    namespace_map = {"container": "urn:oasis:names:tc:opendocument:xmlns:container"}
    rootfile = root.find(".//container:rootfile", namespace_map)
    if rootfile is None:
        raise ValueError("OPF rootfile entry missing from container.xml.")
    opf_path = rootfile.attrib.get("full-path")
    if not opf_path:
        raise ValueError("OPF rootfile path not specified in container.xml.")
    return opf_path


def build_manifest(package_root: ET.Element, ns: str) -> Dict[str, Dict[str, str]]:
    def tag(name: str) -> str:
        return f"{{{ns}}}{name}" if ns else name

    manifest_element = package_root.find(tag("manifest"))
    if manifest_element is None:
        raise ValueError("OPF manifest section not found.")

    manifest: Dict[str, Dict[str, str]] = {}
    for item in manifest_element.findall(tag("item")):
        item_id = item.attrib.get("id")
        href = item.attrib.get("href")
        if not item_id or not href:
            continue
        manifest[item_id] = {
            "href": href,
            "media-type": item.attrib.get("media-type", "").lower(),
        }
    return manifest


def iter_spine_itemrefs(package_root: ET.Element, ns: str) -> Iterable[str]:
    def tag(name: str) -> str:
        return f"{{{ns}}}{name}" if ns else name

    spine = package_root.find(tag("spine"))
    if spine is None:
        raise ValueError("OPF spine section not found.")

    for itemref in spine.findall(tag("itemref")):
        if itemref.attrib.get("linear", "yes").lower() == "no":
            continue
        idref = itemref.attrib.get("idref")
        if idref:
            yield idref


def normalize_member_path(opf_path: str, href: str) -> str:
    if posixpath.isabs(href):
        return href.lstrip("/")
    opf_dir = posixpath.dirname(opf_path)
    if not opf_dir:
        return posixpath.normpath(href)
    return posixpath.normpath(posixpath.join(opf_dir, href))


def extract_spine_text(zip_file: zipfile.ZipFile, opf_path: str) -> List[str]:
    with zip_file.open(opf_path) as opf:
        package_tree = ET.parse(opf)
    package_root = package_tree.getroot()
    namespace_match = re.match(r"\{([^}]+)\}", package_root.tag)
    namespace = namespace_match.group(1) if namespace_match else ""

    manifest = build_manifest(package_root, namespace)
    sections: List[str] = []

    for item_id in iter_spine_itemrefs(package_root, namespace):
        manifest_entry = manifest.get(item_id)
        if not manifest_entry:
            continue
        media_type = manifest_entry.get("media-type", "")
        if media_type and not any(token in media_type for token in ("html", "xml", "text")):
            continue
        member_path = normalize_member_path(opf_path, manifest_entry["href"])
        try:
            with zip_file.open(member_path) as item_file:
                payload = item_file.read()
        except KeyError:
            continue

        if media_type == "text/plain":
            text_content = decode_bytes(payload)
        else:
            text_content = extract_text_from_markup(payload)

        cleaned = text_content.strip()
        if cleaned:
            sections.append(cleaned)

    if not sections:
        raise ValueError("No textual spine items found in the EPUB package.")

    return sections


def convert_epub_to_txt(epub_path: Path, destination: Optional[Path] = None) -> Path:
    epub_path = epub_path.expanduser().resolve()
    if not epub_path.is_file():
        raise FileNotFoundError(f"EPUB file not found: {epub_path}")

    with zipfile.ZipFile(epub_path) as zf:
        opf_path = locate_opf(zf)
        sections = extract_spine_text(zf, opf_path)

    final_text = "\n\n".join(sections)

    output_path = destination or epub_path.with_suffix(".txt")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(final_text + "\n", encoding="utf-8")
    return output_path


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert an EPUB ebook into plain UTF-8 text."
    )
    parser.add_argument("epub", help="Path to the source .epub file.")
    parser.add_argument(
        "-o",
        "--output",
        help="Destination .txt file. Defaults to the source name with a .txt suffix.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    epub_path = Path(args.epub)
    destination = Path(args.output).expanduser() if args.output else None

    try:
        output_path = convert_epub_to_txt(epub_path, destination)
    except Exception as exc:  # noqa: BLE001 - surface helpful message to CLI users
        raise SystemExit(f"Error: {exc}") from exc

    print(f"Wrote text to {output_path}")


if __name__ == "__main__":
    main()
