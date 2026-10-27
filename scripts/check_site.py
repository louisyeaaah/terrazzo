"""Static checks for the GitHub Pages site (no dependencies)."""
from __future__ import annotations

import re
import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse

SITE = Path(__file__).resolve().parents[1] / "site"
PAGES = ["index.html", "zh.html"]

VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link",
        "meta", "param", "source", "track", "wbr"}


class TagBalance(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[str] = []
        self.errors: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag not in VOID:
            self.stack.append(tag)

    def handle_endtag(self, tag):
        if tag in VOID:
            return
        if not self.stack:
            self.errors.append(f"stray </{tag}>")
            return
        if self.stack[-1] != tag:
            self.errors.append(f"expected </{self.stack[-1]}> before </{tag}>")
            if tag in self.stack:
                while self.stack and self.stack.pop() != tag:
                    pass
            return
        self.stack.pop()


def check_page(name: str) -> list[str]:
    path = SITE / name
    text = path.read_text(encoding="utf-8")
    issues: list[str] = []
    parser = TagBalance()
    parser.feed(text)
    parser.close()
    issues.extend(parser.errors)
    if parser.stack:
        issues.append(f"unclosed tags: {parser.stack}")
    for ref in re.findall(r'(?:href|src)="([^"]+)"', text):
        parsed = urlparse(ref)
        if parsed.scheme in ("http", "https") or ref.startswith("#") or ref.startswith("mailto:"):
            continue
        target = (SITE / parsed.path).resolve()
        if not target.exists():
            issues.append(f"missing local asset: {ref}")
    if text.count("```"):
        issues.append("markdown fence leaked into HTML")
    return issues


def main() -> int:
    failed = False
    for name in PAGES:
        issues = check_page(name)
        if issues:
            failed = True
            print(f"{name}:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print(f"{name}: ok")
    for asset in ("styles.css", "app.js", "favicon.svg"):
        if not (SITE / asset).exists():
            failed = True
            print(f"missing asset: {asset}")
    if failed:
        return 1
    print("site checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
