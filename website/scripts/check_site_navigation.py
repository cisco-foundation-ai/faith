#!/usr/bin/env python3
# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Validate rendered local links and control relationships without network access.

Run after the production build. Unlike source-only checks, this inspects links
rendered inside shared React components and verifies documentation reachability
from the landing page. It also checks IDs and accessible control relationships;
this is not a complete accessibility audit. External URLs are outside this check.
"""

from __future__ import annotations

import argparse
import posixpath
import tempfile
import unittest
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urljoin, urlsplit

SITE = Path(__file__).resolve().parents[1]
ORIGIN = "https://docs.invalid"


class Page(HTMLParser):
    """Collect rendered link targets and accessibility relationships."""

    def __init__(self, html: str):
        super().__init__(convert_charrefs=True)
        self.ids: set[str] = set()
        self.duplicate_ids: set[str] = set()
        self.links: list[str] = []
        self.elements: list[tuple[str, dict]] = []
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        self.elements.append((tag, attributes))
        if attributes.get("id"):
            if attributes["id"] in self.ids:
                self.duplicate_ids.add(attributes["id"])
            self.ids.add(attributes["id"])
        if tag == "a":
            if attributes.get("name"):
                self.ids.add(attributes["name"])
            if attributes.get("href"):
                self.links.append(attributes["href"])


def check_controls(page: Page) -> list[str]:
    """Report duplicate targets and invalid accessible control relationships."""
    errors = [f"duplicate ID {identifier}" for identifier in sorted(page.duplicate_ids)]
    targets = {attrs["id"]: attrs for _, attrs in page.elements if attrs.get("id")}
    for tag, attrs in page.elements:
        for reference in ("aria-controls", "aria-describedby", "aria-labelledby"):
            for target in (attrs.get(reference) or "").split():
                if target not in targets:
                    errors.append(f"{tag}: {reference} references missing ID {target}")
        if tag == "button" and "aria-pressed" in attrs:
            if attrs["aria-pressed"] not in ("true", "false", "mixed"):
                errors.append("button: invalid aria-pressed value")
        # Flag annotations must be absent from keyboard/screen-reader navigation
        # while collapsed, not merely clipped to zero visual height.
        target = attrs.get("aria-controls") or ""
        if "-annotations-" in target and attrs.get("aria-expanded") == "false":
            if target in targets and "hidden" not in targets[target]:
                errors.append(f"collapsed annotation {target} is not hidden")
    return errors


def route_for(path: Path, root: Path) -> str:
    """Map a built HTML file to its canonical documentation route."""
    route = "/" + path.relative_to(root).as_posix()
    if route.endswith("/index.html"):
        return route[:-10]
    return route[:-5]


def _route_aliases(pages: dict[str, Page]) -> dict[str, str]:
    aliases = {}
    for route in pages:
        aliases[route] = route
        aliases[route.rstrip("/")] = route
        aliases[route.rstrip("/") + "/"] = route
        aliases[route.rstrip("/") + ".html"] = route
        aliases[route.rstrip("/") + "/index.html"] = route
    aliases["/index.html"] = "/"
    return aliases


def _check_page_links(
    root: Path,
    source: str,
    page: Page,
    pages: dict[str, Page],
    aliases: dict[str, str],
) -> tuple[set[str], set[str], int]:
    errors = {f"{source}: {error}" for error in check_controls(page)}
    destinations: set[str] = set()
    checked = 0
    for href in page.links:
        target = urlsplit(urljoin(ORIGIN + source, href))
        if target.scheme != "https" or target.netloc != "docs.invalid":
            continue
        checked += 1
        path = posixpath.normpath(unquote(target.path))
        destination = aliases.get(path)
        if destination is None:
            asset = (root / path.lstrip("/")).resolve()
            if not asset.is_relative_to(root.resolve()) or not asset.is_file():
                errors.add(f"{source}: missing local target {href}")
            continue
        destinations.add(destination)
        fragment = unquote(target.fragment).split(":~:text=", 1)[0]
        if fragment and fragment not in pages[destination].ids:
            errors.add(f"{source}: missing anchor {href}")
    return errors, destinations, checked


def _reachable_routes(graph: dict[str, set[str]]) -> set[str]:
    seen, pending = set(), ["/"]
    while pending:
        route = pending.pop()
        if route not in seen:
            seen.add(route)
            pending.extend(graph[route] - seen)
    return seen


def check_site(root: Path) -> tuple[list[str], int, int]:
    """Check all built pages and return errors, page count, and link count."""
    pages = {
        route_for(path, root): Page(path.read_text()) for path in root.rglob("*.html")
    }
    if "/" not in pages:
        return ["No built landing page found; run npm run build first."], 0, 0
    errors: set[str] = set()
    aliases = _route_aliases(pages)
    graph: dict[str, set[str]] = {}
    checked = 0
    for source, page in pages.items():
        link_errors, graph[source], count = _check_page_links(
            root, source, page, pages, aliases
        )
        errors.update(link_errors)
        checked += count
    docs = {route for route in pages if route.startswith("/docs/")}
    if not docs:
        errors.add("No documentation pages found in the build.")
    for route in docs - _reachable_routes(graph):
        errors.add(f"{route}: documentation page unreachable from the landing page")
    return sorted(errors), len(pages), checked


class NavigationTests(unittest.TestCase):
    """Exercise navigation failures and accepted URL/control forms."""

    # unittest reports each public test separately, including setup/fixture use.
    # pylint: disable=too-many-public-methods

    def setUp(self):
        # enterContext closes the directory during unittest cleanup.
        # pylint: disable-next=consider-using-with
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        (self.root / "docs").mkdir()

    def page(self, name: str, html: str):
        """Write one HTML page in the temporary build fixture."""
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(html)

    def test_valid_relative_encoded_and_html_links(self):
        self.page("index.html", '<a href="/docs/intro#section%3Aa">Docs</a>')
        self.page(
            "docs/intro.html",
            '<h1 id="section:a">Intro</h1><a href="./next.html?x=1#details">Next</a><a href="mailto:team@example.com">Mail</a><a href="https://example.com/missing">External</a>',
        )
        self.page(
            "docs/next.html",
            '<details id="details"><summary>Details</summary></details>',
        )
        errors, pages, links = check_site(self.root)
        self.assertEqual(errors, [])
        self.assertEqual((pages, links), (3, 2))

    def test_missing_route_and_anchor_are_reported(self):
        self.page(
            "index.html",
            '<a href="/docs/intro#missing">Intro</a><a href="/docs/missing">Missing</a>',
        )
        self.page("docs/intro.html", '<h1 id="intro">Intro</h1>')
        errors, _, _ = check_site(self.root)
        self.assertEqual(len(errors), 2)
        self.assertTrue(any("missing anchor" in error for error in errors))
        self.assertTrue(any("missing local target" in error for error in errors))

    def test_unlinked_documentation_is_reported(self):
        self.page("index.html", "Home")
        self.page("docs/orphan.html", "Orphan")
        self.assertIn("unreachable", check_site(self.root)[0][0])

    def test_directory_routes_assets_and_same_page_anchors(self):
        self.page("index.html", '<a href="/docs/intro/">Docs</a>')
        self.page(
            "docs/intro/index.html",
            '<h1 id="intro">Intro</h1><a href="#intro">Top</a><a href="/asset.txt">Asset</a>',
        )
        (self.root / "asset.txt").write_text("fixture")
        self.assertEqual(check_site(self.root)[0], [])

    def test_multiple_accessible_references_resolve(self):
        page = Page(
            '<button aria-controls="yaml command" aria-describedby="status" aria-pressed="true">Model</button><div id="yaml"></div><div id="command"></div><p id="status" role="status">Selected</p>'
        )
        self.assertEqual(check_controls(page), [])

    def test_missing_accessible_reference_is_reported(self):
        page = Page('<button aria-controls="missing">Expand</button>')
        self.assertIn("references missing ID missing", check_controls(page)[0])

    def test_duplicate_ids_are_reported(self):
        page = Page('<div id="duplicate"></div><div id="duplicate"></div>')
        self.assertEqual(check_controls(page), ["duplicate ID duplicate"])

    def test_collapsed_annotations_must_be_hidden(self):
        button = '<button aria-controls="table-annotations-0" aria-expanded="false">Flag</button>'
        page = Page(button + '<tr id="table-annotations-0"></tr>')
        self.assertIn("is not hidden", check_controls(page)[0])
        page = Page(button + '<tr id="table-annotations-0" hidden></tr>')
        self.assertEqual(check_controls(page), [])

    def test_invalid_selection_state_is_reported(self):
        self.assertEqual(
            check_controls(Page('<button aria-pressed="yes">Model</button>')),
            ["button: invalid aria-pressed value"],
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, default=SITE / "build")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        unittest.main(argv=[__file__], verbosity=2)
    else:
        failures, page_count, link_count = check_site(args.build)
        if failures:
            raise SystemExit("\n".join(failures))
        print(
            f"Verified {link_count} rendered internal links and control relationships across {page_count} pages; all documentation pages are reachable."
        )
