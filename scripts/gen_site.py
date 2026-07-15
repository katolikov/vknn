#!/usr/bin/env python3
"""Build the VKNN documentation site: render the project Markdown into a small, clean, self-contained
static HTML site under docs/site/ (one obvious entry point, docs/site/index.html).

Dependency-free on purpose (matches the engine): a compact GitHub-flavored-Markdown -> HTML converter
plus one stylesheet and one small script, both generated next to the pages. No CDNs, no web fonts,
no frameworks. A PAGES entry whose source ends in .html is passed through verbatim as the article
body (used by the interactive architecture guide). Run via `./build.sh --docs`.
"""
import hashlib
import html
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "site")

# (source, output html, nav title, nav section). First entry is the home page.
# A .html source is inserted as-is (no markdown conversion); everything else is markdown.
PAGES = [
    ("README.md",                          "index.html",               "Overview",             "Get started"),
    ("docs/architecture-guide.html",       "architecture-guide.html",  "How VKNN works",       "Get started"),
    ("docs/class-explorer.html",           "class-explorer.html",      "Neural brain",         "Get started"),
    ("AGENTS.md",                          "agents.html",              "Contributor guide",    "Get started"),
    ("skills/README.md",                   "howto.html",               "How-to guides",        "Get started"),
    ("skills/compile-and-run-a-model.md",  "howto-compile-run.html",   "Compile & run a model","Guides"),
    ("docs/running-an-llm.md",             "running-an-llm.html",      "Run an LLM",           "Guides"),
    ("docs/running-a-vlm.md",              "running-a-vlm.html",       "Run a VLM",            "Guides"),
    ("skills/run-yonosplat.md",            "howto-yonosplat.html",     "Run YoNoSplat",        "Guides"),
    ("skills/benchmark-a-model.md",        "howto-benchmark.html",     "Benchmark a model",    "Guides"),
    ("skills/add-an-operator.md",          "howto-add-operator.html",  "Add an operator",      "Guides"),
    ("skills/add-a-backend.md",            "howto-add-backend.html",   "Add a backend",        "Guides"),
    ("docs/architecture.md",               "architecture.html",        "Architecture",         "Reference"),
    ("docs/config.md",                     "config.html",              "Config",               "Reference"),
    ("docs/op-coverage.md",                "op-coverage.html",         "Op coverage",          "Reference"),
    ("docs/adding-an-operator.md",         "adding-an-operator.html",  "Adding an operator",   "Reference"),
    ("docs/adding-a-backend.md",           "adding-a-backend.html",    "Adding a backend",     "Reference"),
    ("docs/benchmark.md",                  "benchmark.html",           "Benchmarks",           "Reference"),
    ("docs/limitations.md",                "limitations.html",         "Limitations",          "Reference"),
    ("docs/mnn-analysis.md",               "mnn-analysis.html",        "MNN analysis",         "Reference"),
    ("examples/README.md",                 "examples.html",            "Examples",             "Reference"),
    ("app-demo/README.md",                 "app-demo.html",            "Android app demo",     "Reference"),
]
# Architecture Decision Records, added programmatically (kept in nav under "Design records").
for fn in sorted(os.listdir(os.path.join(ROOT, "docs", "adr"))):
    if fn.endswith(".md"):
        num, rest = fn[:-3].split("-", 1)
        title = "ADR " + num + " · " + rest.replace("-", " ")
        PAGES.append(("docs/adr/" + fn, "adr-" + num + ".html", title, "Design records"))

# Map every plausible link spelling of a source doc to its output page, for intra-site links.
# First mapping wins, so the root README owns the bare "README.md" key.
LINKMAP = {}
for src, out, _, _ in PAGES:
    base = os.path.basename(src)
    keys = [src, base, "./" + base, "../" + src, src.replace("docs/", "../docs/")]
    if base == "README.md" and "/" in src:  # a directory link means that directory's README page
        d = os.path.dirname(src)
        keys += [d, d + "/", "./" + d, "./" + d + "/"]
    for key in keys:
        LINKMAP.setdefault(key, out)
LINKMAP["docs/adr/"] = "adr-0001.html"
LINKMAP["docs/adr"] = "adr-0001.html"

# Repo-relative links with no site page (source files, directories, LICENSE, ...) resolve to
# GitHub, so nothing inside the site 404s. A ":<line>" suffix becomes a #L<line> fragment.
GITHUB = "https://github.com/katolikov/vknn"


# --------------------------------------------------------------------------- inline markdown
_ENTITY = re.compile(r"&(?:[a-zA-Z][a-zA-Z0-9]*|#\d+);")


def esc(text):
    """HTML-escape, but leave existing named/numeric entities (&rarr; &ge; ...) intact."""
    out, i = [], 0
    for m in _ENTITY.finditer(text):
        out.append(html.escape(text[i:m.start()], quote=False))
        out.append(m.group(0))
        i = m.end()
    out.append(html.escape(text[i:], quote=False))
    return "".join(out)


# Directory of the page currently being rendered, relative to ROOT; main() sets it per page so
# relative links resolve against the right source location.
CURRENT_SRC_DIR = ""


def rewrite_link(url):
    anchor = ""
    if "#" in url:
        url, anchor = url.split("#", 1)
        anchor = "#" + anchor
    m = re.search(r":(\d+)$", url)
    line = m.group(1) if m else ""
    url = re.sub(r":\d+$", "", url)
    if url in LINKMAP:
        return LINKMAP[url] + anchor
    if anchor and url == "":
        return anchor  # same-page anchor
    if re.match(r"^[a-z][a-z0-9+.-]*:", url):  # http(s):, mailto:, ... — leave external links alone
        return url + anchor
    # Repo file or directory without a site page: link to GitHub instead of a local 404.
    absolute = os.path.normpath(os.path.join(ROOT, CURRENT_SRC_DIR, url))
    if absolute.startswith(ROOT + os.sep) and os.path.exists(absolute):
        rel = os.path.relpath(absolute, ROOT).replace(os.sep, "/")
        if os.path.isdir(absolute):
            return "%s/tree/main/%s" % (GITHUB, rel)
        return "%s/blob/main/%s%s" % (GITHUB, rel, "#L" + line if line else anchor)
    return url + anchor


def inline(text):
    # 1. pull out inline code spans so their contents aren't touched by other rules
    spans = []

    def stash(m):
        spans.append("<code>" + html.escape(m.group(1), quote=False) + "</code>")
        return "\x00%d\x00" % (len(spans) - 1)

    text = re.sub(r"`([^`]+)`", stash, text)
    # 2. escape the rest (entity-aware)
    text = esc(text)
    # 3. links, bold, italic
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)",
                  lambda m: '<a href="%s">%s</a>' % (html.escape(rewrite_link(m.group(2)), quote=True), m.group(1)),
                  text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<!\*)\*(?!\*)([^*\n]+)\*(?!\*)", r"<em>\1</em>", text)
    # 4. restore code spans
    text = re.sub(r"\x00(\d+)\x00", lambda m: spans[int(m.group(1))], text)
    return text


def slug(text):
    # Underscores survive (GitHub-style), so anchors written against GitHub rendering keep working.
    return re.sub(r"[^a-z0-9_]+", "-", re.sub(r"<[^>]+>", "", text).lower()).strip("-")


def plain(text):
    """Strip tags and entities from rendered inline HTML, for ToC labels."""
    return re.sub(r"&[a-zA-Z#0-9]+;", "", re.sub(r"<[^>]+>", "", text))


# --------------------------------------------------------------------------- block markdown
def convert(md, headings=None):
    """Render markdown to HTML. When `headings` is a list, h2/h3 (slug, level, label) are appended
    to it for the on-page ToC."""
    lines = md.split("\n")
    out, i, n = [], 0, len(lines)
    while i < n:
        line = lines[i]

        # fenced code block
        m = re.match(r"^```\s*([\w+-]*)\s*$", line)
        if m:
            lang = m.group(1)
            i += 1
            buf = []
            while i < n and not re.match(r"^```\s*$", lines[i]):
                buf.append(lines[i])
                i += 1
            i += 1
            cls = ' class="language-%s"' % lang if lang else ""
            out.append("<pre><code%s>%s</code></pre>" % (cls, html.escape("\n".join(buf), quote=False)))
            continue

        # blank
        if not line.strip():
            i += 1
            continue

        # raw HTML block (centered headers, badges, etc.) — pass through verbatim until a blank line
        if re.match(r"^\s*</?[a-zA-Z]", line) and not line.lstrip().startswith("|"):
            buf = []
            while i < n and lines[i].strip():
                buf.append(lines[i])
                i += 1
            out.append("\n".join(buf))
            continue

        # heading (anchor link included; h2/h3 recorded for the ToC)
        m = re.match(r"^(#{1,6})\s+(.*?)\s*#*\s*$", line)
        if m:
            level = len(m.group(1))
            content = inline(m.group(2))
            sid = slug(m.group(2))
            anchor = '<a class="anchor" href="#%s" aria-label="Link to this section">#</a>' % sid
            out.append('<h%d id="%s">%s%s</h%d>' % (level, sid, content, anchor, level))
            if headings is not None and level in (2, 3):
                headings.append((sid, level, plain(content)))
            i += 1
            continue

        # horizontal rule
        if re.match(r"^\s*([-*_])\s*(\1\s*){2,}$", line):
            out.append("<hr>")
            i += 1
            continue

        # table (header row + |---| separator); wrapped so wide tables scroll, not the page
        if line.lstrip().startswith("|") and i + 1 < n and re.match(r"^\s*\|?[\s:|-]+\|?\s*$", lines[i + 1]) and "-" in lines[i + 1]:
            def cells(row):
                row = row.strip()
                row = row[1:] if row.startswith("|") else row
                row = row[:-1] if row.endswith("|") else row
                return [c.strip() for c in row.split("|")]
            head = cells(line)
            i += 2
            body = []
            while i < n and lines[i].lstrip().startswith("|"):
                body.append(cells(lines[i]))
                i += 1
            t = ['<div class="table-wrap"><table>', "<thead><tr>"] + ["<th>%s</th>" % inline(c) for c in head] + ["</tr></thead>", "<tbody>"]
            for r in body:
                t.append("<tr>" + "".join("<td>%s</td>" % inline(c) for c in r) + "</tr>")
            t.append("</tbody></table></div>")
            out.append("".join(t))
            continue

        # blockquote
        if line.lstrip().startswith(">"):
            buf = []
            while i < n and lines[i].lstrip().startswith(">"):
                buf.append(re.sub(r"^\s*>\s?", "", lines[i]))
                i += 1
            out.append("<blockquote>%s</blockquote>" % convert("\n".join(buf)))
            continue

        # list (ordered / unordered); items may hold multiple blocks (nested lists, code, paragraphs)
        m = re.match(r"^(\s*)([-*+]|\d+\.)\s+", line)
        if m:
            base = len(m.group(1))
            block = [line]
            i += 1
            while i < n:
                cur = lines[i]
                if not cur.strip():  # blank: keep only if the list continues past it
                    j = i + 1
                    while j < n and not lines[j].strip():
                        j += 1
                    nxt_indent = (len(lines[j]) - len(lines[j].lstrip())) if j < n else -1
                    sibling = j < n and re.match(r"^ {%d}([-*+]|\d+\.)\s" % base, lines[j])
                    if j < n and (nxt_indent > base or sibling):
                        block.append(cur)
                        i += 1
                        continue
                    break
                indent = len(cur) - len(cur.lstrip())
                if indent > base or re.match(r"^ {%d}([-*+]|\d+\.)\s" % base, cur):
                    block.append(cur)
                    i += 1
                else:
                    break
            out.append(render_list(block, base))
            continue

        # paragraph
        buf = []
        while i < n and lines[i].strip() and not re.match(r"^(#{1,6}\s|```|\s*[-*+]\s|\s*\d+\.\s|>|\|)", lines[i]):
            buf.append(lines[i])
            i += 1
        out.append("<p>%s</p>" % inline(" ".join(s.strip() for s in buf)))
    return "\n".join(out)


def render_list(block, base):
    """Render a list block. Each item's content (the marker line's tail + any deeper-indented lines,
    dedented) is converted recursively, so nested lists, code blocks, and multi-paragraph items work."""
    marker = re.compile(r"^( *)([-*+]|\d+\.)( +)(.*)$")
    items = []  # {kind, lines:[...]}
    cur = None
    for ln in block:
        m = marker.match(ln)
        if m and len(m.group(1)) == base:
            ci = len(m.group(1)) + len(m.group(2)) + len(m.group(3))
            cur = {"kind": "ol" if m.group(2)[0].isdigit() else "ul", "ci": ci, "lines": [m.group(4)]}
            items.append(cur)
        elif cur is not None:
            cur["lines"].append(re.sub(r"^ {0,%d}" % cur["ci"], "", ln) if ln.strip() else "")
    if not items:
        return ""
    kind = items[0]["kind"]
    out = []
    for it in items:
        inner = convert("\n".join(it["lines"]).strip())
        one = re.fullmatch(r"<p>(.*)</p>", inner, re.S)  # tighten single-paragraph items
        if one:
            inner = one.group(1)
        out.append("<li>%s</li>" % inner)
    return "<%s>\n%s\n</%s>" % (kind, "\n".join(out), kind)


# --------------------------------------------------------------------------- page template
def nav_html(active):
    """Sidebar navigation. The ADR section renders as a <details> so its 16 entries stay folded
    unless an ADR page is active."""
    sections = []
    seen = []
    for _, _, _, sec in PAGES:
        if sec not in seen:
            seen.append(sec)
    for sec in seen:
        links = []
        has_active = False
        for _, out, title, s in PAGES:
            if s != sec:
                continue
            if out == "class-explorer.html":  # reachable via the pinned Neural brain button instead
                continue
            cls = ' class="active" aria-current="page"' if out == active else ""
            has_active = has_active or out == active
            links.append('<a href="%s"%s>%s</a>' % (out, cls, html.escape(title)))
        if sec == "Design records":
            sections.append(
                '<details class="nav-fold"%s><summary class="nav-title">%s</summary>%s</details>'
                % (" open" if has_active else "", html.escape(sec), "".join(links)))
        else:
            sections.append('<div class="nav-section"><div class="nav-title">%s</div>%s</div>'
                            % (html.escape(sec), "".join(links)))
    return "\n".join(sections)


def toc_html(headings):
    if len(headings) < 2:
        return ""
    items = ['<li class="d%d"><a href="#%s">%s</a></li>' % (lvl, sid, html.escape(label))
             for sid, lvl, label in headings]
    return ('<aside class="toc" aria-label="On this page"><div class="toc-title">On this page</div>'
            '<ul>%s</ul></aside>' % "".join(items))


def pagenav_html(idx):
    parts = []
    if idx > 0:
        _, out, title, _ = PAGES[idx - 1]
        parts.append('<a class="pn prev" href="%s"><span>Previous</span>%s</a>' % (out, html.escape(title)))
    else:
        parts.append("<span></span>")
    if idx + 1 < len(PAGES):
        _, out, title, _ = PAGES[idx + 1]
        parts.append('<a class="pn next" href="%s"><span>Next</span>%s</a>' % (out, html.escape(title)))
    return '<nav class="pagenav">%s</nav>' % "".join(parts)


# A "V" cut into a compute-grid square; inlined so the site stays a folder of flat files.
FAVICON = ("data:image/svg+xml," + html.escape(
    "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'>"
    "<rect width='16' height='16' rx='3' fill='#1f2933'/>"
    "<path d='M4 4l4 8 4-8' fill='none' stroke='#e8590c' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'/>"
    "</svg>", quote=True))

PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light dark">
<title>{title} · VKNN</title>
<link rel="icon" href="{favicon}">
<link rel="stylesheet" href="styles.css?h={css_hash}">
<script>try{{var t=localStorage.getItem('vknn-theme');if(t)document.documentElement.dataset.theme=t}}catch(e){{}}</script>
</head>
<body>
<input type="checkbox" id="nav-toggle" hidden>
<label for="nav-toggle" class="nav-burger" aria-label="Toggle navigation"><span></span></label>
<aside class="sidebar">
  <div class="brand-row">
    <a class="brand" href="index.html"><span class="logo">VKNN</span><span class="brand-sub">Vulkan Neural Network</span></a>
    <button class="theme-toggle" type="button" aria-label="Toggle color theme" title="Toggle color theme">◐</button>
  </div>
  <div class="nav-filter"><input type="search" placeholder="Filter pages…" aria-label="Filter pages"></div>
  <a class="nav-brain{brain_active}" href="class-explorer.html">Neural brain<span>the class graph, explorable</span></a>
  <nav>{nav}</nav>
  <div class="nav-foot">{api}On-device inference engine</div>
</aside>
<div class="page">
<main>
<article class="content">
<p class="kicker">{kicker}</p>
{body}
<footer class="art-foot">
{pagenav}
<p class="src-note">Source: <code>{src}</code></p>
</footer>
</article>
</main>
{toc}
</div>
<script src="site.js?h={js_hash}" defer></script>
</body>
</html>
"""

STYLES = """/* VKNN docs — generated by scripts/gen_site.py. Self-contained: system fonts, no external requests. */
:root{
  --bg:#fcfcfb; --surface:#f5f5f2; --fg:#26292e; --head:#17191d; --muted:#69707a;
  --line:#e5e4df; --line-strong:#b9b8b1;
  --link:#155bc7; --accent:#e8590c;
  --code-bg:#16181d; --code-fg:#dee4ec; --code-line:#2a2e36;
  --inline:#efeeea; --inline-fg:#33363c; --quote-bg:#f6f5f1;
  --sel:#dfe9f8;
  --tok-com:#7d8590; --tok-str:#9ecb88; --tok-kw:#8fb8f6; --tok-num:#e2b86b; --tok-pre:#c39ac9;
}
:root[data-theme="dark"]{
  --bg:#111417; --surface:#0c0f12; --fg:#c9d1d9; --head:#e6edf3; --muted:#8b949e;
  --line:#25292e; --line-strong:#4a4f56;
  --link:#6ca4f8; --accent:#f4793b;
  --code-bg:#0c0f12; --code-fg:#c9d1d9; --code-line:#25292e;
  --inline:#22262b; --inline-fg:#c9d1d9; --quote-bg:#171b1f;
  --sel:#1c3358;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme]){
    --bg:#111417; --surface:#0c0f12; --fg:#c9d1d9; --head:#e6edf3; --muted:#8b949e;
    --line:#25292e; --line-strong:#4a4f56;
    --link:#6ca4f8; --accent:#f4793b;
    --code-bg:#0c0f12; --code-fg:#c9d1d9; --code-line:#25292e;
    --inline:#22262b; --inline-fg:#c9d1d9; --quote-bg:#171b1f;
    --sel:#1c3358;
  }
}
*{box-sizing:border-box}
::selection{background:var(--sel)}
html{scroll-behavior:smooth}
body{margin:0;font:15.5px/1.65 ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI","Inter",Roboto,"Helvetica Neue",Arial,sans-serif;
  color:var(--fg);background:var(--bg);-webkit-font-smoothing:antialiased}
a{color:var(--link);text-decoration:none}
a:hover{text-decoration:underline;text-underline-offset:2px}

/* ---- sidebar ---- */
.sidebar{position:fixed;top:0;left:0;width:264px;height:100vh;overflow-y:auto;background:var(--surface);
  border-right:1px solid var(--line);display:flex;flex-direction:column;padding:18px 0 10px}
.brand-row{display:flex;align-items:flex-start;justify-content:space-between;padding:2px 18px 14px;border-bottom:1px solid var(--line)}
.brand:hover{text-decoration:none}
.logo{display:block;font-weight:750;font-size:19px;letter-spacing:.02em;color:var(--head)}
.logo::after{content:"";display:inline-block;width:7px;height:7px;margin-left:7px;background:var(--accent);
  clip-path:polygon(0 0,100% 0,50% 100%)}
.brand-sub{display:block;font-size:11.5px;color:var(--muted);margin-top:1px}
.theme-toggle{border:1px solid var(--line);background:none;color:var(--muted);font-size:14px;line-height:1;
  width:26px;height:26px;border-radius:6px;cursor:pointer;padding:0}
.theme-toggle:hover{color:var(--fg);border-color:var(--line-strong)}
.nav-filter{padding:12px 18px 2px}
.nav-filter input{width:100%;font:inherit;font-size:13px;color:var(--fg);background:var(--bg);
  border:1px solid var(--line);border-radius:6px;padding:5px 9px;outline:none}
.nav-filter input:focus{border-color:var(--link)}
.nav-brain{display:block;margin:12px 18px 2px;padding:8px 12px;border:1px solid var(--line-strong);
  border-radius:8px;font-size:13px;font-weight:650;color:var(--head);
  background:color-mix(in srgb,var(--accent) 7%,var(--surface))}
.nav-brain span{display:block;font-size:11px;font-weight:400;color:var(--muted);margin-top:1px}
.nav-brain:hover{text-decoration:none;border-color:var(--accent)}
.nav-brain.active{border-color:var(--accent);box-shadow:inset 0 0 0 1px var(--accent)}
.nav-section{margin:14px 0 2px}
.nav-title{font-size:10.5px;text-transform:uppercase;letter-spacing:.1em;color:var(--muted);
  padding:4px 18px;font-weight:650}
.sidebar nav a{display:block;padding:4.5px 18px 4.5px 17px;color:var(--fg);font-size:13.5px;
  border-left:2px solid transparent;opacity:.85}
.sidebar nav a:hover{opacity:1;color:var(--head);text-decoration:none;background:color-mix(in srgb,var(--line) 40%,transparent)}
.sidebar nav a.active{color:var(--head);border-left-color:var(--accent);font-weight:600;opacity:1}
.sidebar nav a.nav-hide{display:none}
.nav-fold{margin:14px 0 2px}
.nav-fold summary{cursor:pointer;list-style:none}
.nav-fold summary::-webkit-details-marker{display:none}
.nav-fold summary::after{content:"›";margin-left:6px;display:inline-block;transition:transform .12s;color:var(--muted)}
.nav-fold[open] summary::after{transform:rotate(90deg)}
.nav-foot{margin-top:auto;padding:14px 18px 6px;font-size:11.5px;color:var(--muted);border-top:1px solid var(--line)}

/* ---- layout ---- */
.page{margin-left:264px;display:flex;justify-content:center;gap:0}
main{min-width:0;flex:1;display:flex;justify-content:center}
.content{max-width:880px;width:100%;padding:44px 44px 80px}
.kicker{font-size:11.5px;text-transform:uppercase;letter-spacing:.12em;color:var(--muted);
  font-weight:650;margin:0 0 6px}
.kicker:empty{display:none}

/* ---- right "on this page" rail ---- */
.toc{display:none}
@media(min-width:1240px){
  .toc{display:block;width:220px;flex-shrink:0;position:sticky;top:0;align-self:flex-start;
    max-height:100vh;overflow-y:auto;padding:48px 20px 40px 0;font-size:12.5px}
  .toc-title{font-size:10.5px;text-transform:uppercase;letter-spacing:.1em;color:var(--muted);font-weight:650;margin-bottom:8px}
  .toc ul{list-style:none;margin:0;padding:0;border-left:1px solid var(--line)}
  .toc li a{display:block;padding:3px 0 3px 12px;color:var(--muted);border-left:2px solid transparent;margin-left:-1px}
  .toc li.d3 a{padding-left:24px}
  .toc li a:hover{color:var(--fg);text-decoration:none}
  .toc li a.active{color:var(--head);border-left-color:var(--accent)}
}

/* ---- typography ---- */
h1,h2,h3,h4{line-height:1.25;font-weight:700;color:var(--head);margin:1.7em 0 .55em;letter-spacing:-.012em}
h1{font-size:1.85rem;margin-top:.15em}
h2{font-size:1.4rem;margin-top:2em;padding-bottom:.3em;border-bottom:1px solid var(--line)}
h3{font-size:1.12rem}
h4{font-size:1rem}
.anchor{margin-left:.4em;color:var(--muted);opacity:0;font-weight:400;font-size:.85em}
h1:hover .anchor,h2:hover .anchor,h3:hover .anchor,h4:hover .anchor{opacity:1}
.anchor:hover{color:var(--link);text-decoration:none}
p{margin:.7em 0}
hr{border:0;border-top:1px solid var(--line);margin:2.2em 0}
ul,ol{padding-left:1.45em;margin:.6em 0}
li{margin:.28em 0}
li>ul,li>ol{margin:.25em 0}
strong{color:var(--head)}
img{max-width:100%;height:auto;border-radius:6px}

/* ---- code ---- */
code{font-family:ui-monospace,"SF Mono",SFMono-Regular,"Cascadia Code","JetBrains Mono",Menlo,Consolas,monospace;
  font-size:.86em;background:var(--inline);color:var(--inline-fg);padding:.13em .38em;border-radius:5px}
pre{position:relative;background:var(--code-bg);color:var(--code-fg);border:1px solid var(--code-line);
  padding:14px 16px;border-radius:8px;overflow-x:auto;margin:1em 0;font-size:13px;line-height:1.6}
pre code{background:none;color:inherit;padding:0;font-size:13px;border-radius:0}
pre .copy{position:absolute;top:7px;right:7px;border:1px solid var(--code-line);background:var(--code-bg);
  color:var(--tok-com);font:11px/1 inherit;font-family:inherit;padding:5px 8px;border-radius:5px;cursor:pointer;
  opacity:0;transition:opacity .12s}
pre:hover .copy{opacity:1}
pre .copy:hover{color:var(--code-fg)}
.tok-com{color:var(--tok-com)} .tok-str{color:var(--tok-str)} .tok-kw{color:var(--tok-kw)}
.tok-num{color:var(--tok-num)} .tok-pre{color:var(--tok-pre)}

/* ---- blocks ---- */
blockquote{margin:1em 0;padding:.55em 1.05em;background:var(--quote-bg);border-left:3px solid var(--line-strong);
  color:var(--fg)}
blockquote p{margin:.3em 0}

/* booktabs-style tables: strong top/bottom rules, thin header rule, no vertical lines.
   Cells wrap aggressively so reference tables fit the column without horizontal scrolling;
   the wrapper's overflow-x is only the narrow-screen fallback. */
.table-wrap{overflow-x:auto;margin:1.1em 0}
table{border-collapse:collapse;width:100%;font-size:13.2px}
th,td{padding:6px 10px;text-align:left;vertical-align:top;border:0;overflow-wrap:break-word}
td code{font-size:.84em;overflow-wrap:anywhere}
td:first-child code{overflow-wrap:normal;white-space:nowrap}
thead th{border-top:2px solid var(--line-strong);border-bottom:1px solid var(--line-strong);
  font-weight:650;color:var(--head)}
tbody tr{border-bottom:1px solid var(--line)}
tbody tr:last-child{border-bottom:2px solid var(--line-strong)}
tbody tr:hover td{background:color-mix(in srgb,var(--line) 30%,transparent)}

/* ---- article footer ---- */
.art-foot{margin-top:56px;border-top:1px solid var(--line)}
.pagenav{display:flex;justify-content:space-between;gap:14px;padding:18px 0 4px}
.pn{display:block;font-size:14px;font-weight:600;color:var(--head);max-width:46%}
.pn span{display:block;font-size:11px;font-weight:650;text-transform:uppercase;letter-spacing:.1em;color:var(--muted);margin-bottom:2px}
.pn:hover{text-decoration:none;color:var(--link)}
.pn.next{text-align:right;margin-left:auto}
.src-note{font-size:12px;color:var(--muted);margin-top:14px}
.src-note code{background:none;padding:0;color:var(--muted)}

/* ---- mobile ---- */
.nav-burger{display:none}
@media(max-width:900px){
  .sidebar{transform:translateX(-100%);transition:transform .18s;z-index:20;box-shadow:none}
  .brand-row{padding-left:60px}
  #nav-toggle:checked ~ .sidebar{transform:none;box-shadow:0 0 0 100vmax rgba(0,0,0,.28)}
  .page{margin-left:0}
  .content{padding:64px 20px 80px}
  .nav-burger{display:flex;position:fixed;top:10px;left:10px;z-index:30;width:40px;height:40px;
    align-items:center;justify-content:center;background:var(--surface);border:1px solid var(--line);
    border-radius:8px;cursor:pointer}
  .nav-burger span,.nav-burger span::before,.nav-burger span::after{display:block;content:"";width:16px;height:2px;
    background:var(--fg);position:relative;border-radius:1px}
  .nav-burger span::before{position:absolute;top:-5px}
  .nav-burger span::after{position:absolute;top:5px}
}
@media print{
  .sidebar,.nav-burger,.toc,.pagenav,.theme-toggle,pre .copy{display:none}
  .page{margin:0}
  pre{border:1px solid #bbb;white-space:pre-wrap}
}
"""

SITE_JS = r"""// VKNN docs — generated by scripts/gen_site.py. Theme toggle, nav filter, ToC scrollspy,
// copy-to-clipboard on code blocks, and a small conservative syntax highlighter.
(function () {
  'use strict';

  // ---- theme toggle (persisted; default follows prefers-color-scheme) ----
  var toggle = document.querySelector('.theme-toggle');
  if (toggle) toggle.addEventListener('click', function () {
    var root = document.documentElement;
    var dark = root.dataset.theme ? root.dataset.theme === 'dark'
      : matchMedia('(prefers-color-scheme: dark)').matches;
    root.dataset.theme = dark ? 'light' : 'dark';
    try { localStorage.setItem('vknn-theme', root.dataset.theme); } catch (e) {}
  });

  // ---- keep the active page visible in the sidebar (ADRs live inside a fold) ----
  var active = document.querySelector('.sidebar nav a.active');
  if (active && active.scrollIntoView) active.scrollIntoView({ block: 'nearest' });

  // ---- sidebar filter ----
  var filter = document.querySelector('.nav-filter input');
  if (filter) filter.addEventListener('input', function () {
    var q = filter.value.trim().toLowerCase();
    document.querySelectorAll('.sidebar nav a').forEach(function (a) {
      a.classList.toggle('nav-hide', q !== '' && a.textContent.toLowerCase().indexOf(q) < 0);
    });
    document.querySelectorAll('.sidebar nav .nav-fold').forEach(function (d) {
      if (q !== '') d.open = !!d.querySelector('a:not(.nav-hide)');
    });
  });

  // ---- ToC scrollspy ----
  var tocLinks = document.querySelectorAll('.toc a');
  if (tocLinks.length && 'IntersectionObserver' in window) {
    var byId = {};
    tocLinks.forEach(function (a) { byId[decodeURIComponent(a.hash.slice(1))] = a; });
    var current = null;
    var obs = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) {
        if (!en.isIntersecting) return;
        if (current) current.classList.remove('active');
        current = byId[en.target.id];
        if (current) current.classList.add('active');
      });
    }, { rootMargin: '0px 0px -75% 0px' });
    document.querySelectorAll('article h2[id], article h3[id]').forEach(function (h) {
      if (byId[h.id]) obs.observe(h);
    });
  }

  // ---- copy button on code blocks ----
  document.querySelectorAll('article pre').forEach(function (pre) {
    var btn = document.createElement('button');
    btn.className = 'copy';
    btn.type = 'button';
    btn.textContent = 'Copy';
    btn.addEventListener('click', function () {
      var code = pre.querySelector('code');
      navigator.clipboard.writeText(code ? code.textContent : pre.textContent).then(function () {
        btn.textContent = 'Copied';
        setTimeout(function () { btn.textContent = 'Copy'; }, 1200);
      });
    });
    pre.appendChild(btn);
  });

  // ---- syntax highlighting ----
  // One alternation regex per language; earlier groups win, untouched text stays plain.
  var esc = function (s) {
    return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  };
  var KW = {
    cpp: 'alignas|auto|bool|break|case|catch|char|class|const|constexpr|continue|default|delete|do|double|else|enum|explicit|extern|false|float|for|friend|if|inline|int|int64_t|int32_t|uint32_t|uint64_t|size_t|long|mutable|namespace|new|noexcept|nullptr|operator|override|private|protected|public|return|short|signed|sizeof|static|struct|switch|template|this|throw|true|try|typedef|typename|union|unsigned|using|virtual|void|while',
    python: 'and|as|assert|async|await|break|class|continue|def|del|elif|else|except|False|finally|for|from|global|if|import|in|is|lambda|None|nonlocal|not|or|pass|raise|return|True|try|while|with|yield',
    sh: 'if|then|else|elif|fi|for|do|done|while|case|esac|function|return|exit|export'
  };
  var RULES = {
    cpp:    [['com', /\/\/[^\n]*|\/\*[\s\S]*?\*\//], ['pre', /^\s*#\s*\w+[^\n]*/m], ['str', /"(?:[^"\\\n]|\\.)*"|'(?:[^'\\\n]|\\.)*'/], ['kw', new RegExp('\\b(?:' + KW.cpp + ')\\b')], ['num', /\b0[xX][0-9a-fA-F]+\b|\b\d+(?:\.\d+)?f?\b/]],
    python: [['com', /#[^\n]*/], ['str', /"{3}[\s\S]*?"{3}|'{3}[\s\S]*?'{3}|"(?:[^"\\\n]|\\.)*"|'(?:[^'\\\n]|\\.)*'/], ['kw', new RegExp('\\b(?:' + KW.python + ')\\b')], ['num', /\b\d+(?:\.\d+)?\b/]],
    sh:     [['com', /(?:^|\s)#[^\n]*/], ['str', /"(?:[^"\\\n]|\\.)*"|'[^'\n]*'/], ['pre', /(?<=\s|^)--?[A-Za-z][\w-]*/], ['kw', new RegExp('\\b(?:' + KW.sh + ')\\b')], ['num', /\$\w+|\$\{[^}]*\}/]],
    json:   [['kw', /"(?:[^"\\\n]|\\.)*"(?=\s*:)/], ['str', /"(?:[^"\\\n]|\\.)*"/], ['num', /-?\b\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\b/], ['com', /\btrue\b|\bfalse\b|\bnull\b/]]
  };
  RULES.bash = RULES.shell = RULES.sh;
  RULES.text = null; RULES.glsl = RULES.c = RULES.cpp; RULES.py = RULES.python;
  document.querySelectorAll('article pre code[class^="language-"]').forEach(function (code) {
    var lang = code.className.slice('language-'.length);
    var rules = RULES[lang];
    if (!rules) return;
    var union = new RegExp(rules.map(function (r) { return '(' + r[1].source + ')'; }).join('|'),
      'g' + (rules.some(function (r) { return r[1].flags.indexOf('m') >= 0; }) ? 'm' : ''));
    var src = code.textContent, out = '', last = 0, m;
    while ((m = union.exec(src)) !== null) {
      out += esc(src.slice(last, m.index));
      for (var g = 1; g <= rules.length; g++) {
        if (m[g] !== undefined) { out += '<span class="tok-' + rules[g - 1][0] + '">' + esc(m[g]) + '</span>'; break; }
      }
      last = m.index + m[0].length;
      if (m[0].length === 0) union.lastIndex++;
    }
    out += esc(src.slice(last));
    code.innerHTML = out;
  });
})();
"""


def main():
    import shutil
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "styles.css"), "w") as f:
        f.write(STYLES)
    with open(os.path.join(OUT, "site.js"), "w") as f:
        f.write(SITE_JS)

    # Copy docs/images/ into the site so README/doc image references resolve (pages rewrite
    # the "docs/images/" prefix to "images/" since they live one level down from the repo root).
    src_img = os.path.join(ROOT, "docs", "images")
    if os.path.isdir(src_img):
        shutil.copytree(src_img, os.path.join(OUT, "images"), dirs_exist_ok=True)

    # If doxygen produced an API reference, link it from the sidebar footer.
    api_link = ""
    if os.path.exists(os.path.join(ROOT, "docs", "api", "html", "index.html")):
        api_link = '<a href="../api/html/index.html">API reference (Doxygen)</a><br>'

    # Content-hashed asset URLs: an edited stylesheet/script is re-fetched even when the site is
    # served over HTTP with heuristic caching.
    css_hash = hashlib.md5(STYLES.encode()).hexdigest()[:8]
    js_hash = hashlib.md5(SITE_JS.encode()).hexdigest()[:8]

    written = 0
    global CURRENT_SRC_DIR
    for idx, (src, out, title, section) in enumerate(PAGES):
        path = os.path.join(ROOT, src)
        if not os.path.exists(path):
            print("  skip (missing): %s" % src, file=sys.stderr)
            continue
        CURRENT_SRC_DIR = os.path.dirname(src)
        with open(path) as f:
            text = f.read()
        headings = []
        if src.endswith(".html"):
            body = text
            headings = [(m.group(1), 2, plain(m.group(2)))
                        for m in re.finditer(r'<h2[^>]*\bid="([^"]+)"[^>]*>(.*?)</h2>', text, re.S)]
        else:
            body = convert(text, headings)
        kicker = "" if out == "index.html" else html.escape(section)
        page = PAGE.format(title=html.escape(title), nav=nav_html(out), body=body,
                           kicker=kicker, toc=toc_html(headings), pagenav=pagenav_html(idx),
                           src=html.escape(src), api=api_link, favicon=FAVICON,
                           css_hash=css_hash, js_hash=js_hash,
                           brain_active=" active" if out == "class-explorer.html" else "")
        page = page.replace("docs/images/", "images/")  # repo-root path -> site-relative
        with open(os.path.join(OUT, out), "w") as f:
            f.write(page)
        written += 1
    print("  wrote %d pages -> docs/site/" % written)


if __name__ == "__main__":
    main()
