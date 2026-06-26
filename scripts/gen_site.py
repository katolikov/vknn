#!/usr/bin/env python3
"""Build the VKNN documentation site: render the project Markdown into a small, clean, self-contained
static HTML site under docs/site/ (one obvious entry point, docs/site/index.html).

Dependency-free on purpose (matches the engine): a compact GitHub-flavored-Markdown -> HTML converter
plus one embedded stylesheet. Run via `./build.sh --docs`.
"""
import html
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "site")

# (source markdown, output html, nav title, nav section). First entry is the home page.
PAGES = [
    ("README.md",                         "index.html",               "Overview",            "Get started"),
    ("AGENTS.md",                          "agents.html",              "Contributor guide",   "Get started"),
    ("skills/README.md",                   "howto.html",               "How-to guides",       "Get started"),
    ("skills/add-an-operator.md",          "howto-add-operator.html",  "Add an operator",     "How-to"),
    ("skills/add-a-backend.md",            "howto-add-backend.html",   "Add a backend",       "How-to"),
    ("skills/compile-and-run-a-model.md",  "howto-compile-run.html",   "Compile & run",       "How-to"),
    ("skills/run-yonosplat.md",            "howto-yonosplat.html",     "Run YoNoSplat",       "How-to"),
    ("docs/ARCHITECTURE.md",               "architecture.html",        "Architecture",        "Reference"),
    ("docs/OP_COVERAGE.md",                "op-coverage.html",         "Op coverage",         "Reference"),
    ("docs/CONFIG.md",                     "config.html",              "Config",              "Reference"),
    ("docs/ADDING_AN_OPERATOR.md",         "adding-an-operator.html",  "Adding an operator",  "Reference"),
    ("docs/ADDING_A_BACKEND.md",           "adding-a-backend.html",    "Adding a backend",    "Reference"),
    ("docs/BENCHMARK.md",                  "benchmark.html",           "Benchmarks",          "Reference"),
    ("docs/LIMITATIONS.md",                "limitations.html",         "Limitations",         "Reference"),
    ("docs/MNN_ANALYSIS.md",               "mnn-analysis.html",        "MNN analysis",        "Reference"),
]
# Architecture Decision Records, added programmatically (kept in nav under "Design").
for fn in sorted(os.listdir(os.path.join(ROOT, "docs", "adr"))):
    if fn.endswith(".md"):
        num, rest = fn[:-3].split("-", 1)
        title = "ADR " + num + " · " + rest.replace("-", " ")
        PAGES.append(("docs/adr/" + fn, "adr-" + num + ".html", title, "Design"))

# Map every plausible link spelling of a source doc to its output page, for intra-site links.
LINKMAP = {}
for src, out, _, _ in PAGES:
    base = os.path.basename(src)
    for key in (src, base, "./" + base, "../" + src, src.replace("docs/", "../docs/")):
        LINKMAP[key] = out
LINKMAP["docs/adr/"] = "adr-0001.html"
LINKMAP["docs/adr"] = "adr-0001.html"


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


def rewrite_link(url):
    anchor = ""
    if "#" in url:
        url, anchor = url.split("#", 1)
        anchor = "#" + anchor
    url = re.sub(r":\d+$", "", url)  # drop a :line suffix
    if url in LINKMAP:
        return LINKMAP[url] + anchor
    if anchor and url == "":
        return anchor  # same-page anchor
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
    return re.sub(r"[^a-z0-9]+", "-", re.sub(r"<[^>]+>", "", text).lower()).strip("-")


# --------------------------------------------------------------------------- block markdown
def convert(md):
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

        # heading
        m = re.match(r"^(#{1,6})\s+(.*?)\s*#*\s*$", line)
        if m:
            level = len(m.group(1))
            content = inline(m.group(2))
            sid = slug(m.group(2))
            out.append("<h%d id=\"%s\">%s</h%d>" % (level, sid, content, level))
            i += 1
            continue

        # horizontal rule
        if re.match(r"^\s*([-*_])\s*(\1\s*){2,}$", line):
            out.append("<hr>")
            i += 1
            continue

        # table (header row + |---| separator)
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
            t = ["<table>", "<thead><tr>"] + ["<th>%s</th>" % inline(c) for c in head] + ["</tr></thead>", "<tbody>"]
            for r in body:
                t.append("<tr>" + "".join("<td>%s</td>" % inline(c) for c in r) + "</tr>")
            t.append("</tbody></table>")
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
    sections = []
    seen = []
    for _, _, _, sec in PAGES:
        if sec not in seen:
            seen.append(sec)
    for sec in seen:
        links = []
        for _, out, title, s in PAGES:
            if s != sec:
                continue
            cls = ' class="active"' if out == active else ""
            links.append('<a href="%s"%s>%s</a>' % (out, cls, html.escape(title)))
        sections.append('<div class="nav-section"><div class="nav-title">%s</div>%s</div>' % (html.escape(sec), "".join(links)))
    return "\n".join(sections)


PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} · VKNN</title>
<link rel="stylesheet" href="styles.css">
</head>
<body>
<input type="checkbox" id="nav-toggle" hidden>
<label for="nav-toggle" class="nav-burger" aria-label="Toggle navigation">☰</label>
<aside class="sidebar">
  <a class="brand" href="index.html"><span class="logo">VKNN</span><span class="brand-sub">Vulkan Neural Network</span></a>
  <nav>{nav}</nav>
  <div class="nav-foot">{api}On-device inference engine</div>
</aside>
<main>
<article class="content">
{body}
</article>
</main>
</body>
</html>
"""

STYLES = """:root{
  --bg:#ffffff; --fg:#1f2933; --muted:#6b7280; --line:#e5e7eb;
  --sidebar:#0f172a; --sidebar-fg:#cbd5e1; --sidebar-active:#38bdf8;
  --accent:#2563eb; --code-bg:#0f172a; --code-fg:#e2e8f0; --inline:#eef2f7; --inline-fg:#0f172a;
  --quote-bg:#f8fafc;
}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{margin:0;font:16px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  color:var(--fg);background:var(--bg);display:flex}
a{color:var(--accent);text-decoration:none}
a:hover{text-decoration:underline}

.sidebar{position:fixed;top:0;left:0;width:270px;height:100vh;overflow-y:auto;background:var(--sidebar);
  color:var(--sidebar-fg);padding:22px 0;display:flex;flex-direction:column}
.brand{display:block;padding:6px 22px 18px;border-bottom:1px solid #1e293b;margin-bottom:10px}
.brand:hover{text-decoration:none}
.logo{display:block;font-weight:800;font-size:24px;letter-spacing:.5px;color:#fff}
.brand-sub{display:block;font-size:12px;color:#64748b;margin-top:2px}
.nav-section{margin:14px 0}
.nav-title{font-size:11px;text-transform:uppercase;letter-spacing:.08em;color:#64748b;padding:4px 22px;font-weight:700}
.sidebar nav a{display:block;padding:6px 22px;color:var(--sidebar-fg);font-size:14.5px;border-left:3px solid transparent}
.sidebar nav a:hover{background:#1e293b;color:#fff;text-decoration:none}
.sidebar nav a.active{color:#fff;border-left-color:var(--sidebar-active);background:#1e293b;font-weight:600}
.nav-foot{margin-top:auto;padding:16px 22px 4px;font-size:12px;color:#475569;border-top:1px solid #1e293b}
.nav-foot a{color:var(--sidebar-active)}

main{margin-left:270px;flex:1;min-width:0}
.content{max-width:820px;margin:0 auto;padding:54px 40px 120px}

h1,h2,h3,h4{line-height:1.25;font-weight:700;color:#0f172a;margin:1.8em 0 .6em}
h1{font-size:2.05rem;margin-top:.2em;padding-bottom:.35em;border-bottom:2px solid var(--line)}
h2{font-size:1.5rem;margin-top:1.9em;padding-bottom:.25em;border-bottom:1px solid var(--line)}
h3{font-size:1.2rem}
h4{font-size:1.02rem}
p{margin:.7em 0}
hr{border:0;border-top:1px solid var(--line);margin:2.2em 0}
ul,ol{padding-left:1.5em;margin:.6em 0}
li{margin:.28em 0}
li>ul,li>ol{margin:.25em 0}
strong{color:#0f172a}

code{font-family:"SF Mono",ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:.875em;
  background:var(--inline);color:var(--inline-fg);padding:.12em .4em;border-radius:5px}
pre{background:var(--code-bg);color:var(--code-fg);padding:16px 18px;border-radius:10px;overflow-x:auto;
  margin:1em 0;font-size:13.5px;line-height:1.55}
pre code{background:none;color:inherit;padding:0;font-size:13.5px;border-radius:0}

blockquote{margin:1em 0;padding:.6em 1.1em;background:var(--quote-bg);border-left:4px solid var(--accent);
  border-radius:0 8px 8px 0;color:#334155}
blockquote p{margin:.3em 0}

table{border-collapse:collapse;width:100%;margin:1.1em 0;font-size:14.5px;display:block;overflow-x:auto}
th,td{border:1px solid var(--line);padding:8px 12px;text-align:left;vertical-align:top}
th{background:#f1f5f9;font-weight:700;color:#0f172a}
tr:nth-child(even) td{background:#fafbfc}

.nav-burger{display:none}
@media(max-width:900px){
  .sidebar{transform:translateX(-100%);transition:transform .2s;z-index:20}
  #nav-toggle:checked ~ .sidebar{transform:none}
  main{margin-left:0}
  .content{padding:64px 22px 90px}
  .nav-burger{display:flex;position:fixed;top:12px;left:12px;z-index:30;width:42px;height:42px;
    align-items:center;justify-content:center;background:var(--sidebar);color:#fff;border-radius:8px;
    cursor:pointer;font-size:20px}
}
"""


def main():
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "styles.css"), "w") as f:
        f.write(STYLES)

    # If doxygen produced an API reference, link it from the sidebar footer.
    api_link = ""
    if os.path.exists(os.path.join(ROOT, "docs", "api", "html", "index.html")):
        api_link = '<a href="../api/html/index.html">API reference (Doxygen)</a><br>'

    for src, out, title, _ in PAGES:
        path = os.path.join(ROOT, src)
        if not os.path.exists(path):
            print("  skip (missing): %s" % src, file=sys.stderr)
            continue
        with open(path) as f:
            md = f.read()
        body = convert(md)
        page = PAGE.format(title=html.escape(title), nav=nav_html(out), body=body, api=api_link)
        with open(os.path.join(OUT, out), "w") as f:
            f.write(page)
    print("  wrote %d pages -> docs/site/" % len(PAGES))


if __name__ == "__main__":
    main()
