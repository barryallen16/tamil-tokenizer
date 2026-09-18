#!/usr/bin/env python3
"""Extract Tamil prose from a MediaWiki XML dump, dropping all markup.

Markup left in the training corpus spends merge budget on `thumb`, `align=`,
`[[Category:` and friends. `clean_wikitext` strips the noisy constructs before
anything reaches the trainer.
"""
import bz2
import html
import re
import sys

TAMIL = re.compile(r'[\u0B80-\u0BFF]')
SENTENCE_SPLIT = re.compile(r'(?<=[.!?।])\s+')

# Order matters: nested constructs go before the constructs that contain them.
CLEANERS = [
    (re.compile(r'<ref[^>]*?>.*?</ref\s*>', re.S | re.I), ' '),   # <ref>...</ref>
    (re.compile(r'<ref[^>]*?/>', re.I), ' '),                     # <ref ... />
    (re.compile(r'<!--.*?-->', re.S), ' '),                       # comments
    (re.compile(r'<[^>]+>'), ' '),                                # leftover tags
    (re.compile(r'\{\{[\s\S]*?\}\}'), ' '),                      # {{template}}
    (re.compile(r'\{[\s\S]*?\}'), ' '),                           # leftover {…}
    (re.compile(r'\[\[(?:File|Image|Category|படம்|வகை):[^\]]*\]\]', re.I), ' '),
    (re.compile(r'\[\[[^\]]*\]\]'), ' '),                         # other [[…]]
    (re.compile(r'\[[^\]]*\]'), ' '),                             # external links
    (re.compile(r'https?://\S+'), ' '),                           # bare URLs
    (re.compile(r"'{2,}"), ''),                                   # bold/italic
    (re.compile(r'={2,}[^=]+={2,}'), ' '),                        # headings
    (re.compile(r'&[a-zA-Z]+;|&#\d+;'), ' '),                     # entities
    (re.compile(r'[ \t]+'), ' '),                                 # runs of space
]


def clean_wikitext(text):
    for pattern, repl in CLEANERS:
        text = pattern.sub(repl, text)
    return text.strip()


def is_tamil_prose(text):
    """True when the line reads as prose rather than leftover markup."""
    if len(text) < 30:
        return False
    letters = sum(TAMIL.match(c) is not None for c in text)
    ascii_punct = sum(c in '|_=<>{}[]#*/\\' for c in text)
    # Tamil must dominate, and markup characters must stay rare.
    return letters >= len(text) * 0.4 and ascii_punct <= len(text) * 0.02


def extract(xml_path, output_path, max_articles=0, min_chars=30):
    """Stream the dump, emit one Tamil sentence per line.

    max_articles of 0 means the whole dump.
    """
    articles = 0
    lines = 0

    with bz2.open(xml_path, 'rt', encoding='utf-8', errors='replace') as src, \
         open(output_path, 'w', encoding='utf-8') as out:
        in_page = False
        in_text = False
        buf = []

        def flush():
            nonlocal lines
            if not buf:
                return
            text = clean_wikitext(html.unescape(' '.join(buf)))
            buf.clear()
            for sentence in SENTENCE_SPLIT.split(text):
                sentence = sentence.strip()
                if is_tamil_prose(sentence):
                    out.write(sentence + '\n')
                    lines += 1

        for raw in src:
            line = raw.strip()

            if '<page>' in line:
                in_page = True
                continue

            if '</page>' in line:
                flush()
                in_text = False
                in_page = False
                articles += 1
                if articles % 5000 == 0:
                    print(f"  {articles} articles, {lines} lines...", flush=True)
                if max_articles and articles >= max_articles:
                    break
                continue

            if not in_page:
                continue

            if not in_text:
                if '<text' in line:
                    in_text = True
                    # Body may start on the same line as the <text> tag.
                    _, _, rest = line.partition('>')
                    if rest and '</text>' not in rest:
                        buf.append(rest)
                    elif '</text>' in rest:
                        body, _, _ = rest.partition('</text>')
                        buf.append(body)
                        in_text = False
                        flush()
                continue

            if '</text>' in line:
                body, _, _ = line.partition('</text>')
                if body:
                    buf.append(body)
                in_text = False
                flush()
            else:
                buf.append(line)

    print(f"Done: {articles} articles, {lines} lines -> {output_path}")
    return articles, lines


if __name__ == '__main__':
    xml_path = sys.argv[1] if len(sys.argv) > 1 else 'data/tawiki-latest-pages-articles.xml.bz2'
    output_path = sys.argv[2] if len(sys.argv) > 2 else 'data/tamil_wiki_corpus.txt'
    max_articles = int(sys.argv[3]) if len(sys.argv) > 3 else 0

    print(f"Extracting from {xml_path} (max_articles={max_articles or 'all'})...")
    extract(xml_path, output_path, max_articles)