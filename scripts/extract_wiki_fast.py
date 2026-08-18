#!/usr/bin/env python3
"""Fast extraction: stream bz2, pull Tamil text lines from XML without full parsing."""
import bz2
import re
import sys

def fast_extract(xml_path, output_path, max_articles=5000):
    """Stream through bz2 XML and extract Tamil text between <text> tags."""
    article_count = 0
    line_count = 0
    
    tamil_re = re.compile(r'[\u0B80-\u0BFF]')
    tag_re = re.compile(r'<text[^>]*>(.*?)</text>', re.DOTALL)
    
    with bz2.open(xml_path, 'rt', encoding='utf-8', errors='replace') as f:
        buf = []
        in_text = False
        
        for line in f:
            if '<text' in line and 'xml:space="preserve"' in line:
                in_text = True
                # Check if </text> is on same line
                m = re.search(r'<text[^>]*>(.*?)</text>', line)
                if m:
                    text = m.group(1)
                    # Quick clean
                    text = re.sub(r'<[^>]+>', ' ', text)
                    text = re.sub(r'\[\[[^\]]*\]\]', ' ', text)
                    text = re.sub(r'\{\{[^}]*\}\}', '', text)
                    text = re.sub(r'\s+', ' ', text).strip()
                    if text and tamil_re.search(text):
                        # Split into sentences/clauses
                        for chunk in re.split(r'(?<=[.!?।\n])\s+', text):
                            chunk = chunk.strip()
                            if len(chunk) > 20 and tamil_re.search(chunk):
                                buf.append(chunk)
                                line_count += 1
                    in_text = False
                    article_count += 1
                continue
            
            if in_text:
                if '</text>' in line:
                    # End of text block
                    text = ' '.join(buf)
                    text = re.sub(r'<[^>]+>', ' ', text)
                    text = re.sub(r'\[\[[^\]]*\]\]', ' ', text)
                    text = re.sub(r'\{\{[^}]*\}\}', '', text)
                    text = re.sub(r'\s+', ' ', text).strip()
                    if text and tamil_re.search(text):
                        for chunk in re.split(r'(?<=[.!?।\n])\s+', text):
                            chunk = chunk.strip()
                            if len(chunk) > 20 and tamil_re.search(chunk):
                                line_count += 1
                    buf = []
                    in_text = False
                    article_count += 1
                else:
                    buf.append(line.strip())
            
            if article_count % 1000 == 0 and article_count > 0:
                print(f"  {article_count} articles, {line_count} lines...", flush=True)
            
            if article_count >= max_articles:
                break
    
    # Write collected lines
    with open(output_path, 'w', encoding='utf-8') as out:
        # Re-read from a second pass would be complex, so write during extraction
        pass
    
    print(f"Done: {article_count} articles, {line_count} lines")


def fast_extract_v2(xml_path, output_path, max_articles=5000):
    """Simpler: just pull Tamil-heavy lines from the bz2 stream."""
    article_count = 0
    line_count = 0
    tamil_re = re.compile(r'[\u0B80-\u0BFF]')
    # Match lines that contain Tamil characters and are reasonably long
    
    with bz2.open(xml_path, 'rt', encoding='utf-8', errors='replace') as f, \
         open(output_path, 'w', encoding='utf-8') as out:
        
        in_page = False
        in_text = False
        text_buf = []
        
        for line in f:
            stripped = line.strip()
            
            if '<page>' in stripped:
                in_page = True
                continue
            if '</page>' in stripped:
                # Process accumulated text
                if text_buf:
                    full_text = ' '.join(text_buf)
                    # Clean wiki markup
                    full_text = re.sub(r'<[^>]+>', ' ', full_text)
                    full_text = re.sub(r'\[\[[^\]]*\]\]', ' ', full_text)
                    full_text = re.sub(r'\{\{[^}]*\}\}', '', full_text)
                    full_text = re.sub(r'\[https?://[^\]]*\]', '', full_text)
                    full_text = re.sub(r'&\w+;', ' ', full_text)
                    full_text = re.sub(r"'{2,}", '', full_text)
                    full_text = re.sub(r'={2,}[^=]+={2,}', '', full_text)
                    full_text = re.sub(r'\s+', ' ', full_text).strip()
                    
                    # Split into sentence-like chunks
                    for chunk in re.split(r'(?<=[.!?।])\s+', full_text):
                        chunk = chunk.strip()
                        if len(chunk) > 30 and tamil_re.search(chunk):
                            out.write(chunk + '\n')
                            line_count += 1
                    
                    text_buf = []
                in_page = False
                article_count += 1
                
                if article_count % 500 == 0:
                    print(f"  {article_count} articles, {line_count} lines...", flush=True)
                if article_count >= max_articles:
                    break
                continue
            
            if in_page and '<text' in stripped:
                in_text = True
                # Extract text from same line
                m = re.search(r'>(.*)', stripped)
                if m:
                    text_buf.append(m.group(1))
                continue
            
            if in_text:
                if '</text>' in stripped:
                    m = re.search(r'(.*?)</text>', stripped)
                    if m:
                        text_buf.append(m.group(1))
                    in_text = False
                else:
                    text_buf.append(stripped)
    
    print(f"\nDone: {article_count} articles, {line_count} lines -> {output_path}")
    return article_count, line_count


if __name__ == '__main__':
    xml_path = sys.argv[1] if len(sys.argv) > 1 else 'data/tawiki-latest-pages-articles.xml.bz2'
    output_path = sys.argv[2] if len(sys.argv) > 2 else 'data/tamil_wiki_corpus.txt'
    max_articles = int(sys.argv[3]) if len(sys.argv) > 3 else 5000
    
    print(f"Fast extracting from {xml_path} (max {max_articles} articles)...")
    fast_extract_v2(xml_path, output_path, max_articles)
