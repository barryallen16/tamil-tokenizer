#!/usr/bin/env python3
"""Extract plain text from Wikipedia XML dump."""
import bz2
import re
import sys
import xml.etree.ElementTree as ET

def extract_text(xml_path, output_path, max_articles=None):
    """Extract article text from a Wikipedia XML dump."""
    ns = {'mw': 'http://www.mediawiki.org/xml/export-0.10/'}
    
    article_count = 0
    total_lines = 0
    
    with bz2.open(xml_path, 'rt', encoding='utf-8', errors='replace') as f:
        # Parse incrementally
        context = ET.iterparse(f, events=('end',))
        
        with open(output_path, 'w', encoding='utf-8') as out:
            for event, elem in context:
                if elem.tag == '{http://www.mediawiki.org/xml/export-0.10/}page':
                    # Get title
                    title_elem = elem.find('.//mw:title', ns)
                    title = title_elem.text if title_elem is not None else ''
                    
                    # Skip non-article pages
                    ns_elem = elem.find('.//mw:ns', ns)
                    if ns_elem is not None and ns_elem.text != '0':
                        elem.clear()
                        continue
                    
                    # Get text content
                    text_elem = elem.find('.//mw:text', ns)
                    if text_elem is not None and text_elem.text:
                        raw_text = text_elem.text
                        
                        # Remove wiki markup
                        text = remove_wiki_markup(raw_text)
                        
                        # Write cleaned text
                        for line in text.split('\n'):
                            line = line.strip()
                            if len(line) > 10:  # Skip very short lines
                                out.write(line + '\n')
                                total_lines += 1
                    
                    article_count += 1
                    if article_count % 1000 == 0:
                        print(f"  Processed {article_count} articles, {total_lines} lines...")
                    
                    if max_articles and article_count >= max_articles:
                        break
                    
                    elem.clear()
    
    print(f"\nDone: {article_count} articles, {total_lines} lines written to {output_path}")
    return article_count, total_lines


def remove_wiki_markup(text):
    """Remove common MediaWiki markup."""
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # Remove wiki links [[...]]
    text = re.sub(r'\[\[([^\]|]*\|)?([^\]]*)\]\]', r'\2', text)
    
    # Remove external links [http://... text]
    text = re.sub(r'\[https?://[^\s]+ ([^\]]*)\]', r'\1', text)
    text = re.sub(r'\[https?://[^\]]+\]', '', text)
    
    # Remove templates {{...}}
    text = re.sub(r'\{\{[^}]*\}\}', '', text)
    
    # Remove file/image links
    text = re.sub(r'\[\[File:[^\]]*\]\]', '', text)
    text = re.sub(r'\[\[Image:[^\]]*\]\]', '', text)
    
    # Remove headings
    text = re.sub(r'^={1,6}\s*$', '', text, flags=re.MULTILINE)
    
    # Remove references
    text = re.sub(r'<ref[^>]*>.*?</ref>', '', text, flags=re.DOTALL)
    text = re.sub(r'<ref[^/]*/>', '', text)
    
    # Remove IPA, pronunciation
    text = re.sub(r'\{\{IPA\|[^}]*\}\}', '', text)
    
    # Remove categories
    text = re.sub(r'\[\[Category:[^\]]*\]\]', '', text)
    
    # Collapse whitespace
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\n\s*\n', '\n', text)
    
    return text.strip()


if __name__ == '__main__':
    xml_path = sys.argv[1] if len(sys.argv) > 1 else 'data/tawiki-latest-pages-articles.xml.bz2'
    output_path = sys.argv[2] if len(sys.argv) > 2 else 'data/tamil_wiki_corpus.txt'
    max_articles = int(sys.argv[3]) if len(sys.argv) > 3 else None
    
    print(f"Extracting text from {xml_path}...")
    extract_text(xml_path, output_path, max_articles)
