"""
scraper.py — arXiv paper fetcher (refactored from the original script).
"""

import arxiv
import os
import re
import requests

LIBRARY_DIR = os.path.join(os.path.dirname(__file__), "..", "quant_library")

def get_paper_links(max_papers: int):
    client = arxiv.Client()
    search = arxiv.Search(
        query="cat:q-fin.TR OR cat:q-fin.PM OR cat:q-fin.ST",
        max_results=max_papers,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    return list(client.results(search))

def sanitize_filename(title: str) -> str:
    return re.sub(r'[^\w\s-]', '', title).strip().replace(' ', '_')[:50]

def download_and_bucket(results):
    for paper in results:
        cat = paper.primary_category
        folder_path = os.path.join(LIBRARY_DIR, cat)
        os.makedirs(folder_path, exist_ok=True)
        filename = f"{sanitize_filename(paper.title)}.pdf"
        filepath = os.path.join(folder_path, filename)
        if not os.path.exists(filepath):
            print(f"📥 Downloading [{cat}]: {paper.title}")
            response = requests.get(paper.pdf_url)
            with open(filepath, 'wb') as f:
                f.write(response.content)
    print("✅ Download and Bucketing complete.")