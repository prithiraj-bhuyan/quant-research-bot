import os
import json
from pathlib import Path

# Configuration
LIBRARY_DIR = Path("data/raw")
MANIFEST_FILE = Path("manifest.json")

def generate_manifest():
    manifest = []
    
    if not LIBRARY_DIR.exists():
        print(f"Error: Library directory '{LIBRARY_DIR}' not found.")
        return

    print(f"Scanning {LIBRARY_DIR}...")

    # Walk through the library directory
    for cat_folder in sorted(LIBRARY_DIR.iterdir()):
        if not cat_folder.is_dir():
            continue
            
        category = cat_folder.name
        
        for file_path in sorted(cat_folder.glob("*.pdf")):
            filename = file_path.name
            # heuristic for title: remove .pdf and replace underscores
            title = filename.replace(".pdf", "").replace("_", " ")
            
            entry = {
                "filename": filename,
                "category": category,
                "title": title,
                "filepath": str(file_path)
            }
            manifest.append(entry)

    # Save to JSON
    with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"✅ Generated manifest with {len(manifest)} entries.")
    print(f"📄 Saved to: {MANIFEST_FILE}")

if __name__ == "__main__":
    generate_manifest()
