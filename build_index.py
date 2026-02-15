import sys
import os

# Ensure the app module can be found
sys.path.append(os.path.join(os.path.dirname(__file__), "app"))

try:
    from app.embeddings import build_full_index
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("Run: pip install pymupdf sentence-transformers faiss-cpu rank-bm25 python-dotenv openai arxiv")
    sys.exit(1)

if __name__ == "__main__":
    print("🚀 Starting Index Build Process...")
    print("This may take a few minutes depending on the number of PDFs.")
    
    try:
        result = build_full_index()
        print("\n✅ Build Complete!")
        print(f"Status: {result.get('status')}")
        print(f"Papers Processed: {result.get('papers')}")
        print(f"Chunks Created: {result.get('chunks')}")
    except Exception as e:
        print(f"\n❌ Build Failed: {e}")
