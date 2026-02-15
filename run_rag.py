import sys
import os

# Add the 'app' directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "app"))

from app.rag import ask
import argparse

def main():
    parser = argparse.ArgumentParser(description="Run the Quant Research Bot RAG pipeline.")
    parser.add_argument("query", type=str, help="The question to ask the bot.")
    args = parser.parse_args()

    print(f"\n🔍 Query: {args.query}")
    print("Thinking...\n")

    try:
        response = ask(args.query)
        
        print("🤖 Answer:")
        print("-" * 60)
        print(response.answer)
        print("-" * 60)
        
        print("\n📚 Citations:")
        for idx, cit in enumerate(response.citations):
            print(f"[{idx+1}] {cit.paper_title} (Score: {cit.relevance_score:.3f})")
            print(f"    Page: {cit.page_number} | Section: {cit.section}")
        
        print("\n✅ Logged to rag_logs.jsonl")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
