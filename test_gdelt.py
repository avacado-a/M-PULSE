from gdeltdoc import GdeltDoc, Filters
import traceback
import time

def test_chunk(start, end, keyword="robotics"):
    print(f"Testing GDELT Chunk: {start} to {end}...")
    f = Filters(
        keyword=keyword,
        start_date=start,
        end_date=end
    )
    gd = GdeltDoc()
    
    try:
        articles = gd.article_search(f)
        if articles.empty:
            print("  Result: Success (but 0 articles found)")
        else:
            print(f"  Result: Success ({len(articles)} articles found)")
    except Exception as e:
        print(f"  Result: FAILED")
        print(f"  Error Type: {type(e).__name__}")
        print(f"  Error Message: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    # Testing the specific chunks that failed earlier
    failed_chunks = [
        ("2024-01-01", "2024-03-31"),
        ("2024-03-31", "2024-06-29"),
        ("2024-12-26", "2025-03-26")
    ]
    
    for start, end in failed_chunks:
        test_chunk(start, end)
        print("-" * 30)
        time.sleep(6) # Being extra careful with rate limits during testing
