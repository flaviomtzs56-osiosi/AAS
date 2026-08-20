import os
from datetime import datetime
import requests
from transformers import pipeline

def fetch_top_tech_news():
    """Fetch top story titles from Hacker News API."""
    print("🌐 Fetching top stories from Hacker News API...")
    url = "https://hacker-news.firebaseio.com/v0/topstories.json"
    top_ids = requests.get(url, timeout=10).json()[:5]

    stories = []
    for story_id in top_ids:
        item_url = f"https://hacker-news.firebaseio.com/v0/item/{story_id}.json"
        data = requests.get(item_url, timeout=10).json()
        if "title" in data:
            stories.append(data["title"])

    return "\n".join(stories)

def main():
    raw_text = fetch_top_tech_news()
    print(f"Fetched Content: \n{raw_text}")

    print("Processing with local AI model...")
    summarizer = pipeline(
        "summarization", model = "sshleifer/distilbart-cnn-12-6"
    )

    summary = summarizer(
        raw_text,
        max_length=45,
        min_length=15,
        do_sample=False,
        clean_up_tokenization_spaces=False,
    )

    output = summary[0]["summary_text"]

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs("reports", exist_ok=True)
    report_file = f"reports/summary_{timestamp}.txt"

    with open(report_file, "w") as f:
        f.write(f"TIMESTAMP: {timestamp}\n")
        f.write("RAW HEADLINES:\n" + raw_text + "\n\n")
        f.write("AI SUMMARY:\n" + output + "\n")

    print(f" Pipeline executed successfully! Output saved to: {report_file}")


if __name__ == "__main__":
    main()