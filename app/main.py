import asyncio
import os
import requests
import json
from dotenv import load_dotenv
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
from bs4 import BeautifulSoup

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not found")

GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"


def clean_html(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    for script in soup(["script", "style"]):
        script.extract()
    text = soup.get_text()
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    return ' '.join(chunk for chunk in chunks if chunk)[:5000]


def gemini_request(prompt: str) -> str:
    payload = {
        "contents": [{
            "parts": [{"text": prompt}]
        }]
    }
    try:
        response = requests.post(
            GEMINI_URL,
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=60
        )
        if response.status_code == 200:
            return response.json()["candidates"][0]["content"]["parts"][0]["text"]
        return f"Error: {response.status_code}, {response.text}"
    except Exception as e:
        return f"Error: {str(e)}"


async def crawl_and_analyze():
    config = CrawlerRunConfig(
        deep_crawl_strategy=BFSDeepCrawlStrategy(
            max_depth=2,
            include_external=False,
            max_pages=5
        ),
        verbose=True
    )

    async with AsyncWebCrawler() as crawler:
        results = await crawler.arun(
            url="https://www.olx.co.id/jakarta-selatan_g4000030/q-supra",
            config=config
        )

        all_summaries = []

        for i, page in enumerate(results, 1):
            print(f"Page {i}: {page.url}")

            if hasattr(page, 'html') and page.html:
                clean_content = clean_html(page.html)
                summary = gemini_request(
                    f"Summarize this web page from {page.url}: {clean_content}"
                )

                print(f"Summary: {summary}\n")
                all_summaries.append(summary)

            else:
                print("No content found\n")

        with open("demofile.txt", "w", encoding="utf-8") as f:
            f.write("\n\n".join(all_summaries))

        with open("demofile.txt", "r", encoding="utf-8") as f:
            content = f.read()

        final_summary = gemini_request(
            f"Create a concise overall summary of these page summaries:\n\n{content}"
        )

        print("\n=== FINAL SUMMARY ===\n")
        print(final_summary)

        with open("demofile_summary.txt", "w", encoding="utf-8") as f:
            f.write(final_summary)


if __name__ == "__main__":
    asyncio.run(crawl_and_analyze())
