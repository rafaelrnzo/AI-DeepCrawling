import asyncio
import os
import requests
import json
from dotenv import load_dotenv
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
from bs4 import BeautifulSoup
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not found")

GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

app = FastAPI(title="Crawl + Gemini Summarizer")


def clean_html(html_content: str) -> str:
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


async def crawl_and_analyze(url: str):
    config = CrawlerRunConfig(
        deep_crawl_strategy=BFSDeepCrawlStrategy(
            max_depth=2,
            include_external=False,
            max_pages=5
        ),
        verbose=True
    )

    async with AsyncWebCrawler() as crawler:
        results = await crawler.arun(url=url, config=config)

        all_summaries = []

        for i, page in enumerate(results, 1):
            if hasattr(page, 'html') and page.html:
                clean_content = clean_html(page.html)
                summary = gemini_request(
                    f"Summarize this web page from {page.url}: {clean_content}"
                )
                all_summaries.append({
                    "url": page.url,
                    "summary": summary
                })

        # gabungkan semua summary jadi 1
        combined_text = "\n\n".join([s["summary"] for s in all_summaries])
        final_summary = gemini_request(
            f"Create a concise overall summary of these page summaries:\n\n{combined_text}"
        )

        return {
            "pages": all_summaries,
            "final_summary": final_summary
        }


@app.get("/crawl")
async def crawl(url: str = Query(..., description="Target URL untuk crawling")):
    try:
        result = await crawl_and_analyze(url)
        return JSONResponse(content=result, status_code=200)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
