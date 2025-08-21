import asyncio
import json
from crawl4ai import AsyncWebCrawler
from crawl4ai import JsonCssExtractionStrategy
from crawl4ai.async_configs import BrowserConfig, CrawlerRunConfig

async def extract_amazon_products():
    browser_config = BrowserConfig(browser_type="chromium", headless=True)

    crawler_config = CrawlerRunConfig(
        extraction_strategy=JsonCssExtractionStrategy(
            schema={
                "name": "Amazon Product Search Results",
                "baseSelector": "[data-component-type='s-search-result']",
                "fields": [
                    {"name": "title", "selector": "a h2 span", "type": "text"},
                    {"name": "url", "selector": ".puisg-col-inner a", "type": "attribute", "attribute": "href"},
                    {"name": "image", "selector": ".s-image", "type": "attribute", "attribute": "src"},
                    {"name": "rating", "selector": ".a-icon-star-small .a-icon-alt", "type": "text"},
                    {"name": "reviews_count", "selector": "[data-csa-c-func-deps='aui-da-a-popover'] ~ span span", "type": "text"},
                    {"name": "price", "selector": ".a-price .a-offscreen", "type": "text"},
                    {"name": "original_price", "selector": ".a-price.a-text-price .a-offscreen", "type": "text"},
                    {"name": "sponsored", "selector": ".puis-sponsored-label-text", "type": "exists"},
                    {"name": "delivery_info", "selector": "[data-cy='delivery-recipe'] .a-color-base.a-text-bold", "type": "text", "multiple": True},
                ]
            }
        )
    )

    url = "https://www.amazon.com/s?k=Samsung+Galaxy+Tab"

    async with AsyncWebCrawler(config=browser_config) as crawler:
        result = await crawler.arun(url=url, config=crawler_config)

        if result and result.extracted_content:
            products = json.loads(result.extracted_content)

            for product in products:
                print("\nProduct Details:")
                print(f"Title: {product.get('title')}")
                print(f"Price: {product.get('price')}")
                print(f"Original Price: {product.get('original_price')}")
                print(f"Rating: {product.get('rating')}")
                print(f"Reviews: {product.get('reviews_count')}")
                print(f"Sponsored: {'Yes' if product.get('sponsored') else 'No'}")
                if product.get("delivery_info"):
                    print(f"Delivery: {' '.join(product['delivery_info'])}")
                print("-" * 80)

if __name__ == "__main__":
    asyncio.run(extract_amazon_products())