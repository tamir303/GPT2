import asyncio

from src.config import Config
from src.logger import Logger
from src.parser import Parser
from src.scraper import Scraper
from src.writer import Writer


async def main():
    config = Config(
        start_url="https://en.wikipedia.org/wiki/Main_Page",
        output_file="scraped_wiki.csv",
        verify_ssl=False,
        max_workers=10,
        max_depth=2,
        max_pages=10000
    )
    logger = Logger("WebScraper").get_logger()
    parser = Parser()
    writer = Writer(config.output_file)
    scraper = Scraper(config, parser, writer, logger)
    await scraper.run()

if __name__ == "__main__":
    asyncio.run(main())