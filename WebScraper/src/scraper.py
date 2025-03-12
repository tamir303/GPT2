import asyncio
import aiohttp
from urllib.parse import urlparse

from src.config import Config
from src.parser import Parser
from src.writer import Writer
import logging


class Scraper:
    def __init__(self, config: Config, parser: Parser, writer: Writer, logger: logging.Logger):
        self.config = config
        self.parser = parser
        self.writer = writer
        self.logger = logger
        self.domain = urlparse(config.start_url).netloc.lower()
        self.visited = set()
        self.queue = asyncio.Queue()
        self.page_count = 0
        self.stop_event = asyncio.Event()
        self.lock = asyncio.Lock()

    async def fetch_html(self, url: str, session: aiohttp.ClientSession) -> tuple:
        try:
            async with session.get(url, ssl=self.config.verify_ssl) as response:
                if response.status == 200 and 'text/html' in response.headers.get('Content-Type', ''):
                    html = await response.text()
                    return html, str(response.url)
                else:
                    self.logger.warning(f"Skipping {url}: Status {response.status} or non-HTML")
                    return None, None
        except Exception as e:
            self.logger.error(f"Error fetching {url}: {e}")
            return None, None

    async def scrape_page(self, url: str, depth: int, session: aiohttp.ClientSession):
        if url in self.visited or (self.config.max_depth != -1 and depth > self.config.max_depth):
            return
        self.visited.add(url)
        html, final_url = await self.fetch_html(url, session)
        if html and urlparse(final_url).netloc.lower() == self.domain:
            self.logger.info(f"Scraping: {final_url} at depth {depth}")
            text = self.parser.extract_text(html)
            subject = self.parser.get_subject(final_url)
            compressed_content = self.parser.compress_content(text)
            self.writer.write_row(subject, compressed_content)
            async with self.lock:
                self.page_count += 1
                if self.config.max_pages != -1 and self.page_count >= self.config.max_pages:
                    self.stop_event.set()
            if not self.stop_event.is_set() and (self.config.max_depth == -1 or depth < self.config.max_depth):
                links = self.parser.extract_links(html, final_url, self.domain)
                for link in links:
                    if link not in self.visited:
                        await self.queue.put((link, depth + 1))

    async def worker(self, session: aiohttp.ClientSession):
        while not self.stop_event.is_set():
            try:
                item = await asyncio.wait_for(self.queue.get(), timeout=1.0)
                url, depth = item
                await self.scrape_page(url, depth, session)
                self.queue.task_done()
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

    async def run(self):
        async with aiohttp.ClientSession() as session:
            workers = [asyncio.create_task(self.worker(session)) for _ in range(self.config.max_workers)]
            await self.queue.put((self.config.start_url, 0))
            await self.queue.join()
            self.stop_event.set()
            for worker in workers:
                worker.cancel()
            await asyncio.gather(*workers, return_exceptions=True)