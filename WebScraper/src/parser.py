from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import zlib
import base64

class Parser:
    def extract_text(self, html: str) -> str:
        soup = BeautifulSoup(html, 'html.parser')
        for script in soup(["script", "style"]):
            script.decompose()
        text = soup.get_text(separator=' ')
        return ' '.join(text.split())

    def extract_links(self, html: str, base_url: str, domain: str) -> list:
        soup = BeautifulSoup(html, 'html.parser')
        links = []
        unwanted_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.pdf', '.zip']
        for a in soup.find_all('a', href=True):
            href = a['href']
            if any(ext in href.lower() for ext in unwanted_extensions):
                continue
            full_url = urljoin(base_url, href)
            if urlparse(full_url).netloc.lower() == domain:
                links.append(full_url)
        return links

    def compress_content(self, text: str) -> str:
        compressed = zlib.compress(text.encode('utf-8'))
        return base64.b64encode(compressed).decode('utf-8')

    def decompress_content(self, compressed: str) -> str:
        decoded = base64.b64decode(compressed)
        decompressed = zlib.decompress(decoded)
        return decompressed.decode('utf-8')

    def get_subject(self, url: str) -> str:
        parsed = urlparse(url)
        path_parts = parsed.path.split('/')
        if 'wiki' in path_parts:
            wiki_index = path_parts.index('wiki')
            if wiki_index + 1 < len(path_parts):
                return path_parts[wiki_index + 1]
        return "Unknown"