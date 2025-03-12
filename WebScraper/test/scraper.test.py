import unittest

from src.parser import Parser


class TestParser(unittest.TestCase):
    def test_extract_text(self):
        html = "<html><body><p>Hello</p><script>var x=1;</script></body></html>"
        parser = Parser()
        text = parser.extract_text(html)
        self.assertEqual(text, "Hello")

    def test_compress_decompress(self):
        text = "This is a test."
        parser = Parser()
        compressed = parser.compress_content(text)
        decompressed = parser.decompress_content(compressed)
        self.assertEqual(text, decompressed)


if __name__ == '__main__':
    unittest.main()
