import pytest

from dtx.core import webpage


def test_webpage_parser_extracts_clean_text_and_labeled_table_rows():
    markup = """<!doctype html><html lang="en"><head><title>Annual Report</title>
      <meta name="description" content="A report about yearly results."></head><body>
      <nav>Navigation noise with a lot of words that should not appear.</nav>
      <main><h1>Annual results</h1><p>Revenue increased during the reporting period.</p>
      <p>Revenue increased during the reporting period.<sup class="reference">[1]</sup></p>
      <table><caption>Regional totals</caption><tr><th>Region</th><th>Revenue</th></tr>
      <tr><td>North</td><td>$120</td></tr></table>
      <script>trackingCode()</script><footer>Footer content that should be removed.</footer>
      </main></body></html>"""

    page = webpage.extract_html(markup, "https://example.org/report")

    assert page["title"] == "Annual Report"
    assert page["language"] == "en"
    assert len(page["records"]) == 3
    combined = " ".join(record["text"] for record in page["records"])
    assert "Navigation noise" not in combined
    assert "trackingCode" not in combined
    assert "Footer content" not in combined
    assert "[1]" not in combined
    table_row = page["records"][-1]
    assert table_row["block_type"] == "table_row"
    assert table_row["section"] == "Regional totals"
    assert table_row["table_headers"] == "Region | Revenue"
    assert table_row["text"] == "Region: North | Revenue: $120"


def test_webpage_fetch_rejects_non_https_and_private_network(monkeypatch):
    with pytest.raises(ValueError, match="HTTPS"):
        webpage.validate_public_url("http://example.org")
    with pytest.raises(ValueError, match="private"):
        webpage.validate_public_url("https://127.0.0.1/admin")
    with pytest.raises(ValueError, match="private"):
        webpage.validate_public_url("https://localhost/admin")

    monkeypatch.setattr(webpage.socket, "getaddrinfo", lambda *args, **kwargs: [
        (2, 1, 6, "", ("93.184.216.34", 443)),
        (2, 1, 6, "", ("10.0.0.4", 443)),
    ])
    with pytest.raises(ValueError, match="private"):
        webpage.validate_public_url("https://mixed-address.example")


def test_scrape_webpage_reads_public_html(monkeypatch):
    markup = b"<html><title>Public facts</title><main><p>Useful extracted paragraph.</p></main></html>"

    class Headers:
        def get_content_type(self):
            return "text/html"

        def get_content_charset(self):
            return "utf-8"

    class Response:
        headers = Headers()

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

        def read(self, limit):
            return markup

        def geturl(self):
            return "https://example.org/facts"

    class Opener:
        def open(self, request, timeout):
            assert request.full_url == "https://example.org/facts"
            assert timeout > 0
            return Response()

    monkeypatch.setattr(webpage, "validate_public_url", lambda _: ["93.184.216.34"])
    monkeypatch.setattr(webpage, "build_opener", lambda *args: Opener())

    page = webpage.scrape_webpage("https://example.org/facts")
    assert page["title"] == "Public facts"
    assert page["records"][0]["text"] == "Useful extracted paragraph."