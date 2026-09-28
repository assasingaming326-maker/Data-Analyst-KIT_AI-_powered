"""Public webpage fetcher and readable-content extractor for link-based imports."""
from __future__ import annotations

import http.client
import ipaddress
import re
import socket
import ssl
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlsplit
from urllib.request import HTTPRedirectHandler, HTTPSHandler, ProxyHandler, Request, build_opener

from bs4 import BeautifulSoup

MAX_PAGE_BYTES = 10 * 1024 * 1024
_CONTENT_TAGS = ("h1", "h2", "h3", "h4", "p", "li", "blockquote")
_REMOVE_TAGS = ("script", "style", "noscript", "nav", "footer", "form", "iframe", "svg", "canvas")


def _is_public_ip(value: str) -> bool:
    try:
        return ipaddress.ip_address(value.split("%", 1)[0]).is_global
    except ValueError:
        return False


def _public_addresses(host: str) -> list[str]:
    try:
        address = ipaddress.ip_address(host)
        addresses = [str(address)] if address.is_global else []
    except ValueError:
        try:
            addresses = list({item[4][0] for item in socket.getaddrinfo(
                host, 443, type=socket.SOCK_STREAM
            )})
        except OSError as exc:
            raise ValueError("Webpage hostname could not be resolved") from exc
    if not addresses or any(not _is_public_ip(address) for address in addresses):
        raise ValueError("Webpage links to private or non-public network addresses are blocked")
    return addresses


def validate_public_url(url: str) -> list[str]:
    parsed = urlsplit(url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("Enter a public HTTPS webpage link without embedded credentials")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("Webpage URL has an invalid port") from exc
    if port not in (None, 443):
        raise ValueError("Webpage links must use the standard HTTPS port")
    return _public_addresses(parsed.hostname.rstrip("."))


class _PublicRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, new_url):
        target = urljoin(request.full_url, new_url)
        validate_public_url(target)
        return super().redirect_request(request, response, code, message, headers, target)


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    def __init__(self, host, *args, connect_address: str, **kwargs):
        self._connect_address = connect_address
        super().__init__(host, *args, **kwargs)

    def connect(self):
        sock = socket.create_connection((self._connect_address, self.port), self.timeout,
                                        self.source_address)
        self.sock = self._context.wrap_socket(sock, server_hostname=self.host)


class _PublicHTTPSHandler(HTTPSHandler):
    def https_open(self, request):
        address = validate_public_url(request.full_url)[0]

        def pinned_connection(host, *args, **kwargs):
            return _PinnedHTTPSConnection(host, *args, connect_address=address, **kwargs)

        return self.do_open(pinned_connection, request, context=self._context)


def extract_html(markup: str, page_url: str) -> dict:
    soup = BeautifulSoup(markup, "html.parser")
    title_node = soup.find("title")
    title = title_node.get_text(" ", strip=True) if title_node else ""
    language = (soup.html.get("lang", "") if soup.html else "").strip()
    for node in soup(_REMOVE_TAGS):
        node.decompose()
    for node in soup.select(
        "[hidden], [aria-hidden='true'], .sidebar, .navbox, .metadata, .toc, "
        ".mw-editsection, .reference, .reflist, .mw-references-wrap, sup.reference"
    ):
        node.decompose()

    content = next((soup.select_one(selector) for selector in (
        "#mw-content-text", "#bodyContent", ".mw-parser-output", "main", "article", "[role=main]"
    ) if soup.select_one(selector)), None)
    content = content or soup.body or soup
    meta_description = soup.find("meta", attrs={"name": "description"})
    description = meta_description.get("content", "").strip() if meta_description else ""
    if not title:
        heading = content.find("h1")
        title = heading.get_text(" ", strip=True) if heading else ""
    records = []
    seen = set()
    section = title

    def add_record(block_type: str, text: str, table_headers: str = ""):
        text = re.sub(r"\[\s*\d+\s*\]", "", text)
        text = " ".join(text.split())
        if not text or (len(text) < 12 and block_type not in {"table_row", "h1", "h2", "h3", "h4"}):
            return
        key = (section, block_type, text)
        if key in seen:
            return
        seen.add(key)
        records.append({
            "source_url": page_url,
            "page_title": title,
            "language": language,
            "section": section,
            "block_type": block_type,
            "table_headers": table_headers,
            "text": text,
        })

    for node in content.find_all((*_CONTENT_TAGS, "tr")):
        if node.name == "tr":
            table = node.find_parent("table")
            if table is None:
                continue
            header_row = next((row for row in table.find_all("tr") if row.find("th")), None)
            if node is header_row:
                continue
            caption = table.find("caption")
            preceding_heading = table.find_previous(["h1", "h2", "h3", "h4"])
            table_title = (caption.get_text(" ", strip=True) if caption else
                           preceding_heading.get_text(" ", strip=True) if preceding_heading else title)
            headers = ([cell.get_text(" ", strip=True) for cell in header_row.find_all("th")]
                       if header_row else [])
            values = [cell.get_text(" ", strip=True) for cell in node.find_all(["th", "td"])]
            text = " | ".join(f"{headers[index]}: {value}" if index < len(headers) else value
                              for index, value in enumerate(values))
            previous_section, section = section, table_title
            add_record("table_row", text, " | ".join(headers))
            section = previous_section
            continue
        if node.find_parent("table"):
            continue
        text = node.get_text(" ", strip=True)
        if node.name in {"h1", "h2", "h3", "h4"} and text:
            section = text
        add_record(node.name, text)

    if not records:
        text = content.get_text(" ", strip=True)
        if text:
            section = title
            add_record("page_text", text)
    if not records:
        raise ValueError("No readable text was found; this page may require JavaScript rendering")
    return {"url": page_url, "title": title, "description": description,
            "language": language, "records": records}


def download_webpage(url: str, timeout: float = 20) -> tuple[str, str]:
    """Download public HTML/plain text, returning its final URL and decoded markup."""
    validate_public_url(url)
    request = Request(url, headers={
        "User-Agent": "RawDataDatasetConverter/1.0 (+webpage text extraction)",
        "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.1",
    })
    opener = build_opener(ProxyHandler({}), _PublicRedirect(), _PublicHTTPSHandler())
    try:
        with opener.open(request, timeout=timeout) as response:
            body = response.read(MAX_PAGE_BYTES + 1)
            content_type = response.headers.get_content_type()
            charset = response.headers.get_content_charset() or "utf-8"
            final_url = response.geturl()
    except (HTTPError, URLError, TimeoutError) as exc:
        raise ValueError("Webpage could not be fetched; check the URL and try again") from exc
    if len(body) > MAX_PAGE_BYTES:
        raise ValueError("Webpage exceeds the 10 MB extraction limit")
    if content_type not in {"text/html", "application/xhtml+xml", "text/plain"}:
        raise ValueError("Link must return an HTML or plain-text webpage")
    validate_public_url(final_url)
    markup = body.decode(charset, errors="replace")
    return final_url, markup


def scrape_webpage(url: str, timeout: float = 20) -> dict:
    """Fetch a public page and extract readable records from its HTML."""
    final_url, markup = download_webpage(url, timeout)
    return extract_html(markup, final_url)