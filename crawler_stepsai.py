import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, urldefrag
import time
import json

class WebCrawler:
    def __init__(self, base_url, max_pages=200, max_depth=5, delay=1):
        self.base_url = base_url
        self.max_pages = max_pages
        self.max_depth = max_depth
        self.delay = delay
        self.visited = set()
        self.data = {}
        self.links_by_depth = {f"depth{i}": [] for i in range(max_depth + 1)}
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

    def crawl(self, url, depth=0):
        if len(self.visited) >= self.max_pages:
            print(f"Reached max pages limit: {self.max_pages}")
            return
        if depth > self.max_depth:
            print(f"Reached max depth limit: {self.max_depth}")
            return
        if url in self.visited:
            print(f"Already visited: {url}")
            return

        print(f"Visiting: {url} at depth {depth}")
        self.visited.add(url)

        try:
            response = requests.get(url, headers=self.headers)
            print(f"Response status code: {response.status_code}")
            if response.status_code != 200:
                return
            soup = BeautifulSoup(response.content, 'html.parser')
            print(f"Successfully fetched and parsed: {url}")
            self.extract_data(soup, url)
            self.extract_links(soup, url, depth)
        except requests.RequestException as e:
            print(f"Failed to fetch {url}: {e}")

        # Add delay to avoid rate limiting
        time.sleep(self.delay)

    def extract_links(self, soup, current_url, depth):
        current_depth = f"depth{depth}"
        for link in soup.find_all('a', href=True):
            href = link['href']
            href = urljoin(current_url, href)  # Convert relative URLs to absolute URLs
            href, _ = urldefrag(href)  # Remove URL fragments
            href = self.normalize_url(href)
            if self.is_valid_url(href):
                print(f"Found link: {href}")
                self.links_by_depth[current_depth].append(href)
                if depth < self.max_depth and href not in self.visited:  # Check if not visited
                    self.crawl(href, depth + 1)

    def extract_data(self, soup, url):
        data = {
            "url": url,
            "title": self.extract_title(soup),
            "headers": self.extract_headers(soup),
            "paragraphs": self.extract_paragraphs(soup),
        }
        print(f"Extracted data from {url}: {data}")
        self.data[url] = data

    def extract_title(self, soup):
        title = soup.find('title')
        return title.text.strip() if title else None

    def extract_headers(self, soup):
        headers = [header.text.strip() for header in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])]
        return headers

    def extract_paragraphs(self, soup):
        paragraphs = [para.text.strip() for para in soup.find_all('p')]
        return paragraphs

    def normalize_url(self, url):
        parsed_url = urlparse(url)
        return parsed_url.scheme + "://" + parsed_url.netloc + parsed_url.path

    def is_valid_url(self, url):
        parsed_base = urlparse(self.base_url)
        parsed_url = urlparse(url)
        is_valid = parsed_url.netloc == parsed_base.netloc
        print(f"Checking if valid URL: {url} -> {is_valid}")
        return is_valid

    def save_data_to_file(self, data_filename, links_filename):
        with open(data_filename, 'w', encoding='utf-8') as file:
            json.dump(self.data, file, indent=4, ensure_ascii=False)
        print(f"Data saved to {data_filename}")

        with open(links_filename, 'w', encoding='utf-8') as file:
            json.dump(self.links_by_depth, file, indent=4, ensure_ascii=False)
        print(f"Links saved to {links_filename}")

    def save_data_as_text(self, data_filename):
        with open(data_filename, 'w', encoding='utf-8') as file:
            for url, data in self.data.items():
                file.write(f"URL: {url}\n")
                file.write(f"Title: {data['title']}\n")
                file.write(f"Headers: {', '.join(data['headers'])}\n")
                file.write("Paragraphs:\n")
                for paragraph in data['paragraphs']:
                    file.write(f"\t{paragraph}\n")
                file.write("\n")
        print(f"Data saved as text to {data_filename}")

if __name__ == "__main__":
    start_url = "https://docs.nvidia.com/cuda/"  # Replace with the URL you want to start crawling
    crawler = WebCrawler(start_url, max_pages=200, max_depth=5, delay=1)
    crawler.crawl(start_url)
    # Save the extracted data and links to files
    crawler.save_data_to_file('crawled_data.json', 'crawled_links.json')
    crawler.save_data_as_text('crawled_data.txt')
