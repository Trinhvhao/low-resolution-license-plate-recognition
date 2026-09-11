
import requests
from bs4 import BeautifulSoup
import os
import re
from urllib.parse import urljoin, urlparse
from selenium import webdriver
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time
import random

# Danh sách User-Agent để xoay
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36',
    'Mozilla/5.0 (X11; Linux x86_64; rv:89.0) Gecko/20100101 Firefox/89.0'
]

# Thư mục lưu trữ website
BASE_DIR = "truongfoods_clone"
DOMAIN = "https://truongfoods.vn"


def create_dir(directory):
    """Tạo thư mục nếu chưa tồn tại."""
    if not os.path.exists(directory):
        os.makedirs(directory)


def save_file(filepath, content, mode='w'):
    """Lưu nội dung vào file, xử lý đúng chế độ văn bản và nhị phân."""
    create_dir(os.path.dirname(filepath))
    if mode == 'w':
        with open(filepath, mode, encoding='utf-8') as f:
            f.write(content)
    else:  # mode='wb'
        with open(filepath, mode) as f:
            f.write(content)


def get_filename_from_url(url):
    """Tạo tên file từ URL, đảm bảo hợp lệ."""
    parsed = urlparse(url)
    path = parsed.path
    if path == '/' or not path:
        return 'index.html'
    # Loại bỏ ký tự không hợp lệ
    path = re.sub(r'[^\w\-./]', '_', path.strip('/'))
    if not path.endswith('.html'):
        path += '.html'
    return path


def download_resource(url, base_dir):
    """Tải tài nguyên (hình ảnh, CSS, JS, video) và lưu vào thư mục."""
    try:
        headers = {'User-Agent': random.choice(USER_AGENTS)}
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            parsed = urlparse(url)
            ext = os.path.splitext(parsed.path)[1] or '.bin'
            # Xử lý các định dạng video
            if ext in ['.mp4', '.webm', '.ogg']:
                resource_path = os.path.join(base_dir, 'resources', 'videos', parsed.path.lstrip('/'))
            else:
                resource_path = os.path.join(base_dir, 'resources', parsed.path.lstrip('/'))
            save_file(resource_path, response.content, mode='wb')
            print(f"Đã tải tài nguyên: {url}")
            return resource_path
        else:
            print(f"Lỗi tải tài nguyên {url}: {response.status_code}")
            return None
    except Exception as e:
        print(f"Lỗi khi tải tài nguyên {url}: {e}")
        return None


def crawl_page(url, visited, base_dir, use_selenium=False):
    """Crawl một trang và trả về danh sách liên kết."""
    if url in visited:
        return []
    visited.add(url)

    headers = {'User-Agent': random.choice(USER_AGENTS)}
    links = []

    try:
        if use_selenium:
            # Sử dụng Selenium để tải trang động
            options = ChromeOptions()
            options.add_argument('--headless')
            options.add_argument(f'user-agent={random.choice(USER_AGENTS)}')
            driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
            driver.get(url)
            time.sleep(random.uniform(2, 5))  # Chờ JavaScript tải
            html_content = driver.page_source
            driver.quit()
        else:
            # Sử dụng requests cho trang tĩnh
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code != 200:
                print(f"Lỗi tải trang {url}: {response.status_code}")
                return []
            html_content = response.text

        soup = BeautifulSoup(html_content, 'html.parser')

        # Lưu HTML
        filename = get_filename_from_url(url)
        save_path = os.path.join(base_dir, filename)
        save_file(save_path, str(soup), mode='w')
        print(f"Đã lưu trang: {url} vào {save_path}")

        # Tải tài nguyên (img, css, js, video)
        for tag in soup.find_all(['img', 'link', 'script', 'video']):
            src = tag.get('src') or tag.get('href')
            if src:
                abs_url = urljoin(url, src)
                if urlparse(abs_url).netloc == urlparse(DOMAIN).netloc:
                    download_resource(abs_url, base_dir)
            # Xử lý thẻ <source> trong <video>
            if tag.name == 'video':
                source = tag.find('source')
                if source and source.get('src'):
                    abs_url = urljoin(url, source['src'])
                    if urlparse(abs_url).netloc == urlparse(DOMAIN).netloc:
                        download_resource(abs_url, base_dir)

        # Thu thập liên kết
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            abs_url = urljoin(url, href)
            parsed = urlparse(abs_url)
            if parsed.netloc == urlparse(DOMAIN).netloc and abs_url not in visited:
                links.append(abs_url)

        return links
    except Exception as e:
        print(f"Lỗi khi crawl {url}: {e}")
        return []


def clone_website(start_url, base_dir):
    """Clone toàn bộ website."""
    create_dir(base_dir)
    create_dir(os.path.join(base_dir, 'resources'))
    create_dir(os.path.join(base_dir, 'resources', 'videos'))

    visited = set()
    queue = [start_url]

    while queue:
        url = queue.pop(0)
        # Thử dùng requests trước, nếu thất bại thì dùng Selenium
        links = crawl_page(url, visited, base_dir, use_selenium=False)
        if not links:
            print(f"Thử dùng Selenium cho {url}")
            links = crawl_page(url, visited, base_dir, use_selenium=True)

        queue.extend([link for link in links if link not in visited and link not in queue])
        time.sleep(random.uniform(1, 3))  # Tránh request quá nhanh


if __name__ == "__main__":
    start_url = "https://truongfoods.vn/"
    clone_website(start_url, BASE_DIR)
    print(f"Đã clone website vào thư mục {BASE_DIR}")