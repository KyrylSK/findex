import asyncio
import logging
import random
import time
import collections
from urllib.parse import urlparse, urlunparse, urljoin
from urllib.robotparser import RobotFileParser
from typing import Optional, Tuple, Set, Dict, AsyncGenerator
import httpx
from selectolax.lexbor import LexborHTMLParser

logger = logging.getLogger("findex.crawler")

def canonize_url(url: str, base_url: Optional[str] = None) -> Optional[str]:
    """Канонізує URL: нижній регістр, відкидання фрагментів."""
    if base_url:
        url = urljoin(base_url, url)
        
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return None
        
    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()
    
    if netloc.endswith(":80") and scheme == "http":
        netloc = netloc[:-3]
    elif netloc.endswith(":443") and scheme == "https":
        netloc = netloc[:-4]
        
    path = parsed.path or "/"
    if not path.startswith("/"):
        path = "/" + path
        
    normalized = urlunparse((scheme, netloc, path, parsed.params, parsed.query, ''))
    return normalized

class RobotsCache:
    """Кешує правила robots.txt для кожного хоста."""
    def __init__(self, client: httpx.AsyncClient, user_agent: str):
        self.client = client
        self.user_agent = user_agent
        self._cache: Dict[str, RobotFileParser] = {}

    async def can_fetch(self, url: str) -> bool:
        parsed = urlparse(url)
        origin = f"{parsed.scheme}://{parsed.netloc}"
        
        if origin not in self._cache:
            rp = RobotFileParser()
            robots_url = f"{origin}/robots.txt"
            rp.set_url(robots_url)
            try:
                response = await self.client.get(robots_url, follow_redirects=True, timeout=5.0)
                if response.status_code == 200:
                    rp.parse(response.text.splitlines())
                else:
                    rp.parse(["User-agent: *", "Allow: /"])
            except Exception:
                rp.parse(["User-agent: *", "Allow: /"])
            self._cache[origin] = rp
            
        return self._cache[origin].can_fetch(self.user_agent, url)

async def fetch_page(
    client: httpx.AsyncClient, url: str, robots_cache: RobotsCache, max_retries: int = 3
) -> Tuple[str, int, bytes, float]:
    """Завантажує сторінку з exponential backoff та jitter."""
    if not await robots_cache.can_fetch(url):
        logger.info(f"Blocked by robots.txt: {url}")
        return url, 999, b"", 0.0

    delay = 1.0
    for attempt in range(max_retries + 1):
        start_time = time.perf_counter()
        try:
            async with asyncio.timeout(10.0):
                response = await client.get(url, follow_redirects=True)
                elapsed = time.perf_counter() - start_time
                status = response.status_code
                
                if status == 429 or (500 <= status < 600):
                    if attempt < max_retries:
                        retry_after = response.headers.get("Retry-After")
                        if retry_after and retry_after.isdigit():
                            wait_time = float(retry_after)
                        else:
                            wait_time = delay + random.uniform(0, 0.5)
                            delay *= 2
                        logger.warning(f"Retry {url} in {wait_time:.2f}s")
                        await asyncio.sleep(wait_time)
                        continue
                return url, status, response.content, elapsed
                
        except (httpx.TimeoutException, asyncio.TimeoutError) as e:
            elapsed = time.perf_counter() - start_time
            if attempt < max_retries:
                wait_time = delay + random.uniform(0, 0.5)
                delay *= 2
                logger.warning(f"Timeout {url}. Retrying in {wait_time:.2f}s")
                await asyncio.sleep(wait_time)
                continue
            return url, 408, b"", elapsed
        except Exception as e:
            logger.error(f"Error {url}: {e}")
            return url, 500, b"", time.perf_counter() - start_time

    return url, 500, b"", 0.0

async def crawl(
    start_url: str,
    max_pages: int,
    concurrency: int,
    user_agent: str = "findex-bot/1.0 (contact: admin@example.com)"
) -> AsyncGenerator[dict, None]:
    """Асинхронний генератор-краулер."""
    url_queue: asyncio.Queue[str] = asyncio.Queue()
    await url_queue.put(start_url)
    seen: Set[str] = {start_url}
    output_queue: asyncio.Queue[dict] = asyncio.Queue()
    
    active_tasks = 0
    pages_crawled = 0
    
    global_sem = asyncio.Semaphore(concurrency)
    host_sems: Dict[str, asyncio.Semaphore] = collections.defaultdict(lambda: asyncio.Semaphore(1))
    host_last_visit: Dict[str, float] = {}

    async with httpx.AsyncClient(headers={"User-Agent": user_agent}, follow_redirects=True) as client:
        robots_cache = RobotsCache(client, user_agent)
        
        async def worker():
            nonlocal active_tasks
            while True:
                url = await url_queue.get()
                active_tasks += 1
                try:
                    parsed = urlparse(url)
                    host = parsed.netloc
                    
                    async with global_sem:
                        async with host_sems[host]:
                            # Мінімум 1 секунда між запитами до одного хоста
                            now = time.time()
                            last = host_last_visit.get(host, 0.0)
                            if now - last < 1.0:
                                await asyncio.sleep(1.0 - (now - last))
                            host_last_visit[host] = time.time()
                            
                            _, status, content, elapsed = await fetch_page(client, url, robots_cache)
                    
                    text_content = ""
                    if status == 200 and content:
                        try:
                            tree = LexborHTMLParser(content)
                            text_content = tree.text(separator=' ', strip=True)
                            
                            for node in tree.css("a"):
                                href = node.attributes.get("href")
                                if href:
                                    canonized = canonize_url(href, base_url=url)
                                    if canonized and canonized not in seen and canonized.startswith("http"):
                                        seen.add(canonized)
                                        await url_queue.put(canonized)
                        except Exception as e:
                            logger.error(f"HTML Parse error {url}: {e}")

                    await output_queue.put({
                        "url": url,
                        "status": status,
                        "elapsed": elapsed,
                        "text": text_content,
                        "queue_size": url_queue.qsize()
                    })
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Worker error: {e}")
                finally:
                    url_queue.task_done()
                    active_tasks -= 1
                        
        try:
            # TaskGroup гарантує правильне управління скасуванням воркерів
            async with asyncio.TaskGroup() as tg:
                workers = [tg.create_task(worker()) for _ in range(concurrency)]
                
                while pages_crawled < max_pages:
                    # Умова зупинки: черга порожня, воркери сплять, нових результатів немає
                    if url_queue.empty() and active_tasks == 0 and output_queue.empty():
                        break
                        
                    try:
                        res = await asyncio.wait_for(output_queue.get(), timeout=0.1)
                        pages_crawled += 1
                        yield res
                    except asyncio.TimeoutError:
                        continue
                        
                # Скасовуємо всіх воркерів після досягнення ліміту max_pages
                for w in workers:
                    w.cancel()
        except asyncio.CancelledError:
            pass