import time, os
import asyncio
from selenium.webdriver.chrome.webdriver import WebDriver
from selenium.common.exceptions import JavascriptException, WebDriverException
from util import *

SOURCE_URLS = [
    "https://www.reddit.com/search/?q=share.polytopia.io%2Fg&type=posts&sort=comments&cId=ee9ed1b4-cffd-4a3b-81e1-7333fbe08d22&iId=84e4878b-d734-443c-bd96-c6c506ed7d9d",
    "https://www.reddit.com/search/?q=share.polytopia.io%2Fg&type=posts&sort=top&cId=ee9ed1b4-cffd-4a3b-81e1-7333fbe08d22&iId=828de114-520a-455f-a6ed-56152fd6e407",
    "https://www.reddit.com/search/?q=share.polytopia.io%2Fg&type=posts&sort=relevance&cId=ee9ed1b4-cffd-4a3b-81e1-7333fbe08d22&iId=4facf515-a7b2-4e2c-abd4-5a2738caaf0b",
    "https://www.reddit.com/search/?q=share.polytopia.io%2Fg&type=posts&sort=new&cId=ee9ed1b4-cffd-4a3b-81e1-7333fbe08d22&iId=07c528c1-3b16-4e3b-9cb0-01dd1a4ac906",
]

OUTPUT_ROOT = "src/scraper/data/"
OUTPUT_SOURCES = f"{OUTPUT_ROOT}sources_reddit.txt"
OUTPUT_FAILED_SOURCES = f"{OUTPUT_ROOT}sources_reddit_failed.txt"
OUTPUT_REPLAYS = f"{OUTPUT_ROOT}replays_reddit.txt"
SCROLL_PAUSE = 3.0
SCROLL_AMOUNT = 1000
MAX_EMPTY_SCROLLS = 3

tornaments = []

def _extract_current_items(driver):
    js = r"""
    try {
        const res = [];
        const e = document.querySelector('a[data-testid="post-title"]');

        res.push()
        const containers = Array.from(document.querySelectorAll('a[data-testid="post-title"]'));
        containers.forEach(c => {
            const idx = c.getAttribute('href');
            res.push([idx, idx]);
        });
        return res;
    } catch(e) {
        return {__error: String(e)};
    }
    """.replace('%JOE%', 'MOMMA')

    try:
        raw = driver.execute_script(js)
    except (JavascriptException, WebDriverException) as e:
        print("JS execution failed:", e)
        return None
    if isinstance(raw, dict) and "__error" in raw:
        print("Page JS error:", raw["__error"])
        return None
    items = {}
    for pair in raw:
        if not pair or len(pair) < 2:
            continue
        idx, link = pair[0], pair[1]
        if idx is None:
            continue
        items[str(idx)] = link
    return items

def _scroll_down(driver):
    driver.execute_script("window.scrollBy({ top: SCROLL_AMOUNT, behavior: 'smooth' })".replace('SCROLL_AMOUNT', str(SCROLL_AMOUNT)))

async def scrape_sources(driver: WebDriver, url: str):
    seen = {}
    empty_scrolls = 0
    scroll_count = 0

    driver.get(url)

    while empty_scrolls < MAX_EMPTY_SCROLLS:
        if scroll_count % 100 == 0 and scroll_count != 0:
            driver.close()
            driver = make_driver()

        items = _extract_current_items(driver)

        if items is None:
            print('failed! no items')
            break

        _scroll_down(driver)
        scroll_count += 1

        new_count_before = len(seen)
        for k, v in items.items():
            if k not in seen and v:
                seen[k] = v

        if len(seen) > new_count_before:
            print(f"+{len(seen) - new_count_before} [{len(seen)}]")
            save_and_merge(OUTPUT_SOURCES, list(items.values()))
            empty_scrolls = 0
        else:
            print(f"exiting in {MAX_EMPTY_SCROLLS - empty_scrolls}..")
            empty_scrolls += 1
        
        time.sleep(SCROLL_PAUSE)

def scrape_source(driver: WebDriver, url: str):
    driver.get(url)
    time.sleep(1)

    magic = lambda: driver.execute_script(
        """
        return document.querySelectorAll("a[href*='share.polytopia.io/g']")
            .values()
            .map(x => x.getAttribute('href'))
            .toArray()
        """
    )

    values = magic()

    if values is None:
        time.sleep(3)
        values = magic()
        if values is None:
            time.sleep(3)
            values = magic()

    return values

def check_for_pending(driver):
    if os.path.exists(OUTPUT_SOURCES):
        pending_sources = open(OUTPUT_SOURCES).read().split('\n')
        pending_sources = [uri for uri in pending_sources if uri]

        failed = []
        replays = []
        count = len(pending_sources)

        for i, uri in enumerate(pending_sources):
            if i % 100 == 0 and i != 0:
                driver.close()
                driver = make_driver()

            try:
                urls = scrape_source(driver, "https://www.reddit.com" + uri)
                if len(urls) == 0:
                    raise Exception('No replays found')
                save_and_merge(OUTPUT_REPLAYS, urls)
                replays.extend(urls)
                print(f"({len(urls)}) {i}/{count} [{len(replays)}]")

            except Exception as e:
                failed.append(uri)
                print(f"Failed to scrape {uri}: {e}")

        os.remove(OUTPUT_SOURCES)
        open(OUTPUT_FAILED_SOURCES, "w").write('\n'.join(failed))

async def main():
    driver = make_driver()
    # tasks = []
    # for url in SOURCE_URLS:
    #     tasks.append(asyncio.create_task(scrape_sources(make_driver(), url)))
    # await asyncio.gather(*tasks)

    check_for_pending(driver)

    driver.close()

if __name__ == "__main__":
    asyncio.run(main())
