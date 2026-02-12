# scrape_data_index.py
import time
import os
from selenium.common.exceptions import JavascriptException, WebDriverException
from util import *

URL = "https://www.challengermode.com/s/PolyChampions/tournaments?filter=%7B%22state%22%3A%22past%22%7D"
PROFILE_DIR = os.path.join(os.path.expanduser("~"), "Desktop/Coding/PolyAI/src/watcher/.profile") 
SCROLL_PAUSE = 3.0
SCROLL_AMOUNT = 1000
MAX_EMPTY_SCROLLS = 15
OUTPUT_CSV = "src/watcher/data/sources.challenger_tornaments.csv"

def extract_current_items(driver):
    js = r"""
    try {
        const containers = Array.from(document.querySelectorAll('div[data-index]'));
        const res = [];
        containers.forEach(c => {
            const idx = c.getAttribute('data-index');
            const anchor = c.querySelector('div > div > a:first-of-type');
            if (anchor) {
                const link = anchor.getAttribute('href') || anchor.getAttribute('src') || '';
                if (link == '/s/PolyChampions/tournaments') {
                    throw Error("Page is buffering");
                }
                res.push([idx, link]);
            } else {
                // res.push([idx, null]);
            }
        });
        return res;
    } catch(e) {
        return {__error: String(e)};
    }
    """
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

def main():
    y = 0
    driver = make_driver(PROFILE_DIR)
    try:
        if URL:
            driver.get(URL)
        seen = {}
        empty_scrolls = 0
        scroll_count = 0

        items = extract_current_items(driver)
        
        while items is None or len(items) == 0:
            time.sleep(SCROLL_PAUSE)
            items = extract_current_items(driver)
            if items is None or len(items) == 0:
                print('Buffering..')
        
        seen.update({k: v for k, v in items.items() if v})
        _scroll_down(driver, y)

        while empty_scrolls < MAX_EMPTY_SCROLLS:
            time.sleep(SCROLL_PAUSE)
            items = extract_current_items(driver)

            if items is None:
                y -= 50
                if y < 0: y = 0
                continue

            _scroll_down(driver, y)
            scroll_count += 1

            new_count_before = len(seen)
            for k, v in items.items():
                if k not in seen and v:
                    seen[k] = v

            if len(seen) > new_count_before:
                print(f"[{scroll_count}] New items found: {len(seen) - new_count_before} (total {len(seen)})")
                empty_scrolls = 0
            else:
                empty_scrolls += 1
                print(f"[{scroll_count}] No new items. empty_scrolls={empty_scrolls}/{MAX_EMPTY_SCROLLS}")

            time.sleep(0.2)
            y += SCROLL_AMOUNT

        print("Finished scrolling. Total items found:", len(seen))
        save_results(OUTPUT_CSV, seen)

    except Exception as e:
        print(e)

    finally:
        while True:
            try:
                _ = driver.title
                time.sleep(1)
            except Exception:
                break

if __name__ == "__main__":
    main()
