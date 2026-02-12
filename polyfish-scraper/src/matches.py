import time, json
from selenium.common.exceptions import JavascriptException, WebDriverException
from selenium.webdriver.common.bidi.network import Request
from util import *
from seleniumwire import webdriver

HOST = "www.challengermode.com"
ROOT = f"https://{HOST}"
URI_MATCHES = "matches/1"
OUTPUT_CSV = "src/watcher/data/matches.csv"
INPUT_CSV = "src/watcher/data/tornaments.csv"
SCROLL_PAUSE = 3.0
SCROLL_AMOUNT = 1000
MAX_EMPTY_SCROLLS = 15

tornaments = open(INPUT_CSV, "r").read().split('\n')

# $A = https://www.challengermode.com/s/PolyChampions/tournaments/1079a7c7-5037-4a62-629c-08dadeabba09
# eg: $A/matches/1

import time, json
from playwright.sync_api import sync_playwright

def collect_challenger_ids_playwright(url, listen_time=10, headless=False):
    challenger_ids = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        page = browser.new_page()

        def on_response(response):
            try:
                # Try content-type as an initial cheap filter
                ct = response.headers.get("content-type", "")
                if "application/json" in ct or response.url.endswith(".json"):
                    data = json.loads(response.text())
                    if isinstance(data, dict) and "ChallengeId" in data:
                        id = data["ChallengeId"]
                        if id not in challenger_ids:
                            challenger_ids.append(data["ChallengeId"])
                            print('found challenge id')

            except Exception:
                pass

        page.on("response", on_response)

        page.goto(url, timeout=120000)
        # wait while network calls happen
        time.sleep(listen_time)

        # browser.close()

    return challenger_ids
    
def extract_current_items(driver):
    js = r"""
    try {
        const res = [];
        const e = document.querySelector('cm-organizer-section > div.lblqdnim > div > div > div > span > div > span:nth-child(1) > span > span > span');

        res.push()
        const containers = Array.from(document.querySelectorAll('div[data-index]'));
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

def main():
    y = 0
    driver = make_driver()

    try:
        raise "ass";
        for tour in tornaments:
            if len(tour) < 2: 
                continue

            # tour="2514,/s/PolyChampions/tournaments/619c7c5d-0226-422c-da82-08daea783f18"
            # print(tour)

            [tour_id, tour_uri] = tour.split(',')

            if type(tour_id) != str or type(tour_uri) != str:
                continue

            url = f"{ROOT}/{tour_uri}/{URI_MATCHES}"

            print(f"Extracting #{tour_id}..")

            driver.get(url)

            matches = driver.execute("""
            try {
                (async () => {
                    const data = [];
                    const tappables = document.querySelectorAll('button[aria-label="View more"]').values().toArray().slice(9);
                    for (const tap of tappables) { 
                        tap.scrollIntoView();
                        await new Promise((res) => setTimeout(res, 1000));
                        tap.click();
                        await new Promise((res) => setTimeout(res, 2000));
                        const statuses = document.querySelectorAll('#react-modal-root > div > div.popup-container--modal.lblqdn1cm.lblqdn1hw.lblqdn118.lblqdn21c.lblqdn122.lblqdn17c > div > div > div > div > div > div > div > div > div.lblqdn1e.lblqdn16.lblqdn5u.lblqdn5m.lblqdnaa.lblqdna2.lblqdneq.lblqdnei > div > div.lblqdn2hg.lblqdn27q.lblqdn2fs.lblqdn10o > div > div > div > div > div.lblqdn2hg.lblqdn2fs > div > div.lblqdn2hg.lblqdn2gc.lblqdn2xk > div > div > div > div.lblqdn2hg.lblqdn2a8.lblqdn2g2.lblqdn1uo > div > span > div');
                        const matches = document.querySelectorAll('a[href*="PolyChampions/games"].do8nb00');
                        if (matches.length > statuses.length) {
                            console.log('ASS');
                        }
                        for (let i = 0; i < matches.length; i++) {
                            if (statuses[i].textContent === 'Played') {
                                console.log(matches[i].getAttribute('href'));
                                data.push(matches[i].getAttribute('href'));
                            }
                        }
                        await new Promise((res) => setTimeout(res, 1000));
                        document.querySelector('#react-modal-root > div > div.popup-container--modal.lblqdn1cm.lblqdn1hw.lblqdn118.lblqdn21c.lblqdn122.lblqdn17c > div > div > div > div > div > div > div > div > div.lblqdn1hw.lblqdn14.lblqdn5k.lblqdna0.lblqdneg.lblqdn10y.lblqdn1po.lblqdn122 > a').click();
                        await new Promise((res) => setTimeout(res, 1000));
                    }
                    console.log(data);
                    return data;
                })().catch(console.log);
            }
            catch (e) {
                return String(e);
            }
            """)

            print(matches)

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
    # print(collect_challenger_ids_playwright(
    #     url=f"{ROOT}/s/PolyChampions/tournaments/619c7c5d-0226-422c-da82-08daea783f18/{URI_MATCHES}",
    #     listen_time=60
    # ))
