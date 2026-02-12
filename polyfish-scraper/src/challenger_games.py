import time
import multiprocessing as mp
from selenium.common.exceptions import JavascriptException, WebDriverException
from util import *

HOST = "www.challengermode.com"
ROOT = f"https://{HOST}"
URI_MATCHES = "matches/1?state=3"
DIR_REPLAYS = "src/scraper/data/replays_polysseum.txt"
DIR_GAMES = "src/scraper/data/sources.challenger_games.txt"
DIR_TORNAMENTS = "src/scraper/data/sources.challenger_tornaments.txt"
DIR_DONE = DIR_GAMES.replace('.txt', '.done.txt')
DIR_FAILED = DIR_GAMES.replace('.txt', '.failed.txt')
SCROLL_PAUSE = 3.0
SCROLL_AMOUNT = 1000
MAX_EMPTY_SCROLLS = 3
# NUM_WORKERS = 13
NUM_WORKERS = 5
RESTART_EVERY = 20

done_replays = open(DIR_DONE, "r").read().split('\n')

def log(msg, wid):
    print(f"[minion-{'0'+str(wid) if int(wid) < 10 else str(wid)}] {msg}")

def chunkify(lst, n):
    """Split list lst into n chunks as evenly as possible."""
    k, m = divmod(len(lst), n)
    chunks = []
    start = 0
    for i in range(n):
        size = k + (1 if i < m else 0)
        chunks.append(lst[start:start+size])
        start += size
    return chunks

def extract_indexes(driver):
    js = r"""
    try {
        const res = [];
        const containers = Array.from(document.querySelectorAll('div[data-index]'));
        containers.forEach(c => {
            const idx = c.getAttribute('data-index');
            res.push(idx);
        });
        return res;
    } catch(e) {
        return [{__error: String(e)}];
    }
    """
    try:
        items = driver.execute_script(js)
    except (JavascriptException, WebDriverException) as e:
        print("JS execution failed:", e)
        return None
    return items

def scrape_games(driver, url: str):
    driver.get(url)
    time.sleep(3)

    try:
        seen = []
        empty_scrolls = 0

        while empty_scrolls < MAX_EMPTY_SCROLLS:
            items = extract_indexes(driver)

            if items is None:
                continue
            
            new_seen = []
            for v in items:
                if v not in seen:
                    new_seen.append(v)
                    seen.append(v)

            if len(new_seen) > 0:
                for idx in new_seen:
                    if not driver.execute_script("""
                        const gameBtn = document.querySelector('div[data-index="%ID%"] button[aria-label="View more"]');
                        if (!gameBtn) return false;
                        gameBtn.scrollIntoView();
                        return true;
                    """.replace('%ID%', str(idx))):
                        print(f"finished")
                        break

                    time.sleep(0.5)
                    driver.execute_script("""
                        const gameBtn = document.querySelector('div[data-index="%ID%"] button[aria-label="View more"]');
                        gameBtn.scrollIntoView();
                        gameBtn.click();
                    """.replace('%ID%', str(idx)))
                    time.sleep(2.5)

                    game_urls = driver.execute_script("""
                        const data = [];
                        const statuses = document.querySelectorAll('#react-modal-root > div > div.popup-container--modal.lblqdn1cm.lblqdn1hw.lblqdn118.lblqdn21c.lblqdn122.lblqdn17c > div > div > div > div > div > div > div > div > div.lblqdn1e.lblqdn16.lblqdn5u.lblqdn5m.lblqdnaa.lblqdna2.lblqdneq.lblqdnei > div > div.lblqdn2hg.lblqdn27q.lblqdn2fs.lblqdn10o > div > div > div > div > div.lblqdn2hg.lblqdn2fs > div > div.lblqdn2hg.lblqdn2gc.lblqdn2xk > div > div > div > div.lblqdn2hg.lblqdn2a8.lblqdn2g2.lblqdn1uo > div > span > div');
                        const matches = document.querySelectorAll('a[href*="PolyChampions/games"].do8nb00');
                        for (let i = 0; i < matches.length; i++) {
                            if (statuses[i].textContent === 'Played') {
                                data.push(matches[i].getAttribute('href').trim());
                            }
                        }
                        document.querySelector('#react-modal-root > div > div.popup-container--modal.lblqdn1cm.lblqdn1hw.lblqdn118.lblqdn21c.lblqdn122.lblqdn17c > div > div > div > div > div > div > div > div > div.lblqdn1hw.lblqdn14.lblqdn5k.lblqdna0.lblqdneg.lblqdn10y.lblqdn1po.lblqdn122 > a').click();
                        return data;
                    """)

                    save_and_merge(DIR_GAMES, game_urls)

                    print(f"+{len(game_urls)} ({len(seen)})")

                empty_scrolls = 0
                
            else:
                empty_scrolls += 1
                print(f"exiting in {MAX_EMPTY_SCROLLS - empty_scrolls}..")

            time.sleep(1.0)

    except Exception as e:
        print("scrape_games exception:", e)

def scrape_game(driver, url: str):
    driver.get(url)
    time.sleep(3)

    js = r"""
    let urls = Array.from(document.querySelectorAll('a[href*="share.polytopia.io/g"]'));
    urls = urls.map(u => u.getAttribute('href'));
    return urls;
    """
    try:
        items = driver.execute_script(js)
        tries = 6
        while tries > 0 and len(items) == 0:
            time.sleep(3)
            items = driver.execute_script(js)
            tries -= 1

    except (JavascriptException, WebDriverException) as e:
        print("JS execution failed:", e)
        return None
    
    return items

def scrape_games_worker(tournaments_chunk, wid):
    log(f"starting with {len(tournaments_chunk)} tournaments", wid)
    driver = None
    
    try:
        driver = make_driver()
        for i, uri in enumerate(tournaments_chunk):
            if i % RESTART_EVERY == 0 and i > 0:
                try:
                    driver.quit()
                except Exception:
                    pass
                driver = make_driver()

            try:
                full_url = f"{ROOT}{uri}/{URI_MATCHES}"
                print(f"[minion-{wid}] {uri.split('/')[-1]} ({i+1}/{len(tournaments_chunk)})")
                scrape_games(driver, full_url)

                data_now = open(DIR_GAMES, "r").read().split('\n')
                data_now = [x for x in data_now if x != uri]

                save_and_merge(DIR_DONE, data_now, True)

            except Exception as e:
                print(f"[minion-{wid}] error scraping {uri}: {e}")

    except Exception as e:
        print(f"[minion-{wid}] fatal error: {e}")
    
    finally:
        if driver is not None:
            try:
                driver.quit()
            except Exception:
                pass
        print(f"[minion-{wid}] exiting")

def scrape_game_worker(games_chunk: list[str], wid: str):
    global done_replays
    log(f"starting with {len(games_chunk)} games", wid)
    driver = None
    
    try:
        driver = make_driver()
        for i, uri in enumerate(games_chunk):
            if uri in done_replays:
                log(f"skipping {i+1}/{len(games_chunk)}", wid)
                continue

            if i % RESTART_EVERY == 0 and i > 0:
                try:
                    driver.quit()
                except Exception:
                    pass
                driver = make_driver()

            try:
                full_url = f"{ROOT}{uri}"
                items = scrape_game(driver, full_url)

                if len(items) == 0:
                    log(f"failed ({i+1}/{len(games_chunk)})", wid)
                    save_and_merge(DIR_FAILED, [uri])
                    continue

                # save to replays
                save_and_merge(DIR_REPLAYS.replace('.txt', f'{wid}.txt'), items)

                log(f"+{len(items)} ({i+1}/{len(games_chunk)})", wid)

                # save to .done
                save_and_merge(DIR_DONE, [uri])

                # data_now = open(DIR_GAMES, "r").read().split('\n')
                # data_now = [x for x in data_now if x != uri]
                
                # remove from src
                # save_and_merge(DIR_GAMES, data_now, True)
                
            except Exception as e:
                log(f"error scraping {uri}: {e}", wid)

    except Exception as e:
        log(f"fatal error: {e}", wid)
    
    finally:
        if driver is not None:
            try:
                driver.quit()
            except Exception:
                pass
        log(f"exiting", wid)

if __name__ == "__main__":
    replays = [x.strip() for x in open(DIR_GAMES, "r").read().splitlines() if x.strip()]

    if not replays:
        print("No data")
        exit(1)

    NUM_WORKERS = min(NUM_WORKERS, len(replays))
    chunks = chunkify(replays, NUM_WORKERS)

    processes = []
    for wid, chunk in enumerate(chunks):
        time.sleep(1)
        p = mp.Process(target=scrape_game_worker, args=(chunk, wid), daemon=False)
        p.start()
        processes.append(p)
        log(f"started with {len(chunk)} items", wid)

    try:
        for wid, p in enumerate(processes):
            p.join()
            log(f"finished", wid)
    except KeyboardInterrupt:
        print("[god] stopping")
        for p in processes:
            try:
                p.terminate()
            except Exception:
                pass
        for wid, p in enumerate(processes):
            p.join()
            log(f"terminated", wid)
