// Source 1:
// https://www.reddit.com/search/?q=share.polytopia.io%2Fg&type=comments&sort=new&cId=ee9ed1b4-cffd-4a3b-81e1-7333fbe08d22&iId=9806a585-1793-4159-a485-9104be73f331
// const OUTPUT = "src/scraper/data/replays_reddit.txt"

(async () => {
    const data = [];
    
    let idx = 1;
    
    while (true) {
        const gameBtn = document.querySelector('div:nth-child(%X%) button[aria-label="View more"]'
            .replace('%X%', idx)
        );

        if (gameBtn == undefined) {
            break;
        }

        gameBtn.scrollIntoView();
        await new Promise((res) => setTimeout(res, 500));

        gameBtn.click();
        await new Promise((res) => setTimeout(res, 1500));

        const statuses = document.querySelectorAll('#react-modal-root > div > div.popup-container--modal.lblqdn1cm.lblqdn1hw.lblqdn118.lblqdn21c.lblqdn122.lblqdn17c > div > div > div > div > div > div > div > div > div.lblqdn1e.lblqdn16.lblqdn5u.lblqdn5m.lblqdnaa.lblqdna2.lblqdneq.lblqdnei > div > div.lblqdn2hg.lblqdn27q.lblqdn2fs.lblqdn10o > div > div > div > div > div.lblqdn2hg.lblqdn2fs > div > div.lblqdn2hg.lblqdn2gc.lblqdn2xk > div > div > div > div.lblqdn2hg.lblqdn2a8.lblqdn2g2.lblqdn1uo > div > span > div');
        const matches = document.querySelectorAll('a[href*="PolyChampions/games"].do8nb00');

        for (let i = 0; i < matches.length; i++) {
            if (statuses[i].textContent === 'Played') {
                console.log(`+ 1`);
                data.push(matches[i].getAttribute('href').trim());
            }
        }

        idx++;

        // tap exit button
        document.querySelector('#react-modal-root > div > div.popup-container--modal.lblqdn1cm.lblqdn1hw.lblqdn118.lblqdn21c.lblqdn122.lblqdn17c > div > div > div > div > div > div > div > div > div.lblqdn1hw.lblqdn14.lblqdn5k.lblqdna0.lblqdneg.lblqdn10y.lblqdn1po.lblqdn122 > a').click();
    }

    console.log(data);

    return data;
})().catch(console.log);

(async () => {
    await new Promise(resolve => setTimeout(resolve, 5000));

    const SCROLL_PAUSE = 1;
    const SCROLL_AMOUNT = 500;
    const MAX_EMPTY_SCROLLS = 15;
    let seen = [];
    let empty_scrolls = 0;
    let scroll_count = 0;

    function scroll_down() {
        window.scrollBy({ top: SCROLL_AMOUNT, behavior: 'smooth' });
    }

    while (empty_scrolls < MAX_EMPTY_SCROLLS) {
        items = document.querySelectorAll("a[href*='share.polytopia.io/g']").values();
        await new Promise(resolve => setTimeout(resolve, SCROLL_PAUSE * 1000));
        scroll_down();
        scroll_count += 1;
        const new_count_before = seen.length;
        
        seen.push(...items.map(x => x.getAttribute("href").trim()));
        seen = [...new Set(seen)];

        if (seen.length > new_count_before) {
            // console.log(`[${scroll_count}] New items found: ${seen.length - new_count_before} (total ${seen.length})`);
            empty_scrolls = 0;
        } else {
            empty_scrolls += 1;
            // console.log(`[${scroll_count}] No new items. empty_scrolls=${empty_scrolls}/${MAX_EMPTY_SCROLLS}`);
        }
        await new Promise(resolve => setTimeout(resolve, 200));
    }

    console.log("Total items found:", seen.length);
    console.log("-------START DATA-------");
    console.log(seen.join("\n"));
    console.log("-------END DATA-------");
})();
