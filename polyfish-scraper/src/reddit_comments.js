// Source 1:
// https://www.reddit.com/search/?q=share.polytopia.io%2Fg&type=comments&sort=new&cId=ee9ed1b4-cffd-4a3b-81e1-7333fbe08d22&iId=9806a585-1793-4159-a485-9104be73f331
// const OUTPUT = "src/scraper/data/replays_reddit.txt"

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
        
        seen.push(...items.map(x => x.getAttribute("href")));
        seen = [...new Set(seen.flat().map(x => x.trim()))];

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
