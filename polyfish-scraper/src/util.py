import csv, os, subprocess
from selenium import webdriver

def ydo(args):
    subprocess.run(["ydotool"] + args)

from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options

def make_driver():
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    
    service = ChromeService(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    return driver

def save_results(output: str, results_dict: dict):
    with open(output, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["data-index", "link"])
        for k, v in sorted(results_dict.items(), key=lambda x: int(x[0]) if x[0].isdigit() else x[0]):
            writer.writerow([k, v])
    print(f"Saved {len(results_dict)} items -> {output}")

def _scroll_down(driver, y):
    js = """
    try {
        const e = document.querySelector('#popup-container--arena-content > div > div > div');
        if (e) {
            const y = %Y% > e.scrollHeight? e.scrollHeight : %Y%; 
            e.scrollTo({top: y, behavior: 'auto'});
        }
    } catch(e) { }
    """.replace('%Y%', str(y))
    driver.execute_script(js)

def save_and_merge(path: str, data: list[str], overwrite=False):
    if type(data) != list:
        raise Exception("data must be a list")
        
    if os.path.exists(path) and not overwrite:
        other_data = open(path, "r").read().split('\n')
        data = list(set(data + other_data))

    open(path, "w").write('\n'.join([d for d in data if d]))

