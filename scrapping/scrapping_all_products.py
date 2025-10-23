#dont need to run this code becuase this is for scrapping products from the amazon site that i already do and its indata/raw folder
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import csv, time

# ----------------------------
# CONFIGURATION
# ----------------------------
CATEGORIES = [
    "Electronics", "Mobiles", "Laptops", "Computers", "Tablets",
    "Headphones", "Cameras", "Television", "Appliances", "Refrigerator",
    "Washing Machine", "Microwave Oven", "Home & Kitchen", "Furniture",
    "Toys & Games", "Books", "Movies & TV Shows", "Music & Instruments",
    "Video Games", "Clothing", "Shoes", "Watches", "Bags & Luggage",
    "Beauty & Personal Care", "Health & Household", "Groceries & Gourmet Foods",
    "Sports & Outdoors", "Pet Supplies", "Baby Products", "Office Products",
    "Industrial & Scientific", "Automotive", "Jewelry", "Tools & Home Improvement"
]

BASE_URL = "https://www.amazon.in/s?k="
chrome_options = Options()
chrome_options.add_argument("--disable-blink-features=AutomationControlled")
chrome_options.add_argument("--window-size=1920,1080")
chrome_options.add_argument("--headless")  # hide browser window

driver = webdriver.Chrome(service=Service(), options=chrome_options)
wait = WebDriverWait(driver, 10)

# ----------------------------
# FUNCTION TO SCRAPE A PAGE
# ----------------------------
def scrape_page(soup):
    data = []
    for product in soup.find_all("div", {"data-component-type": "s-search-result"}):
        try:
            title_tag = product.h2
            title = title_tag.text.strip() if title_tag else "N/A"

            price_tag = product.select_one("span.a-price-whole")
            price = price_tag.text.strip().replace(",", "") if price_tag else "N/A"

            desc_tag = product.find("h2")
            description = desc_tag.text.strip() if desc_tag else "N/A"

            link_tag = product.find("a", {"class": "a-link-normal", "href": True})
            link = "https://www.amazon.in" + link_tag["href"] if link_tag else "N/A"

            img_tag = product.find("img", {"class": "s-image"})
            image = img_tag["src"] if img_tag else "N/A"

            data.append([title, description, price, image, link])
        except:
            continue
    return data

# ----------------------------
# MAIN CATEGORY LOOP
# ----------------------------
for category in CATEGORIES:
    print(f"\n🟦 Scraping category: {category}")
    url = BASE_URL + category.replace(" ", "+")
    driver.get(url)
    time.sleep(3)

    results = []
    page = 1

    while True:
        print(f"   ➤ Page {page}")
        # Wait until products are loaded
        wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "div[data-component-type='s-search-result']")))
        soup = BeautifulSoup(driver.page_source, "html.parser")
        batch = scrape_page(soup)
        if not batch:
            print("   ⚠️ No products found on this page.")
            break

        results.extend(batch)
        print(f"   Collected {len(batch)} items (Total so far: {len(results)})")

        # Try to click "Next" page
        try:
            next_button = driver.find_element(By.CSS_SELECTOR, "a.s-pagination-next")
            if "s-pagination-disabled" in next_button.get_attribute("class"):
                print("   ✅ Last page reached.")
                break
            driver.execute_script("arguments[0].scrollIntoView(true);", next_button)
            next_button.click()
            page += 1
            time.sleep(3)
        except:
            print("   ✅ No further pages found.")
            break

    # Save category data to CSV
    filename = f"{category.replace(' ', '_').lower()}_products.csv"
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Title", "Description", "Price (INR)", "Image URL", "Product Link"])
        writer.writerows(results)
    print(f"✅ Saved {len(results)} items for {category} → {filename}")

driver.quit()
print("\n🎉 All categories fully scraped!")
