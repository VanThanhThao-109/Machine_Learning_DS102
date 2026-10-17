import pandas as pd
import requests
from bs4 import BeautifulSoup

def scrape_bookstoscrape():
    url = "https://books.toscrape.com/"
    response = requests.get(url)
    print(f"Mã trạng thái HTTP: {response.status_code}")

    data_list = []

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        book_pods = soup.find_all("article", class_="product_pod")
        print(f"Số lượng sách tìm thấy: {len(book_pods)}")

    for book in book_pods:
      title_tag = book.find("h3").find("a")
      title = title_tag.get("title", "") if title_tag else ""
      price_tag = book.find("p", class_="price_color")
      price = price_tag.get_text(strip=True) if price_tag else ""

      availability_tag = book.find("p", class_="instock availability")
      availability = (
          availability_tag.get_text(strip=True) if availability_tag else ""
      )

      star_tag = book.find("p", class_="star-rating")
      rating = star_tag.get("class")[1] if star_tag and len(star_tag.get("class", [])) > 1 else ""

      book_dict = {
          "title": title,
          "price": price,
          "availability": availability,
          "rating": rating,
      }
      data_list.append(book_dict)

    return data_list

scraped_data = scrape_bookstoscrape()

if scraped_data:
  df = pd.DataFrame(scraped_data)
  df.to_csv("books.csv", index=False, encoding="utf-8-sig")
  print("Đã lưu dữ liệu thành công vào file books.csv!")

print(scraped_data)