import requests
from bs4 import BeautifulSoup
from textblob import TextBlob
import nltk
import sys
import codecs

# تنظیم کدگذاری پیش‌فرض (حل مشکل چاپ در ویندوز)
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

# دانلود ابزارهای NLTK (فقط بار اول اجرا)
nltk.download('punkt')

# 1. استخراج محتوای یک سایت خبری
def scrape_website(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            # فرض می‌کنیم محتوای اصلی در تگ <p> قرار دارد
            paragraphs = soup.find_all('p')
            content = " ".join([para.get_text() for para in paragraphs])
            return content
        else:
            print(f"خطا در دریافت داده‌ها. وضعیت: {response.status_code}")
            return None
    except Exception as e:
        print(f"خطا: {e}")
        return None

# 2. پیش‌پردازش متن
def preprocess_text(text):
    # حذف کاراکترهای غیرضروری
    clean_text = text.replace("\n", " ").strip()
    return clean_text

# 3. تحلیل احساسات
def analyze_sentiment(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment
    return sentiment

# 4. اجرای برنامه
if __name__ == "__main__":
    # نمونه لینک (می‌توانید تغییر دهید)
    url = "https://www.cnbc.com/us-economy/"
    
    print("در حال استخراج محتوا از سایت...")
    content = scrape_website(url)
    
    if content:
        print("پیش‌پردازش متن...")
        clean_content = preprocess_text(content)
        
        print("تحلیل احساسات...")
        sentiment = analyze_sentiment(clean_content)
        
        print("\nنتایج تحلیل:")
        print(f"متن استخراج‌شده: {clean_content[:200]}...")  # نمایش 200 کاراکتر اول
        print(f"قطبیت (Polarity): {sentiment.polarity}")  # عددی بین -1 (منفی) تا 1 (مثبت)
        print(f"ذهنیت (Subjectivity): {sentiment.subjectivity}")  # عددی بین 0 (عینی) تا 1 (ذهنی)
    else:
        print("محتوایی یافت نشد.")
