import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

import sys
sys.stdout.reconfigure(encoding='utf-8')


# داده‌های نمونه برای آموزش مدل
categories = ['اقتصادی', 'سیاسی', 'ورزشی']
texts = [
    "تورم و نرخ ارز افزایش یافت",  # اقتصادی
    "بورس امروز رشد خوبی داشت",   # اقتصادی
    "مجلس لایحه جدیدی تصویب کرد",  # سیاسی
    "رئیس جمهور سخنرانی کرد",      # سیاسی
    "تیم ملی به فینال رسید",       # ورزشی
    "لیگ برتر شروع شد"             # ورزشی
]
labels = [0, 0, 1, 1, 2, 2]  # برچسب‌های موضوعات (0=اقتصادی، 1=سیاسی، 2=ورزشی)

# ایجاد مدل با استفاده از TF-IDF و Naïve Bayes
model = make_pipeline(TfidfVectorizer(), MultinomialNB())

# آموزش مدل
model.fit(texts, labels)

# تابع برای پیش‌بینی موضوع متن جدید
def predict_category(text):
    pred = model.predict([text])[0]
    return categories[pred]

# تست مدل
new_text = "Is it for a day or four years?' Tariff uncertainty spooks small businesses"
print(f"موضوع متن: {predict_category(new_text)}")
