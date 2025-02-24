import cv2
import numpy as np
from pdf2image import convert_from_path
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from io import BytesIO
from PIL import Image
import concurrent.futures
import time

import sys
sys.stdout.reconfigure(encoding='utf-8')


def pdf_to_images(pdf_path, dpi=200):  # کاهش DPI برای سرعت بیشتر
    """تبدیل PDF به تصاویر با DPI کم‌تر برای پردازش سریع‌تر"""
    return convert_from_path(pdf_path, dpi=dpi)

def extract_important_regions(image):
    """فقط بخش‌های حاوی جداول را استخراج می‌کنیم (بدون نیاز به پردازش تمام تصویر)"""
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)  # استخراج لبه‌ها برای شناسایی بخش‌های مهم
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    table_regions = [cv2.boundingRect(cnt) for cnt in contours if cv2.contourArea(cnt) > 500]
    return table_regions

def process_page(img1, img2):
    """مقایسه تغییرات در هر صفحه از دو PDF"""
    tables1 = extract_important_regions(img1)
    tables2 = extract_important_regions(img2)

    diff_images = []
    for table1, table2 in zip(tables1, tables2):
        x, y, w, h = table1
        table_img1 = np.array(img1.crop((x, y, x + w, y + h)))
        table_img2 = np.array(img2.crop((x, y, x + w, y + h)))
        
        # مقایسه سریع دو بخش
        if not np.array_equal(table_img1, table_img2):  
            diff_images.append(table_img1)  # ذخیره تصویری که تغییر کرده

    return diff_images

def save_differences_to_pdf(diff_images, output_pdf):
    """ذخیره تصاویر تفاوت‌ها در یک فایل PDF"""
    c = canvas.Canvas(output_pdf, pagesize=letter)
    y_position = 700  

    for img in diff_images:
        _, img_encoded = cv2.imencode('.jpg', img)  
        img_bytes = BytesIO(img_encoded.tobytes())  
        img_reader = ImageReader(Image.open(img_bytes))
        
        c.drawImage(img_reader, 50, y_position, width=500, height=80)
        y_position -= 100
        if y_position < 100:
            c.showPage()
            y_position = 700

    c.save()

def process_pdfs(pdf1, pdf2, output_pdf):
    """مقایسه دو PDF و ذخیره تفاوت‌ها فقط در بخش‌های مهم"""
    images1 = pdf_to_images(pdf1, dpi=200)  # کاهش DPI برای سرعت بیشتر
    images2 = pdf_to_images(pdf2, dpi=200)

    all_diff_images = []

    # استفاده از multithreading برای پردازش صفحات به‌صورت هم‌زمان
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(process_page, images1, images2))

    # ادغام تمام تفاوت‌ها در لیستی یکپارچه
    for diff in results:
        all_diff_images.extend(diff)

    save_differences_to_pdf(all_diff_images, output_pdf)

# مثال استفاده:
pdf1_path = "G:\\PythonTest\\TwoPDF\\One.pdf"
pdf2_path = "G:\\PythonTest\\TwoPDF\\Two.pdf"
output_pdf_path = "differences.pdf"

if __name__ == "__main__":
    start_time = time.time()
    process_pdfs(pdf1_path, pdf2_path, output_pdf_path)
    print(f"زمان اجرا: {time.time() - start_time} ثانیه")
