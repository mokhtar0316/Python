import sys
import os
import time
import cv2
import numpy as np
from pdf2image import convert_from_path
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog, QLabel, QMessageBox, QProgressBar
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from io import BytesIO
from PIL import Image
import concurrent.futures

class PDFComparerApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        self.label1 = QLabel("📄 انتخاب فایل PDF اول:")
        layout.addWidget(self.label1)

        self.btn1 = QPushButton("انتخاب فایل اول")
        self.btn1.clicked.connect(self.load_pdf1)
        layout.addWidget(self.btn1)

        self.label2 = QLabel("📄 انتخاب فایل PDF دوم:")
        layout.addWidget(self.label2)

        self.btn2 = QPushButton("انتخاب فایل دوم")
        self.btn2.clicked.connect(self.load_pdf2)
        layout.addWidget(self.btn2)

        self.progress = QProgressBar(self)
        layout.addWidget(self.progress)

        self.compare_btn = QPushButton("🔍 مقایسه فایل‌ها")
        self.compare_btn.clicked.connect(self.compare_pdfs)
        layout.addWidget(self.compare_btn)

        self.setLayout(layout)
        self.setWindowTitle("مقایسه فایل‌های PDF")
        self.setGeometry(300, 200, 400, 250)

        self.pdf1_path = None
        self.pdf2_path = None

    def load_pdf1(self):
        file, _ = QFileDialog.getOpenFileName(self, "انتخاب فایل اول", "", "PDF Files (*.pdf)")
        if file:
            self.pdf1_path = file
            self.label1.setText(f"✅ فایل اول: {os.path.basename(file)}")

    def load_pdf2(self):
        file, _ = QFileDialog.getOpenFileName(self, "انتخاب فایل دوم", "", "PDF Files (*.pdf)")
        if file:
            self.pdf2_path = file
            self.label2.setText(f"✅ فایل دوم: {os.path.basename(file)}")

    def compare_pdfs(self):
        if not self.pdf1_path or not self.pdf2_path:
            QMessageBox.warning(self, "خطا", "لطفا دو فایل PDF را انتخاب کنید!")
            return

        output_pdf = "differences.pdf"
        self.progress.setValue(10)

        process_pdfs(self.pdf1_path, self.pdf2_path, output_pdf, self.progress)

        self.progress.setValue(100)
        QMessageBox.information(self, "نتیجه", f"✅ مقایسه انجام شد! فایل خروجی: {output_pdf}")

def pdf_to_images(pdf_path, dpi=200):
    return convert_from_path(pdf_path, dpi=dpi)

def extract_important_regions(image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [cv2.boundingRect(cnt) for cnt in contours if cv2.contourArea(cnt) > 500]

def process_page(img1, img2):
    tables1 = extract_important_regions(img1)
    tables2 = extract_important_regions(img2)

    diff_images = []
    for table1, table2 in zip(tables1, tables2):
        x, y, w, h = table1
        table_img1 = np.array(img1.crop((x, y, x + w, y + h)))
        table_img2 = np.array(img2.crop((x, y, x + w, y + h)))

        if not np.array_equal(table_img1, table_img2):
            diff_images.append(table_img1)

    return diff_images

def save_differences_to_pdf(diff_images, output_pdf):
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

def process_pdfs(pdf1, pdf2, output_pdf, progress):
    images1 = pdf_to_images(pdf1, dpi=200)
    images2 = pdf_to_images(pdf2, dpi=200)

    all_diff_images = []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(process_page, images1, images2))

    for diff in results:
        all_diff_images.extend(diff)

    save_differences_to_pdf(all_diff_images, output_pdf)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = PDFComparerApp()
    window.show()
    sys.exit(app.exec())
