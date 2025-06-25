#!/usr/bin/env python
# coding: utf-8


import pdfplumber
import fitz
import os
import csv
from PIL import Image
import io
import pytesseract
import numpy as np
import cv2



pytesseract.pytesseract.tesseract_cmd = r"..."

def get_output_dir(pdf_path):
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    output_dir = os.path.join("output", base_name)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def extract_images(pdf_path, output_dir, min_size=10000, min_dpi=72):
    pdf = fitz.open(pdf_path)
    image_count = 0

    for page_num in range(len(pdf)):
        try:
            page = pdf[page_num]
            image_list = page.get_images(full=True)
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = pdf.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]

                img_pil = Image.open(io.BytesIO(image_bytes))
                width, height = img_pil.size
                dpi = img_pil.info.get("dpi", (72, 72))[0]
                if width * height < min_size or dpi < min_dpi:
                    continue

                image_path = os.path.join(output_dir, f"page_{page_num+1}_img_{img_index+1}.{image_ext}")
                with open(image_path, "wb") as f:
                    f.write(image_bytes)
                image_count += 1
        except Exception as e:
            print(f"[Error] Extracting image on page {page_num + 1}: {e}")
    
    pdf.close()
    return image_count

def preprocess_image_for_ocr(image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 15, 2)
    return Image.fromarray(thresh)

def extract_tables_with_ocr(pdf_path, output_dir, dpi=300):
    table_count = 0
    pdf = fitz.open(pdf_path)

    with pdfplumber.open(pdf_path) as pdf_plumber:
        for page_num in range(len(pdf)):
            try:
                page = pdf[page_num]
                matrix = fitz.Matrix(dpi / 72, dpi / 72)
                pix = page.get_pixmap(matrix=matrix)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

                page_plumber = pdf_plumber.pages[page_num]
                tables = page_plumber.extract_tables({
                    "vertical_strategy": "lines_strict",
                    "horizontal_strategy": "lines_strict",
                    "snap_tolerance": 3,
                    "join_tolerance": 3,
                })

                if tables:
                    for table_idx, table in enumerate(tables):
                        table_path = os.path.join(output_dir, f"page_{page_num+1}_table_{table_idx+1}.csv")
                        with open(table_path, "w", newline="", encoding="utf-8") as f:
                            writer = csv.writer(f)
                            writer.writerows(table)
                        table_count += 1
                else:
                    processed_img = preprocess_image_for_ocr(img)
                    table_data = pytesseract.image_to_data(processed_img, output_type=pytesseract.Output.DICT)

                    if not table_data.get("text") or not table_data["text"]:
                        continue

                    lines = []
                    current_line = []
                    last_top = None

                    for i, text in enumerate(table_data["text"]):
                        if not text.strip():
                            continue
                        top = table_data["top"][i]
                        if last_top is None or abs(top - last_top) > 20:
                            if current_line:
                                lines.append(current_line)
                            current_line = [text]
                            last_top = top
                        else:
                            current_line.append(text)

                    if current_line:
                        lines.append(current_line)

                    if lines:
                        table_path = os.path.join(output_dir, f"page_{page_num+1}_table_ocr_{table_count+1}.csv")
                        with open(table_path, "w", newline="", encoding="utf-8") as f:
                            writer = csv.writer(f)
                            writer.writerows(lines)
                        table_count += 1

            except Exception as e:
                print(f"[Error] Extracting table on page {page_num + 1}: {e}")

    pdf.close()
    return table_count

def main(pdf_path):
    if not os.path.exists(pdf_path):
        print(f"[Error] PDF file not found: {pdf_path}")
        return

    output_dir = get_output_dir(pdf_path)
    print(f"[Info] Output directory: {output_dir}")

    img_count = extract_images(pdf_path, output_dir)
    print(f"[Info] Extracted {img_count} images")

    table_count = extract_tables_with_ocr(pdf_path, output_dir)
    print(f"[Info] Extracted {table_count} tables")

if __name__ == "__main__":
    pdf_path = r"..."
    main(pdf_path)

