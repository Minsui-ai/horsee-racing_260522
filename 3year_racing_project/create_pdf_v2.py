import os
from fpdf import FPDF
import re

class MarkdownPDF(FPDF):
    def __init__(self):
        super().__init__()
        # 폰트 경로 (사용자 환경에 맞게 조정 필요, 여기서는 NanumGothic 사용 가정)
        # 폰트 파일이 없으면 기본 폰트로 대체되나 한글이 깨질 수 있음.
        # 이전 턴에서 NanumGothic.ttf가 d:/개인/2. 강의/2. 패스트캠퍼스_이너서클/Antigravity/ICB8project2/racing/NanumGothic.ttf 에 있었음.
        font_path = "NanumGothic.ttf"
        font_bold_path = "NanumGothicBold.ttf"
        if os.path.exists(font_path):
            self.add_font("Nanum", "", font_path)
            if os.path.exists(font_bold_path):
                self.add_font("Nanum", "B", font_bold_path)
            else:
                self.add_font("Nanum", "B", font_path)
            self.add_font("Nanum", "I", font_path)
            self.set_font("Nanum", size=10)
        else:
            # 폰트가 없으면 에러가 나거나 한글이 깨질 수 있으므로 확인 필요
            print("Warning: NanumGothic.ttf not found. PDF may have broken characters.")

    def header(self):
        self.set_font("Nanum", "B", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, "서울 경마장 3개년 데이터 분석 리포트 (심층 전처리 포함)", align="R")
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font("Nanum", "", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

def create_pdf_from_md(md_path, pdf_path):
    pdf = MarkdownPDF()
    pdf.set_left_margin(20)
    pdf.set_right_margin(20)
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    content_width = 170 # 210 - 20 - 20

    with open(md_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            pdf.ln(5)
            continue

        # Header 1
        if line.startswith("# "):
            pdf.set_font("Nanum", "B", 18)
            pdf.multi_cell(content_width, 10, line[2:])
            pdf.ln(5)
        # Header 2
        elif line.startswith("## "):
            pdf.set_font("Nanum", "B", 14)
            pdf.multi_cell(content_width, 10, line[3:])
            pdf.ln(3)
        # Header 3
        elif line.startswith("### "):
            pdf.set_font("Nanum", "B", 12)
            pdf.multi_cell(content_width, 8, line[4:])
            pdf.ln(2)
        # Image
        elif line.startswith("!["):
            img_match = re.search(r'\((.*?)\)', line)
            if img_match:
                img_path = img_match.group(1)
                abs_img_path = os.path.join(os.path.dirname(md_path), img_path)
                if os.path.exists(abs_img_path):
                    # 이미지 크기 조정
                    pdf.image(abs_img_path, w=content_width)
                    pdf.ln(5)
                else:
                    pdf.set_font("Nanum", "I", 10)
                    pdf.cell(content_width, 10, f"[Image not found: {img_path}]")
                    pdf.ln(5)
        # Blockquote / Interpretation
        elif line.startswith("> "):
            pdf.set_font("Nanum", "I", 9)
            pdf.set_text_color(80, 80, 80)
            pdf.multi_cell(content_width, 6, line[2:])
            pdf.set_text_color(0, 0, 0)
            pdf.ln(3)
        # List items
        elif line.startswith("* ") or line.startswith("- "):
            pdf.set_font("Nanum", "", 10)
            pdf.multi_cell(content_width, 7, f"  • {line[2:]}")
        elif re.match(r'^\d+\.', line):
            pdf.set_font("Nanum", "", 10)
            pdf.multi_cell(content_width, 7, f"  {line}")
        # Table
        elif "|" in line and "---" not in line:
            cols = [c.strip() for c in line.split("|") if c.strip()]
            if not cols: continue
            
            # 테이블 헤더 여부 대략적 판단
            is_header = False
            if i + 1 < len(lines) and "---" in lines[i+1]:
                is_header = True
            
            pdf.set_font("Nanum", "B" if is_header else "", 8) # 테이블 글씨 크기 축소
            col_width = content_width / len(cols)
            
            # 테이블 셀 높이 계산을 위해 텍스트 길이 고려 (간이형)
            max_h = 8
            for col in cols:
                # multi_cell로 그릴 때 필요한 높이 계산은 어려우므로 일단 cell 사용
                # 단, 너무 길면 잘리므로 여기서 multi_cell을 쓰는 대신 글씨를 더 줄이거나 너비를 조절해야 함
                pass
            
            for col in cols:
                pdf.cell(col_width, 8, col, border=1, align="C")
            pdf.ln()
        # Normal text
        else:
            pdf.set_font("Nanum", "", 10)
            pdf.multi_cell(content_width, 7, line)

    pdf.output(pdf_path)
    print(f"PDF Created: {pdf_path}")

if __name__ == "__main__":
    md_file = "horserace_ai_data/racing_deep_eda_report_3years_seoul.md"
    pdf_file = "horserace_ai_data/racing_deep_eda_report_3years_seoul.pdf"
    create_pdf_from_md(md_file, pdf_file)
