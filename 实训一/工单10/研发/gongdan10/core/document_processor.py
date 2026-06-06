"""文档解析 + 文本清洗 + 按字符分块（带重叠）。

模块功能：
  1. 按字符数切分文本（chunk_text）
  2. 文档清洗流水线（clean_text），包含：
     - 去水印、保密标记、装饰符号
     - 去页眉页脚、页码
     - 去特殊字符 / 乱码
     - 清理表格噪音
     - 提取正文（去目录/声明/附录）
     - CJK 兼容字符归一化
  3. 支持 PDF / DOCX / TXT 文件解析
  4. 主入口 process_documents：扫描目录 -> 解析 -> 清洗 -> 分块
"""

import os
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Optional, Set


# ============================================================
#  分块
# ============================================================

def chunk_text(text: str, chunk_size: int = 900, chunk_overlap: int = 300) -> List[str]:
    """
    按字符数切分文本，相邻块之间保留重叠部分。

    Args:
        text: 待切分文本
        chunk_size: 每块最大字符数
        chunk_overlap: 相邻块重叠字符数

    Returns:
        切分后的文本块列表
    """
    # 空文本直接返回空列表
    if not text or not text.strip():
        return []

    # 重叠数不能超过块大小，否则会无限循环；取块大小的 1/3 作为安全值
    if chunk_overlap >= chunk_size:
        chunk_overlap = chunk_size // 3

    text = text.strip()
    chunks = []
    start = 0

    # 滑动窗口：每次取 chunk_size 长度，然后向前移动 (chunk_size - chunk_overlap)
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - chunk_overlap
        # 如果起始位置已经超过文本末尾，退出循环
        if start >= len(text):
            break

    print(f"  [DOC] chunk_text done: chunk_count={len(chunks)}, chunk_size={chunk_size}, overlap={chunk_overlap}")
    return chunks


# ============================================================
#  文档清洗
# ============================================================

# ---------- 1. 去除水印文字 ----------

# 水印匹配正则列表：内部资料、机密标记、自动生成、装饰符号等
WATERMARK_PATTERNS = [
    r"(?:内部资料|仅供[^\n]*使用|机密|绝密| Confidential)",
    r"(?:未经.*?许可.*?不得转载|版权所有.*?翻印必究)",
    r"(?:本文件由.*?生成|自动生成|Auto[- ]?generated)",
    r"(?:Watermark|WATERMARK)",
    r"(?:★+|☆+|◆+|◇+|▪+|▫+)",           # 装饰符号串
]

def remove_watermarks(text: str) -> str:
    """去除水印、保密标记、装饰符号。"""
    for pattern in WATERMARK_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    return text


# ---------- 2. 去除页眉页脚 ----------

# 页眉页脚匹配正则：页码、纯数字行等
HEADER_FOOTER_PATTERNS = [
    # 页码：1、2、3... / 第1页 / Page 1 / - 1 - / 1/10
    r"(?:^|\n)\s*(?:第?\s*\d+\s*页|[-—]\s*\d+\s*[-—]|\d+\s*/\s*\d+|Page\s+\d+)\s*(?:\n|$)",
    r"(?:^|\n)\s*\d{1,4}\s*(?:\n|$)",       # 独占一行的纯数字（页码）
    # 注意：不要加"重复出现的文本"检测正则！会把正文中的重复关键词（如公司名）误判为页眉，导致大量内容被删除。
]

def remove_headers_footers(text: str) -> str:
    """去除页码、重复页眉等。"""
    for pattern in HEADER_FOOTER_PATTERNS:
        text = re.sub(pattern, "\n", text, flags=re.MULTILINE)
    return text


# ---------- 3. 去除特殊字符 / 乱码 ----------

# 收集 Unicode 0x0000 ~ 0xFFFF 范围内的控制字符（保留 \n \r \t）
CONTROL_CHARS = "".join(
    c for c in (chr(i) for i in range(0x10000))
    if unicodedata.category(c).startswith("C")
    and c not in ("\n", "\r", "\t")
)

# 预编译正则：保留字母/数字/空白/CJK统一汉字/CJK兼容/CJK标点/常见英文标点
# 提升重复匹配性能，避免每次调用都重新编译
_SPECIAL_CHAR_RE = re.compile(
    r"[^\w\s\u4e00-\u9fff\u3400-\u4dbf\u2e80-\u2eff"
    r"\u2f00-\u2fdf\u3000-\u303f\uff00-\uffef\uf900-\ufaff"
    r".,;:!?()\-]"
)

def remove_special_characters(text: str) -> str:
    """去除控制字符、乱码符号、连续特殊符号。"""
    # 第一步：去除控制字符
    text = text.translate(str.maketrans("", "", CONTROL_CHARS))
    # 第二步：去除特殊符号（保留 CJK、字母数字、常见标点）
    text = _SPECIAL_CHAR_RE.sub("", text)
    # 第三步：将非换行空白字符压缩为单个空格
    text = re.sub(r"[^\S\n]+", " ", text)
    return text


# ---------- 4. 去除表格噪音 ----------

def clean_table_noise(text: str) -> str:
    """
    清理 PDF 表格转文本后的常见噪音：
    - 单元格之间的竖线 |
    - 表格对齐产生的多余空格
    - 纯数字行（常为表格数据碎片）
    """
    # 去除竖线分隔符
    text = re.sub(r"\s*\|\s*", " ", text)
    # 去除连续的制表符
    text = re.sub(r"\t+", " ", text)
    # 去除只有数字和小数点的碎片行（表格数据残留）
    text = re.sub(r"(?:^|\n)\s*[\d.,%\s]{3,}\s*(?:\n|$)", "\n", text)
    return text


# ---------- 5. 正文提取（去掉目录/声明/附录） ----------

# 非正文部分的匹配标题
NON_CONTENT_PATTERNS = [
    # 目录
    r"(?:^|\n)\s*(?:目\s*录|Contents?|Table\s+Contents?)\s*(?:\n|$)",
    r"(?:^|\n)\s*\.{3,}\s*\d+\s*(?:\n|$)",              # 目录点线 ... 5
    # 版权声明 / 免责声明
    r"(?:^|\n)\s*(?:版权声明|免责声明|Legal\s+Notice|Disclaimer)\s*[:：]?\s*\n",
    # 前言 / 致谢 / 附录标题
    r"(?:^|\n)\s*(?:前\s*言|致\s*谢|附\s*录|Appendix)\s*[:：]?\s*\n",
]

def extract_main_content(text: str) -> str:
    """
    尝试提取正文内容，去掉目录、声明、附录等非正文部分。
    策略：删除已知的非正文标题及其后续内容。
    """
    # 删除目录区域（从"目录"到下一个大标题）
    text = re.sub(
        r"(?:^|\n)\s*(?:目\s*录|Contents?)\s*\n[\s\S]*?(?=(?:\n\s*(?:第[一二三四五六七八九十\d]+[章节]|Chapter\s+一[、.]|正文|概述)))",
        "\n", text, flags=re.IGNORECASE
    )
    # 删除附录及其之后的所有内容
    text = re.sub(
        r"(?:^|\n)\s*(?:附\s*录|Appendix)\s*[A-Z\d]?[:：]?\s*\n[\s\S]*$",
        "\n", text, flags=re.IGNORECASE
    )
    # 删除版权声明块（通常在开头，连续多行含"版权"关键词）
    text = re.sub(
        r"(?:^|\n)(?:.*(?:版权|版权所有|翻印必究|Confidential).*\n?){2,}",
        "\n", text
    )
    return text


def extract_tables_from_pdf(pdf_path: str) -> List[Dict]:
    """提取 PDF 中的表格，转为 Markdown 格式。"""
    import pdfplumber
    tables = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            for table in page.find_tables():
                data = table.extract()
                if data:
                    md_table = convert_to_markdown(data)
                    tables.append({
                        "content": md_table,
                        "page": page_num + 1,
                        "source": Path(pdf_path).name,
                        "type": "table"
                    })
    return tables

def convert_to_markdown(data: List[List]) -> str:
    """二维数组转 Markdown 表格。"""
    if not data or not data[0]:
        return ""
    header = "| " + " | ".join(str(cell or "") for cell in data[0]) + " |"
    separator = "| " + " | ".join("---" for _ in data[0]) + " |"
    rows = []
    for row in data[1:]:
        rows.append("| " + " | ".join(str(cell or "") for cell in row) + " |")
    return "\n".join([header, separator] + rows)

def ocr_image_rapid(image_bytes: bytes) -> str:
    """使用 RapidOCR 识别图片中的文字（更适合中文）。"""
    try:
        # 优先使用 RapidOCR（基于 ONNX Runtime，中文识别效果好）
        from rapidocr_onnxruntime import RapidOCR
        import io
        ocr = RapidOCR()
        result, _ = ocr(image_bytes)
        if result:
            return "\n".join([line[1] for line in result])
        return ""
    except ImportError:
        # 降级到 pytesseract
        from PIL import Image
        import pytesseract
        import io
        img = Image.open(io.BytesIO(image_bytes))
        return pytesseract.image_to_string(img, lang='chi_sim+eng').strip()

def extract_text_from_scanned_pdf(pdf_path: str) -> str:
    """提取扫描页 PDF 的文字（逐页提取图片 -> OCR）。"""
    import fitz
    doc = fitz.open(pdf_path)
    texts = []
    for page in doc:
        images = page.get_images()
        for img in images:
            xref = img[0]
            base_img = doc.extract_image(xref)
            if base_img:
                text = ocr_image_rapid(base_img["image"])
                if text:
                    texts.append(text)
    doc.close()
    return "\n".join(texts)


# ---------- 6. CJK 兼容字符归一化 ----------

def normalize_cjk(text: str) -> str:
    """
    CJK 兼容字符归一化。
    PDF 抽取的文本可能包含兼容部首（⾏->行、⾦->金、⻓->长），
    统一转换为标准 Unicode 字符，避免 LLM 误判。
    """
    return unicodedata.normalize('NFKC', text)


# ---------- 7. 综合清洗流水线 ----------

def clean_text(
    text: str,
    remove_watermark: bool = True,
    remove_header_footer: bool = True,
    remove_special_chars: bool = True,
    clean_tables: bool = True,
    extract_content: bool = True,
) -> str:
    """
    文档综合清洗流水线。

    Args:
        text: 原始文本
        remove_watermark: 是否去水印
        remove_header_footer: 是否去页眉页脚
        remove_special_chars: 是否去特殊字符
        clean_tables: 是否清理表格噪音
        extract_content: 是否提取正文（去目录/声明/附录）

    Returns:
        清洗后的文本
    """
    if not text:
        return ""

    # 记录清洗前原始长度
    original_len = len(text)

    # ---- 流水线步骤 1：提取正文（去目录/声明/附录） ----
    if extract_content:
        text = extract_main_content(text)
        print(f"  [DOC] clean_text step [extract]: len {original_len} -> {len(text)}")

    # ---- 流水线步骤 2：去除水印、保密标记、装饰符号 ----
    if remove_watermark:
        before_len = len(text)
        text = remove_watermarks(text)
        print(f"  [DOC] clean_text step [watermark]: len {before_len} -> {len(text)}")

    # ---- 流水线步骤 3：去除页眉页脚、页码 ----
    if remove_header_footer:
        before_len = len(text)
        text = remove_headers_footers(text)
        print(f"  [DOC] clean_text step [headers]: len {before_len} -> {len(text)}")

    # ---- 流水线步骤 4：清理表格噪音（竖线、制表符、纯数字碎片行） ----
    if clean_tables:
        before_len = len(text)
        text = clean_table_noise(text)
        print(f"  [DOC] clean_text step [table]: len {before_len} -> {len(text)}")

    # ---- 流水线步骤 5：去除特殊字符 / 乱码符号 ----
    if remove_special_chars:
        before_len = len(text)
        text = remove_special_characters(text)
        print(f"  [DOC] clean_text step [special]: len {before_len} -> {len(text)}")

    # ---- 流水线步骤 6：统一空白字符 ----
    before_len = len(text)
    text = re.sub(r"\n{3,}", "\n\n", text)       # 连续空行压缩为两个换行
    text = re.sub(r"[^\S\n]+", " ", text)         # 非换行空白压缩为单空格
    text = text.strip()
    print(f"  [DOC] clean_text step [whitespace]: len {before_len} -> {len(text)}")

    # ---- 流水线步骤 7：CJK 兼容字符归一化（⾏->行 等） ----
    before_len = len(text)
    text = normalize_cjk(text)
    print(f"  [DOC] clean_text step [cjk]: len {before_len} -> {len(text)}")

    print(f"  [DOC] clean_text done: original={original_len}, final={len(text)}")
    return text


# ============================================================
#  文档解析
# ============================================================

def extract_text_from_pdf(pdf_path: str) -> str:
    """从 PDF 提取纯文本（使用 pdfplumber 逐页提取）。"""
    import pdfplumber
    texts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                texts.append(text)
    return "\n".join(texts)


def extract_text_from_docx(docx_path: str) -> str:
    """从 Word 文档提取纯文本（遍历所有段落，跳过空段落）。"""
    from docx import Document
    doc = Document(docx_path)
    return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])


def extract_text_from_txt(txt_path: str) -> str:
    """从 TXT 文件读取文本（UTF-8 编码）。"""
    with open(txt_path, "r", encoding="utf-8") as f:
        return f.read()


def extract_text(file_path: str) -> str:
    """根据文件后缀自动选择解析方式（PDF/DOCX/TXT）。"""
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    elif ext in (".docx", ".doc"):
        return extract_text_from_docx(file_path)
    elif ext == ".txt":
        return extract_text_from_txt(file_path)
    else:
        raise ValueError(f"不支持的文件格式: {ext}")


# ============================================================
#  主入口
# ============================================================

def _resolve_doc_workers(max_workers: Optional[int], file_count: int) -> int:
    """Return a conservative worker count for file-level document parsing."""
    if file_count <= 1:
        return 1
    if max_workers is None or max_workers <= 0:
        cpu_count = os.cpu_count() or 1
        return max(1, min(file_count, cpu_count, 4))
    return max(1, min(file_count, max_workers))


def _process_document_file(
    file_path: Path,
    file_index: int,
    chunk_size: int,
    chunk_overlap: int,
) -> tuple:
    print(f"[DOC] parsing file {file_index}: {file_path.name}")

    raw_text = extract_text(str(file_path))
    print(f"[DOC] text extraction done: {file_path.name}, len={len(raw_text)}")

    cleaned = clean_text(raw_text)
    chunks = chunk_text(cleaned, chunk_size, chunk_overlap)
    print(f"[DOC] chunks for {file_path.name}: {len(chunks)} chunks")

    file_chunks = [
        {
            "text": chunk,
            "source": file_path.name,
            "chunk_id": i,
        }
        for i, chunk in enumerate(chunks)
    ]
    return file_index, file_chunks


def process_documents(
    docs_dir: str,
    chunk_size: int = 900,
    chunk_overlap: int = 300,
    max_workers: Optional[int] = None,
) -> List[Dict]:
    """
    扫描文档目录，解析 -> 清洗 -> 分块。

    Returns:
        [{"text": chunk, "source": filename, "chunk_id": i}, ...]
    """
    all_chunks = []
    # 支持的文件后缀集合
    supported = {".pdf", ".docx", ".doc", ".txt"}
    docs_path = Path(docs_dir)

    # 打印主入口启动日志
    print(f"[DOC] process_documents start: dir={docs_dir}, chunk_size={chunk_size}, overlap={chunk_overlap}")

    # 如果目录不存在，创建空目录并返回
    if not docs_path.exists():
        print(f"[DOC] docs_dir not found, creating: {docs_dir}")
        os.makedirs(docs_path, exist_ok=True)
        return all_chunks

    file_paths = [
        file_path
        for file_path in sorted(docs_path.iterdir())
        if file_path.suffix.lower() in supported
    ]
    file_count = len(file_paths)
    worker_count = _resolve_doc_workers(max_workers, file_count)
    print(f"[DOC] document workers={worker_count}, files={file_count}")

    if worker_count <= 1:
        for file_index, file_path in enumerate(file_paths, 1):
            _, file_chunks = _process_document_file(
                file_path,
                file_index,
                chunk_size,
                chunk_overlap,
            )
            all_chunks.extend(file_chunks)
    else:
        ordered_chunks = {}
        with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="doc") as executor:
            futures = [
                executor.submit(
                    _process_document_file,
                    file_path,
                    file_index,
                    chunk_size,
                    chunk_overlap,
                )
                for file_index, file_path in enumerate(file_paths, 1)
            ]
            for future in as_completed(futures):
                file_index, file_chunks = future.result()
                ordered_chunks[file_index] = file_chunks

        for file_index in range(1, file_count + 1):
            all_chunks.extend(ordered_chunks.get(file_index, []))

    # 打印最终汇总日志
    print(f"[DOC] process_documents done: total_files={file_count}, total_chunks={len(all_chunks)}")
    return all_chunks
