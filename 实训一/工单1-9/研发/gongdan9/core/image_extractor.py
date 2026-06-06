"""图片提取 + 视觉大模型描述。

从 PDF 文档中提取嵌入的图片，并调用视觉大模型 API 对每张图片生成文字描述。
用于 RAG 系统中的多模态内容理解。
"""
import base64
import time

import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Dict, Optional


class ImageExtractor:
    """从 PDF 提取图片并用视觉大模型生成描述。"""

    def __init__(self, api_key: str, api_base: str, model: str):
        """初始化图片提取器。

        Args:
            api_key: 视觉大模型的 API 密钥
            api_base: API 基础地址（不含末尾斜杠）
            model: 使用的模型名称
        """
        self.api_key = api_key
        self.api_base = api_base.rstrip("/")
        self.model = model
        # [IMAGE] 初始化日志：展示 API 地址和模型名称
        print(f"[IMAGE] 初始化 ImageExtractor | api_base={self.api_base} | model={self.model}")

    def extract_images_from_pdf(self, pdf_path: str) -> List[Dict]:
        """提取 PDF 中所有嵌入的图片。

        遍历 PDF 的每一页，获取其中的嵌入图片数据（字节、格式、尺寸等）。

        Args:
            pdf_path: PDF 文件路径

        Returns:
            包含图片信息字典的列表，每个字典包含：
            image_bytes, ext, page, width, height
        """
        # [IMAGE] 提取开始日志：显示 PDF 路径
        print(f"[IMAGE] 提取图片开始 | pdf_path={pdf_path}")

        doc = fitz.open(pdf_path)
        images = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            # 获取该页所有嵌入图片的引用信息
            for img_idx, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                # 根据 xref 提取图片的实际字节数据
                base_img = doc.extract_image(xref)
                if base_img:
                    images.append({
                        "image_bytes": base_img["image"],
                        "ext": base_img["ext"],
                        "page": page_num + 1,
                        "width": base_img["width"],
                        "height": base_img["height"],
                    })
        doc.close()

        # [IMAGE] 提取完成日志：显示提取到的图片总数
        print(f"[IMAGE] 提取图片完成 | 总图片数={len(images)}")
        return images

    def describe_image(self, image_bytes: bytes, ext: str = "png") -> str:
        """调用视觉大模型对图片进行描述。

        将图片字节数据编码为 base64，发送给视觉大模型 API，
        请求模型生成对图片内容的详细描述。

        Args:
            image_bytes: 图片的原始字节数据
            ext: 图片格式（如 png, jpeg 等）

        Returns:
            模型生成的图片描述文本
        """
        import requests

        # 计算图片大小（KB），用于日志记录
        image_size_kb = len(image_bytes) / 1024
        print(f"[IMAGE] 描述图片开始 | 大小={image_size_kb:.1f}KB | 格式={ext}")

        start_time = time.time()

        # 将图片字节编码为 base64 字符串
        b64 = base64.b64encode(image_bytes).decode("utf-8")

        # 构建 API 请求头
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # 构建请求体：包含图片和文字提示
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "请详细描述这张图片的内容，如果是图表请提取其中的关键数据。"},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/{ext};base64,{b64}"},
                        },
                    ],
                }
            ],
            "max_tokens": 512,
        }

        try:
            # 发送 POST 请求到视觉大模型 API
            resp = requests.post(
                f"{self.api_base}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"].strip()
            elapsed = time.time() - start_time

            # [IMAGE] 描述完成日志：显示响应长度和耗时
            print(f"[IMAGE] 描述图片完成 | 响应长度={len(content)}字符 | 耗时={elapsed:.2f}秒")
            return content

        except Exception as e:
            elapsed = time.time() - start_time
            error_msg = f"[图片描述生成失败: {e}]"
            # [IMAGE] 描述失败日志
            print(f"[IMAGE] 描述图片失败 | 错误={e} | 耗时={elapsed:.2f}秒")
            return error_msg

    def process_pdf(self, pdf_path: str, max_images: int = 20) -> List[Dict]:
        """处理 PDF 中的所有图片，生成描述列表。

        先提取 PDF 中的图片，然后逐个调用视觉大模型进行描述。
        受 max_images 参数限制，最多处理指定数量的图片。

        Args:
            pdf_path: PDF 文件路径
            max_images: 最多处理的图片数量，默认 20 张

        Returns:
            包含图片描述信息的字典列表，每个字典包含：
            description, page, width, height, source
        """
        # [IMAGE] 处理开始：提取图片
        images = self.extract_images_from_pdf(pdf_path)
        total_count = len(images)
        results = []
        success_count = 0

        # 逐个描述图片（最多 max_images 张）
        for idx, img in enumerate(images[:max_images]):
            print(f"[IMAGE] 正在描述第 {idx + 1}/{min(total_count, max_images)} 张图片 (第{img['page']}页)")
            desc = self.describe_image(img["image_bytes"], img["ext"])
            results.append({
                "description": desc,
                "page": img["page"],
                "width": img["width"],
                "height": img["height"],
                "source": Path(pdf_path).name,
            })
            # 统计成功描述的图片数
            if not desc.startswith("[图片描述生成失败"):
                success_count += 1

        # [IMAGE] 处理完成日志：显示汇总信息
        print(f"[IMAGE] PDF图片处理完成 | 总图片数={total_count} | 成功描述={success_count} | 输出结果数={len(results)}")
        return results
