"""
文生图智能体 - 前端页面 + Qwen API调用 + 图片生成
工单编号：人工智能NLP-Agent 数字人项目-文生图智能体任务
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import requests
import base64
import os
import re
from dotenv import load_dotenv

# 加载环境变量
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

app = Flask(__name__)
CORS(app)  # 启用跨域支持

# 硅基流动 SiliconFlow API配置
QWEN_API_KEY = os.environ.get("SILICONFLOW_API_KEY", "sk-ssdytevbxeeomwszcxpxixqtwxadujrdroqjvnkpwooakplk")
QWEN_API_URL = "https://api.siliconflow.cn/v1/images/generations"
QWEN_MODEL = "Qwen/Qwen-Image-Edit-2509"  # 图像编辑模型

# 输出目录
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 三个方向的精确提示词
direction_map = {
    "right": (
        "将图片中人物面部向右旋转约30度，呈现左侧四分之三视角，"
        "脸部朝向画面右侧，视线向右前方。"
        "严格要求：1）100%保留原图人物所有五官形状、大小、比例，"
        "眼睛颜色、瞳孔大小、眉毛形状完全一致；"
        "2）肤色、肤质、面部纹理与原图保持一致；"
        "3）面部轮廓自然流畅，下颌线、颧骨、鼻梁线条不发生变形；"
        "4）光照方向自然，面部明暗过渡平滑，无明显拼接痕迹；"
        "5）图像清晰锐利，无像素化、模糊或失真；"
        "6）背景与面部自然融合，扩图区域颜色、光影与原图一致"
    ),
    "left": (
        "将图片中人物面部向左旋转约30度，呈现右侧四分之三视角，"
        "脸部朝向画面左侧，视线向左前方。"
        "严格要求：1）100%保留原图人物所有五官形状、大小、比例，"
        "眼睛颜色、瞳孔大小、眉毛形状完全一致；"
        "2）肤色、肤质、面部纹理与原图保持一致；"
        "3）面部轮廓自然流畅，下颌线、颧骨、鼻梁线条不发生变形；"
        "4）光照方向自然，面部明暗过渡平滑，无明显拼接痕迹；"
        "5）图像清晰锐利，无像素化、模糊或失真；"
        "6）背景与面部自然融合，扩图区域颜色、光影与原图一致"
    ),
    "front": (
        "将图片中人物调整为标准正面朝向，双眼正视镜头，面部完全对称。"
        "严格要求：1）100%保留原图人物所有五官形状、大小、比例，"
        "眼睛颜色、瞳孔大小、眉毛形状完全一致；"
        "2）肤色、肤质、面部纹理与原图保持一致；"
        "3）面部轮廓自然流畅，左右脸对称协调，无变形；"
        "4）光照均匀自然，面部明暗过渡平滑，无明显拼接痕迹；"
        "5）图像清晰锐利，无像素化、模糊或失真；"
        "6）背景与面部自然融合，扩图区域颜色、光影与原图一致"
    )
}


@app.route("/")
def index():
    """主页"""
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    """生成图片"""
    try:
        data = request.json
        user_prompt = data.get("prompt", "")
        image_type = data.get("type", "right")
        reference_image = data.get("reference_image", "")  # base64 data URL

        if not user_prompt:
            return jsonify({"error": "请输入生成要求"}), 400

        if not reference_image:
            return jsonify({"error": "请先上传参考图片"}), 400

        # 调用API生成图片
        response = call_qwen_api(user_prompt, image_type, reference_image)

        if response.get("success"):
            return jsonify({
                "success": True,
                "message": response.get("message", ""),
                "image_url": response.get("image_url", "")
            })
        else:
            return jsonify({"error": response.get("error", "生成失败")}), 500

    except Exception as e:
        import traceback
        print(f"[IMAGE ERROR] {e}")
        print(f"[IMAGE ERROR DETAIL] {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


@app.route("/generate_all", methods=["POST"])
def generate_all():
    """一键生成三个方向的图片"""
    try:
        data = request.json
        reference_image = data.get("reference_image", "")

        if not reference_image:
            return jsonify({"error": "请先上传参考图片"}), 400

        results = {}
        for img_type in ["left", "front", "right"]:
            prompt = direction_map.get(img_type, direction_map["front"])
            response = call_qwen_api(prompt, img_type, reference_image)
            results[img_type] = response

        return jsonify({"success": True, "results": results})

    except Exception as e:
        import traceback
        print(f"[IMAGE ERROR] {e}")
        print(f"[IMAGE ERROR DETAIL] {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


def call_qwen_api(prompt, image_type, reference_image=""):
    """调用硅基流动图像编辑API"""
    try:
        headers = {
            "Authorization": f"Bearer {QWEN_API_KEY}",
            "Content-Type": "application/json"
        }

        full_prompt = prompt

        print(f"[API调用] 模型: {QWEN_MODEL}")
        print(f"[API调用] 提示词: {full_prompt[:80]}...")
        print(f"[API调用] 图片: {'有' if reference_image else '无'}")

        # 图片格式：SiliconFlow API要求data URL格式，保留前缀
        img_data = reference_image
        if img_data:
            print(f"[API调用] 图片格式: {'data URL' if img_data.startswith('data:') else '其他'}")

        # 构建请求payload
        payload = {
            "model": QWEN_MODEL,
            "prompt": full_prompt,
            "image": img_data,
            "image_size": "1024x1024"
        }

        response = requests.post(QWEN_API_URL, headers=headers, json=payload, timeout=120)
        response.raise_for_status()

        result = response.json()
        print(f"[API返回] status={response.status_code}")

        # 解析返回的图片URL
        if "images" in result and len(result["images"]) > 0:
            image_url = result["images"][0].get("url", "")
            if image_url:
                return {
                    "success": True,
                    "message": f"图片编辑成功！\n提示词：{full_prompt}",
                    "image_url": image_url
                }

        return {"success": False, "error": "API返回中没有图片数据"}

    except requests.exceptions.RequestException as e:
        error_detail = ""
        if hasattr(e, 'response') and e.response is not None:
            error_detail = f"\n状态码: {e.response.status_code}\n响应: {e.response.text[:500]}"
        return {"success": False, "error": f"API调用失败: {str(e)}{error_detail}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.route("/output/<filename>")
def output_file(filename):
    """提供生成的图片"""
    # 路径遍历防护：验证文件名不包含路径分隔符或..，防止目录穿越攻击
    if not filename or os.sep in filename or '/' in filename or '\\' in filename or '..' in filename:
        return jsonify({"error": "无效的文件名"}), 400
    if not re.match(r'^[\w\-\.]+$', filename):
        return jsonify({"error": "无效的文件名"}), 400
    return send_from_directory(OUTPUT_DIR, filename)


@app.route("/health")
def health():
    """健康检查接口"""
    return jsonify({"code": 0, "message": "success", "data": {"status": "ok"}})


@app.route("/api/v1/image/generate", methods=["POST"])
def api_v1_generate():
    """统一API - 生成单张图片（与 /generate 功能一致，使用统一响应格式）"""
    try:
        data = request.json
        user_prompt = data.get("prompt", "")
        image_type = data.get("type", "right")
        reference_image = data.get("reference_image", "")

        if not user_prompt:
            return jsonify({"code": -1, "message": "请输入生成要求", "data": None}), 400

        if not reference_image:
            return jsonify({"code": -1, "message": "请先上传参考图片", "data": None}), 400

        response = call_qwen_api(user_prompt, image_type, reference_image)

        if response.get("success"):
            return jsonify({
                "code": 0,
                "message": "success",
                "data": {
                    "image_url": response.get("image_url", ""),
                    "detail": response.get("message", "")
                }
            })
        else:
            return jsonify({"code": -1, "message": response.get("error", "生成失败"), "data": None}), 500

    except Exception as e:
        return jsonify({"code": -1, "message": str(e), "data": None}), 500


@app.route("/api/v1/image/generate_all", methods=["POST"])
def api_v1_generate_all():
    """统一API - 一键生成三个方向的图片（与 /generate_all 功能一致，使用统一响应格式）"""
    try:
        data = request.json
        reference_image = data.get("reference_image", "")

        if not reference_image:
            return jsonify({"code": -1, "message": "请先上传参考图片", "data": None}), 400

        results = {}
        for img_type in ["left", "front", "right"]:
            prompt = direction_map.get(img_type, direction_map["front"])
            response = call_qwen_api(prompt, img_type, reference_image)
            results[img_type] = response

        return jsonify({
            "code": 0,
            "message": "success",
            "data": results
        })

    except Exception as e:
        return jsonify({"code": -1, "message": str(e), "data": None}), 500

if __name__ == "__main__":
    print("文生图智能体启动中...")
    print("访问地址: http://localhost:5003")
    app.run(debug=True, host="0.0.0.0", port=5003)
