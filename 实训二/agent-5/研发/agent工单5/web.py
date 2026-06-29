# 工单编号：人工智能NLP-Agent数字人项目-招股书数据问答智能体任务
"""
Web前端 - 招股书数据问答智能体
提供网页界面进行问答测试
"""
import os
import warnings
# 屏蔽所有第三方库的警告信息
warnings.filterwarnings("ignore")

import traceback
import logging
import logging
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from config import OUTPUT_DIR
from query import query_answer

# 配置日志，输出详细错误信息
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# 创建Flask应用
app = Flask(__name__)

# 启用CORS跨域支持
CORS(app)


@app.route("/")
def index():
    """渲染主页"""
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health_check():
    """健康检查接口"""
    return jsonify({"code": 0, "message": "success", "data": {"status": "healthy"}})


@app.route("/api/query", methods=["POST"])
def api_query():
    """
    问答接口 - 接收问题，返回答案
    支持自动重试：如果首次返回"未找到相关信息"，自动重试最多2次
    """
    data = request.get_json()
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"error": "请输入问题"}), 400

    max_retries = 2
    for attempt in range(max_retries + 1):
        try:
            if attempt > 0:
                logger.info(f"第{attempt+1}次尝试...")
                time.sleep(3)  # 重试前等待3秒

            logger.info(f"收到问题: {question}")
            answer = query_answer(question)

            # 如果得到有效答案，直接返回
            if answer and answer != "未找到相关信息":
                logger.info(f"生成答案: {answer[:100]}...")
                return jsonify({"answer": answer})

            # 如果是"未找到相关信息"且还有重试次数，继续
            if attempt < max_retries:
                logger.info(f"答案为空，准备重试...")
                continue
            else:
                # 重试次数用完，返回空答案
                logger.warning(f"重试{max_retries}次后仍无答案")
                return jsonify({"answer": answer or "未找到相关信息"})

        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            logger.error(f"处理出错: {error_msg}")

            # 如果还有重试次数，继续
            if attempt < max_retries:
                logger.info(f"出错，准备重试...")
                continue
            else:
                logger.error(traceback.format_exc())
                return jsonify({"error": error_msg}), 500

    # 不会走到这里，但作为安全返回
    return jsonify({"answer": "未找到相关信息"})


@app.route("/api/v1/prospectus/query", methods=["POST"])
def api_v1_prospectus_query():
    """
    统一格式问答接口 - 接收问题，返回统一格式的响应
    请求体: {"question": "..."}
    成功响应: {"code": 0, "message": "success", "data": {"answer": "...", "question": "..."}}
    失败响应: {"code": -1, "message": "错误描述", "data": null}
    """
    # 解析请求参数
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"code": -1, "message": "请求体必须为JSON格式", "data": None}), 400

    question = data.get("question", "").strip()
    if not question:
        return jsonify({"code": -1, "message": "请输入问题（question字段不能为空）", "data": None}), 400

    # 带重试的问答流程
    max_retries = 2
    for attempt in range(max_retries + 1):
        try:
            if attempt > 0:
                logger.info(f"[统一接口] 第{attempt+1}次尝试...")
                time.sleep(3)

            logger.info(f"[统一接口] 收到问题: {question}")
            answer = query_answer(question)

            # 如果得到有效答案，直接返回
            if answer and answer != "未找到相关信息":
                logger.info(f"[统一接口] 生成答案: {answer[:100]}...")
                return jsonify({
                    "code": 0,
                    "message": "success",
                    "data": {"answer": answer, "question": question}
                })

            # 如果是"未找到相关信息"且还有重试次数，继续
            if attempt < max_retries:
                logger.info(f"[统一接口] 答案为空，准备重试...")
                continue
            else:
                # 重试次数用完，返回空答案（视为成功，答案为"未找到相关信息"）
                logger.warning(f"[统一接口] 重试{max_retries}次后仍无答案")
                return jsonify({
                    "code": 0,
                    "message": "success",
                    "data": {"answer": answer or "未找到相关信息", "question": question}
                })

        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            logger.error(f"[统一接口] 处理出错: {error_msg}")

            if attempt < max_retries:
                logger.info(f"[统一接口] 出错，准备重试...")
                continue
            else:
                logger.error(traceback.format_exc())
                return jsonify({
                    "code": -1,
                    "message": f"服务器内部错误: {error_msg}",
                    "data": None
                }), 500

    # 安全返回（理论上不会到达）
    return jsonify({
        "code": 0,
        "message": "success",
        "data": {"answer": "未找到相关信息", "question": question}
    })


@app.route("/api/batch", methods=["POST"])
def api_batch():
    """
    批量处理接口 - 处理question.json中所有问题
    返回处理进度和结果文件路径
    """
    from main import load_questions, process_questions, save_results
    from config import QUESTION_FILE

    questions = load_questions(QUESTION_FILE)
    results = process_questions(questions)
    output_path = os.path.join(OUTPUT_DIR, "answer.jsonl")
    save_results(results, output_path)
    return jsonify({
        "message": f"处理完成，共 {len(results)} 道题",
        "output_file": output_path,
    })


if __name__ == "__main__":
    print("=" * 60)
    print("招股书数据问答智能体 - Web前端")
    print("访问地址: http://localhost:5005")
    print("工单编号：人工智能NLP-Agent数字人项目-招股书数据问答智能体任务")
    print("=" * 60)
    # 预加载embedding模型和reranker模型
    try:
        from query import get_embedding_model, get_reranker_model
        get_embedding_model()
        get_reranker_model()
        print("  Embedding+Reranker模型预加载完成")
    except Exception as e:
        print(f"  模型预加载失败(不影响启动): {e}")
    app.run(host="0.0.0.0", port=5005, debug=False)
