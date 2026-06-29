# 工单编号：人工智能NLP-Agent数字人项目-基金问答智能体任务
"""
Flask Web界面 - 基金数据问答系统
"""
import sys
import os
import warnings

# 抑制警告信息
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, render_template, request, jsonify
from chat import ask

app = Flask(__name__)


@app.route('/')
def index():
    """主页"""
    return render_template('index.html')


@app.route('/api/ask', methods=['POST'])
def api_ask():
    """问答API"""
    data = request.get_json()
    question = data.get('question', '').strip()
    
    if not question:
        return jsonify({'error': '请输入问题'}), 400
    
    try:
        result = ask(question)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("=" * 50)
    print("  基金数据问答系统 - Web界面")
    print("  访问地址: http://localhost:5000")
    print("  工单编号: 人工智能NLP-Agent数字人项目-基金问答智能体任务")
    print("=" * 50)
    app.run(host='127.0.0.1', port=5000, debug=False)
