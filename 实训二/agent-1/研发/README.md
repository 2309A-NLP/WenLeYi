# 小家记账本

人工智能NLP-Agent数字人项目 - 记账本任务工单

## 运行方式

1. 安装依赖：
   ```
   cd /d D:\桌面\agent工单\agent工单1
   pip install -r requirements.txt
   ```

2. 启动服务：
   ```
   python app.py
   ```

3. 打开浏览器访问：http://localhost:5000

## 项目结构

| 文件 | 功能 |
|------|------|
| config.py | 配置文件（数据库连接、LLM接口） |
| database.py | 数据库操作层（增删查改） |
| prompt.py | Prompt模板（系统提示词） |
| tools.py | 工具函数（日期解析、成员映射） |
| agent.py | 核心Agent逻辑（调LLM→解析→分发） |
| app.py | Flask应用入口（路由+启动） |
| templates/index.html | 前端对话页面 |

## 功能说明

- 记账：输入消费/收入信息自动记录到MySQL
- 查询：按日期、成员、类别查询记录
- 汇总：按月统计总支出/收入
- 删除：删除指定记录（需确认）
- 修改：修改已有记录（需确认）
