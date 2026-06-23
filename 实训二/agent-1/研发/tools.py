# 工单编号：人工智能NLP-Agent数字人项目-记账本任务
"""
工具函数：日期解析、成员映射、格式化回复等
"""

from datetime import datetime


# 分类图标映射
CAT_ICONS = {
    '买书': '📚', '图书': '📚', '书': '📚',
    '鞋': '👟', '鞋类': '👟',
    '护肤': '💄', '化妆品': '💄', '美妆': '💄',
    '餐饮': '🍜', '吃饭': '🍜', '食品': '🍜', '零食': '🍪',
    '交通': '🚗', '出行': '🚗',
    '报销': '💰', '工资': '💰', '收入': '💰',
    '旅游': '✈️', '旅行': '✈️',
    '衣服': '👗', '服装': '👗',
    '医疗': '💊', '药': '💊',
    '玩具': '🧸',
    '其他': '📦',
}


def getCatIcon(cat):
    return CAT_ICONS.get(cat, '📦')


# 成员别名映射
MEMBER_MAP = {
    "爸爸": "爸爸", "父亲": "爸爸", "老爸": "爸爸",
    "他爸": "爸爸", "孩儿他爸": "爸爸", "孩儿她爸": "爸爸",
    "妈妈": "妈妈", "母亲": "妈妈", "老妈": "妈妈",
    "她妈": "妈妈", "孩儿他妈": "妈妈", "孩儿她妈": "妈妈",
    "女儿": "女儿", "闺女": "女儿", "孩子": "女儿",
    "娃": "女儿", "我": "女儿", "本人": "女儿",
}


def normalize_member(raw):
    """将口语化的成员称呼标准化，已知的做映射，未知的原样保留"""
    if not raw:
        return None
    raw = raw.strip()
    return MEMBER_MAP.get(raw, raw)  # 未知成员原样保留，不拒绝


def get_today():
    """返回今天的日期对象"""
    return datetime.now()


def format_record(r):
    """格式化单条记录为可读文本"""
    member = r.get("member", "")
    date_str = str(r.get("date", ""))
    type_ = r.get("type", "")
    category = r.get("category", "")
    desc = r.get("description", "")
    amount = r.get("amount", 0)
    rid = r.get("id", "")

    parts = []
    parts.append(date_str)
    parts.append(member)
    if category:
        parts.append(category)
    if desc:
        parts.append(desc)
    parts.append(f"{type_} {amount}元")
    return "，".join(parts)


def format_summary(summary_data):
    """格式化汇总信息"""
    month = summary_data["month"]
    member = summary_data["member"]
    total = summary_data["total"]
    records = summary_data["records"]
    count = summary_data["count"]

    lines = [f"{member} {month} 月总支出/收入 {total} 元，共 {count} 笔"]
    if records:
        lines.append("明细如下：")
        for i, r in enumerate(records, 1):
            lines.append(f"  {i}. {format_record(r)}")
    return "\n".join(lines)


# 当前项目不需要这些工具函数（已删除旧内容）


def format_records_list(records):
    """格式化多条记录列表"""
    if not records:
        return "没有找到相关记录"
    lines = []
    for i, r in enumerate(records, 1):
        lines.append(f"{i}. {format_record(r)}")
    return "\n".join(lines)
