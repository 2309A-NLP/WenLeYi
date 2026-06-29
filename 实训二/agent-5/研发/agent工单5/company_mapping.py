# 工单编号：人工智能NLP-Agent数字人项目-招股书数据问答智能体任务
"""
公司名-文件映射模块
从每个TXT文件内容中提取公司名，建立公司名到文件名的映射
检索时用于按公司名过滤，避免跨公司混淆
"""
import os
import re
import pickle
from config import TXT_DIR, DATA_DIR


def clean_text(text):
    """
    清理文本：去除字符间的空格和重复字符
    例如："杭杭州州州州中中中中恒恒恒恒" → "杭州中恒电气"
    "浙 江 双 飞 无 油" → "浙江双飞无油"
    """
    # 去除汉字之间的空格
    text = re.sub(r"([\u4e00-\u9fa5])\s+([\u4e00-\u9fa5])", r"\1\2", text)
    text = re.sub(r"([\u4e00-\u9fa5])\s+([\u4e00-\u9fa5])", r"\1\2", text)
    text = re.sub(r"([\u4e00-\u9fa5])\s+([\u4e00-\u9fa5])", r"\1\2", text)

    # 去除重复字符：连续出现2次以上的汉字只保留1次
    text = re.sub(r"([\u4e00-\u9fa5])\1+", r"\1", text)
    return text


def extract_company_name(content):
    """
    从招股书TXT内容中提取公司名称
    处理各种特殊情况：空格、重复字符、公司名位置靠后等
    """
    # 先清理文本（处理空格和重复字符）
    cleaned = clean_text(content[:20000])

    # 匹配模式：XX股份有限公司 或 XX有限公司 或 XX股份有限（截断）
    patterns = [
        r"([\u4e00-\u9fa5]{4,30}(?:股份|集团)(?:有限)?公司)",
        r"([\u4e00-\u9fa5]{4,30}有限公司)",
        r"([\u4e00-\u9fa5]{4,30}股份有限)",  # 处理截断情况
    ]

    # 排除关键词
    exclude_kws = ["承销", "投资风险", "发行人", "创业板", "科创板", "本公司", "证券公司", "保荐", "评估", "会计", "律师", "事务所", "不转让", "委托他人", "回购", "控股股东", "实际控制人", "中介机构"]

    for pattern in patterns:
        matches = re.findall(pattern, cleaned)
        for m in matches:
            if len(m) < 6:
                continue
            # 排除非公司名的匹配
            if any(kw in m for kw in exclude_kws):
                continue
            if m.endswith("公司") or m.endswith("有限"):
                # 清理前缀杂质（取最后一个完整的公司名）
                # 如果匹配结果前面有非公司名内容，提取最后的公司名部分
                final = re.search(r"([\u4e00-\u9fa5]{2,}(?:股份|集团)(?:有限)?公司|[\u4e00-\u9fa5]{2,}有限公司|[\u4e00-\u9fa5]{2,}股份有限)$", m)
                if final:
                    return final.group(1)
                return m

    # 策略2：找"公司名称：XXX"格式
    match = re.search(r"公司名称[：:]\s*([\u4e00-\u9fa5]+公司)", cleaned)
    if match:
        return match.group(1)

    # 策略3：在原始文本中搜索更长范围
    full_cleaned = clean_text(content[:50000])
    for pattern in patterns:
        matches = re.findall(pattern, full_cleaned)
        for m in matches:
            if len(m) < 6:
                continue
            if any(kw in m for kw in ["承销", "投资风险", "发行人", "创业板", "科创板", "本公司", "证券公司", "保荐", "评估", "会计", "律师", "事务所"]):
                continue
            if m.endswith("公司"):
                return m

    return None


def build_company_mapping():
    """
    遍历所有TXT文件，建立公司名到文件名的映射
    包含自动提取和手动补充
    返回：{公司名: 文件名, ...}
    """
    # 手动补充的映射（自动提取识别不到的公司）
    manual_mapping = {
        "北京天宜上佳高新材料股份有限公司": "4bc783ca9fd53beb4be3a79b1712d5e42be209e1.txt",
        "上海中兴派能能源科技股份有限公司": "54d148902b889679830174597830f0d0f22c1073.txt",
        "森赫电梯股份有限公司": "88acba2e98dedd85fdd38192a639dbcc56faf3ed.txt",
        "厦门安妮股份有限公司": "ad2bf1b94db9ec7cb2689ace2daec396d2965dce.txt",
        "海看网络科技（山东）股份有限公司": "afa8c5a4a91c3ecf7bd38a1c1f09b8a68e472909.txt",
        "湖南长远锂科股份有限公司": "eec23035376ae0e339a7643402fdbdccd92ad703.txt",
        "上海维科精密模塑有限公司": "f30bfe8be4ad535d348d74f80eaef8d93b3c8ac5.txt",
        "苏州东微半导体股份有限公司": "f587290218d881e18e88fc1431b022b2c5aca81a.txt",
        # 补充6家缺失公司的映射
        "青海互助青稞酒股份有限公司": "ca1191af7558e0f18966e4df589368a3a5f5e1e6.txt",
        "南京中电联环保股份有限公司": "9c4a118cf576f91aed791fb6fe6926180e2dcc65.txt",
        "深圳市铁汉生态环境股份有限公司": "2389de12d78fe1ca4fa24910e6b1573902098bc3.txt",
        "确成硅化学股份有限公司": "756171248e278806a56171d59c6519a38eac9012.txt",
        "上海真兰仪表科技股份有限公司": "398c8e64f18a13e695b5956122ef2f6a6fd3b274.txt",
        "广东银禧科技股份有限公司": "e6ff749bb533a47173aaca91fe5d44080d9d37b3.txt",
        "宁波立立电子股份有限公司": "42518828d97dd45ac34dc34a5814d18c1ebe9a83.txt",
        # 修复：TXT中公司名被截断为"汉兴图"，手动映射正确的公司名
        "武汉兴图新科电子股份有限公司": "a0d278016e9baded15e0f5b3964563e525b1b787.txt",
    }

    mapping = {}
    for filename in sorted(os.listdir(TXT_DIR)):
        if not filename.endswith(".txt"):
            continue
        filepath = os.path.join(TXT_DIR, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read(20000)  # 读前20000字符

        company = extract_company_name(content)
        if company:
            mapping[company] = filename
            print(f"  {company} -> {filename}")
        else:
            print(f"  ⚠ 无法识别: {filename}")

    # 合并手动补充的映射
    mapping.update(manual_mapping)
    for company, filename in manual_mapping.items():
        if company not in [v for v in mapping.values()]:
            print(f"  [手动补充] {company} -> {filename}")

    return mapping


def find_company_in_question(question, mapping):
    """
    从问题中识别涉及的公司名
    支持全名匹配、部分匹配和简称匹配
    返回：匹配到的公司名列表
    """
    matched = []
    for company in mapping:
        # 全名匹配
        if company in question:
            matched.append(company)
            continue
        # 部分匹配：公司名去掉"股份有限公司"/"有限公司"后检查
        short_name = re.sub(r"(股份|集团)(有限)?公司$", "", company)
        if len(short_name) >= 4 and short_name in question:
            matched.append(company)
            continue
        # 更短的匹配：取前4个字
        if len(company) >= 4 and company[:4] in question:
            matched.append(company)
            continue
        # 简称匹配：去掉省市区前缀和"科技"等通用词
        # 例如："上海真兰仪表科技股份有限公司" -> "真兰仪表"
        simple_name = re.sub(r"^(上海|北京|深圳|广州|南京|杭州|武汉|成都|重庆|天津|苏州|西安|长沙|郑州|青岛|大连|宁波|厦门|珠海|东莞|佛山|无锡|合肥|福州|济南|昆明|南昌|太原|贵阳|兰州|海口|银川|西宁|拉萨|乌鲁木齐|呼和浩特|南宁|石家庄|哈尔滨|长春|沈阳)[市]?", "", company)
        simple_name = re.sub(r"(科技|电子|电气|环保|生物|化学|新材|能源|精密|智能|信息|网络|传媒|文化|教育|医疗|医药|健康|农业|食品|饮料|酒业|建材|钢铁|有色|煤炭|石油|化工|机械|汽车|船舶|航空|航天|兵器|核工业|电力|燃气|水务|环保|园林|生态|环境|检测|认证|咨询|法律|会计|审计|评估|经纪|代理|拍卖|典当|租赁|担保|保险|证券|期货|基金|银行|信托|投资|控股|集团|实业|发展|建设|工程|设计|规划|咨询|管理|服务|商贸|贸易|商业|零售|批发|物流|仓储|运输|快递|邮政|电信|移动|联通|铁通|网通)", "", simple_name)
        # 再去掉"股份有限公司"/"有限公司"等后缀
        simple_name = re.sub(r"(股份|集团)(有限)?公司$", "", simple_name)
        simple_name = re.sub(r"有限公司$", "", simple_name)
        if len(simple_name) >= 3 and simple_name in question:
            matched.append(company)
            continue
    return matched


def save_mapping(mapping):
    """保存映射到文件"""
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, "company_mapping.pkl")
    with open(path, "wb") as f:
        pickle.dump(mapping, f)
    print(f"\n映射已保存: {path}")
    return path


def load_mapping():
    """加载映射"""
    path = os.path.join(DATA_DIR, "company_mapping.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    print("=" * 60)
    print("构建公司名-文件映射")
    print("=" * 60)
    mapping = build_company_mapping()
    save_mapping(mapping)
    print(f"\n共识别 {len(mapping)} 家公司")

    # 测试
    print("\n" + "=" * 60)
    print("测试：从问题中识别公司名")
    print("=" * 60)
    test_questions = [
        "云南沃森生物技术股份有限公司负责产品研发的是什么部门？",
        "湖南长远锂科股份有限公司变更设立时作为发起人的法人有哪些？",
        "东莞勤上光电股份有限公司实际控制人是谁？",
        "武汉兴图新科电子股份有限公司参与制定了哪个技术标准？",
        "湖南国科微电子股份有限公司2014年度营业收入增长多大幅度？",
    ]
    for q in test_questions:
        companies = find_company_in_question(q, mapping)
        print(f"\n问题: {q}")
        print(f"识别到公司: {companies}")
