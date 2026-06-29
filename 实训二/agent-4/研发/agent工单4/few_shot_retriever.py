# 工单编号：人工智能NLP-Agent数字人项目-基金问答智能体任务
"""
Milvus向量检索模块 - 动态Few-shot示例检索
"""
import json
import os
from config import (
    MILVUS_HOST, MILVUS_PORT, MILVUS_COLLECTION,
    EMBEDDING_MODEL_PATH, FEW_SHOT_COUNT, QUESTION_FILE
)

# Milvus连接
_milvus_client = None
_embedding_model = None


def get_embedding_model():
    """获取embedding模型"""
    global _embedding_model
    if _embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _embedding_model = SentenceTransformer(EMBEDDING_MODEL_PATH)
            print(f"Embedding模型加载完成: {EMBEDDING_MODEL_PATH}")
        except Exception as e:
            print(f"Embedding模型加载失败: {e}")
            return None
    return _embedding_model


def get_milvus_client():
    """获取Milvus客户端"""
    global _milvus_client
    if _milvus_client is None:
        try:
            from pymilvus import connections, utility
            connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
            _milvus_client = True
            print(f"Milvus连接成功: {MILVUS_HOST}:{MILVUS_PORT}")
        except Exception as e:
            print(f"Milvus连接失败: {e}")
            return None
    return _milvus_client


def create_collection():
    """创建Milvus集合"""
    from pymilvus import Collection, FieldSchema, CollectionSchema, DataType
    
    # 定义schema
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="question", dtype=DataType.VARCHAR, max_length=2000),
        FieldSchema(name="sql", dtype=DataType.VARCHAR, max_length=2000),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768)
    ]
    
    schema = CollectionSchema(fields, description="基金问答Few-shot示例")
    collection = Collection(MILVUS_COLLECTION, schema)
    
    # 创建索引
    index_params = {
        'metric_type': 'L2',
        'index_type': 'IVF_FLAT',
        'params': {'nlist': 128}
    }
    collection.create_index("embedding", index_params)
    
    print(f"集合 {MILVUS_COLLECTION} 创建成功")
    return collection


def load_fewshot_examples():
    """加载Few-shot示例数据"""
    # 内置的高质量Q-SQL示例
    examples = [
        {
            "question": "景顺长城中短债债券C基金在20210331的季报里，前三大持仓占比的债券名称是什么?",
            "sql": "SELECT 债券名称 FROM 基金债券持仓明细 WHERE 基金代码 = (SELECT 基金代码 FROM 基金基本信息 WHERE 基金简称 = '景顺长城中短债债券C') AND 持仓日期 = '20210331' AND 报告类型 = '季报' AND 第N大重仓股 <= 3 ORDER BY 第N大重仓股"
        },
        {
            "question": "20210415日，建筑材料一级行业涨幅超过5%的股票数量是多少?",
            "sql": "SELECT COUNT(DISTINCT a.股票代码) FROM A股票日行情表 a JOIN A股公司行业划分表 ind ON a.股票代码 = ind.股票代码 WHERE ind.一级行业名称 = '建筑材料' AND ind.交易日期 = '20210415' AND a.交易日 = '20210415' AND (a.收盘价(元) - a.昨收盘(元)) / a.昨收盘(元) * 100 > 5"
        },
        {
            "question": "基金000006的名称是什么?",
            "sql": "SELECT 基金简称 FROM 基金基本信息 WHERE 基金代码 = '000006'"
        },
        {
            "question": "20210331季度报告中，持有贵州茅台股票数量最多的基金是哪个?",
            "sql": "SELECT 基金简称, 数量 FROM 基金股票持仓明细 WHERE 股票名称 = '贵州茅台' AND 持仓日期 = '20210331' ORDER BY 数量 DESC LIMIT 1"
        },
        {
            "question": "2021年，管理费率最高的基金是哪个?",
            "sql": "SELECT 基金代码, 基金简称, 管理费率 FROM 基金基本信息 ORDER BY CAST(REPLACE(管理费率, '%', '') AS REAL) DESC LIMIT 1"
        },
        {
            "question": "20210304日，一级行业为非银金融的股票的成交量合计是多少?",
            "sql": "SELECT SUM(h.成交量(股)) FROM A股票日行情表 h JOIN A股公司行业划分表 ind ON h.股票代码 = ind.股票代码 WHERE h.交易日 = '20210304' AND ind.交易日期 = '20210304' AND ind.行业划分标准 = '中信行业分类' AND ind.一级行业名称 = '非银金融'"
        },
        {
            "question": "2021年度，688338股票涨停天数?",
            "sql": "SELECT COUNT(*) FROM A股票日行情表 a1 WHERE a1.股票代码 = '688338' AND a1.交易日 >= '20210101' AND a1.交易日 <= '20211231' AND EXISTS (SELECT 1 FROM A股票日行情表 a2 WHERE a2.股票代码 = a1.股票代码 AND a2.交易日 = (SELECT MAX(交易日) FROM A股票日行情表 WHERE 股票代码 = a1.股票代码 AND 交易日 < a1.交易日) AND (a1.收盘价(元) / a2.收盘价(元) - 1) >= 0.098)"
        },
        {
            "question": "基金000001在2021年1月的日均净值是多少?",
            "sql": "SELECT AVG(单位净值) FROM 基金日行情表 WHERE 基金代码 = '000001' AND 交易日期 >= '20210101' AND 交易日期 <= '20210131'"
        },
        {
            "question": "20210331，个人投资者持有份额占比最高的基金是哪个?",
            "sql": "SELECT 基金简称, 个人投资者持有的基金份额占总份额比例 FROM 基金份额持有人结构 WHERE 截止日期 LIKE '2021-03-31%' ORDER BY 个人投资者持有的基金份额占总份额比例 DESC LIMIT 1"
        },
        {
            "question": "2021年1月，基金000006的资产净值是多少?",
            "sql": "SELECT 资产净值 FROM 基金日行情表 WHERE 基金代码 = '000006' AND 交易日期 >= '20210101' AND 交易日期 <= '20210131' LIMIT 1"
        }
    ]
    return examples


def init_milvus():
    """初始化Milvus：创建集合并导入示例数据"""
    from pymilvus import Collection
    
    client = get_milvus_client()
    if client is None:
        return False
    
    # 检查集合是否已存在
    from pymilvus import utility
    if utility.has_collection(MILVUS_COLLECTION):
        print(f"集合 {MILVUS_COLLECTION} 已存在")
        collection = Collection(MILVUS_COLLECTION)
        collection.load()
        return True
    
    # 创建集合
    collection = create_collection()
    
    # 加载示例数据
    examples = load_fewshot_examples()
    
    # 获取embedding
    model = get_embedding_model()
    if model is None:
        print("无法加载embedding模型")
        return False
    
    questions = [ex['question'] for ex in examples]
    embeddings = model.encode(questions).tolist()
    
    # 插入数据
    collection.insert([
        questions,
        [ex['sql'] for ex in examples],
        embeddings
    ])
    
    collection.load()
    print(f"已导入 {len(examples)} 条Few-shot示例")
    return True


def search_similar_questions(question, top_k=5):
    """
    检索最相似的问题
    
    Args:
        question: 用户问题
        top_k: 返回数量
    
    Returns:
        list of dicts: [{'question': str, 'sql': str, 'distance': float}]
    """
    from pymilvus import Collection
    
    model = get_embedding_model()
    if model is None:
        return []
    
    client = get_milvus_client()
    if client is None:
        return []
    
    try:
        collection = Collection(MILVUS_COLLECTION)
        collection.load()
        
        # 编码问题
        query_embedding = model.encode([question]).tolist()
        
        # 搜索
        results = collection.search(
            data=query_embedding,
            anns_field="embedding",
            param={"metric_type": "L2", "params": {"nprobe": 16}},
            limit=top_k,
            output_fields=["question", "sql"]
        )
        
        similar = []
        for hits in results:
            for hit in hits:
                similar.append({
                    'question': hit.entity.get('question'),
                    'sql': hit.entity.get('sql'),
                    'distance': hit.distance
                })
        
        return similar
    
    except Exception as e:
        print(f"Milvus搜索失败: {e}")
        return []


if __name__ == '__main__':
    # 测试
    print("初始化Milvus...")
    if init_milvus():
        print("\n测试检索...")
        results = search_similar_questions("基金000006的名称", top_k=3)
        for r in results:
            print(f"  相似度: {r['distance']:.2f}")
            print(f"  问题: {r['question'][:60]}")
            print(f"  SQL: {r['sql'][:60]}")
            print()
