# -*- coding: utf-8 -*-
"""
Milvus向量搜索模块
功能：把用户问题转成向量，在Milvus向量数据库中找语义最相似的问答对
原理：
1. 用BGE/m3e嵌入模型把文本转成768维的向量
2. 在Milvus中用余弦相似度搜索最相似的向量
3. 返回匹配的问答对和相似度分数
注意：AutoDL上如果没装pymilvus，此模块会不可用，系统自动降级到jieba搜索
"""

import os
import logging
import numpy as np
from transformers import AutoModel, AutoTokenizer
import torch

logger = logging.getLogger(__name__)

# 嵌入模型路径（m3e-base，把文本转成768维向量）
MODEL_PATH = "D:/桌面/模型/m3e-base"
# Milvus集合名称（存储金融知识向量的表）
COLLECTION_NAME = "financial_knowledge"

_embed_model = None  # 嵌入模型（单例，只加载一次）
_tokenizer = None    # 分词器


def _load_model():
    """
    加载嵌入模型（惰性加载，只在第一次调用时加载）
    模型作用：把中文文本转成768维的数值向量，用于计算语义相似度
    """
    global _embed_model, _tokenizer
    if _embed_model is not None:
        return _embed_model, _tokenizer
    logger.info("加载嵌入模型...")
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    _embed_model = AutoModel.from_pretrained(MODEL_PATH)
    _embed_model.eval()  # 设为评估模式（关闭dropout等训练行为）
    logger.info("嵌入模型加载完成")
    return _embed_model, _tokenizer


def embed_texts(texts):
    """
    批量文本编码：把文本列表转成向量矩阵
    原理：
    1. 用tokenizer把文本转成token IDs
    2. 送入BERT模型得到每个token的隐藏状态
    3. 取[CLS] token的输出作为整句话的向量表示
    返回：shape为 (batch_size, 768) 的numpy数组
    """
    model, tokenizer = _load_model()
    # tokenize：把文字转成模型能理解的数字序列
    encoded = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
    with torch.no_grad():  # 不计算梯度，节省内存
        output = model(**encoded)
    # 取第一个token（[CLS]）的输出作为整句话的向量
    return output.last_hidden_state[:, 0, :].numpy()


def vector_search(query, top_k=5, score_threshold=0.6):
    """
    【核心函数】向量搜索：在Milvus中找与问题最相似的知识条目
    参数：
        query: 用户的问题文本
        top_k: 返回最相似的前k条结果
        score_threshold: 相似度阈值，低于此分数的结果会被过滤掉
    返回：匹配的问答对列表，每个包含question/answer/category/score
    """
    try:
        from pymilvus import connections, Collection

        # 连接Milvus向量数据库
        connections.connect(host="127.0.0.1", port=19530)
        col = Collection(COLLECTION_NAME)
        col.load()  # 把集合加载到内存

        # 用嵌入模型把用户问题转成向量
        vec = embed_texts([query])

        # 在Milvus中搜索最相似的向量
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 16}}
        results = col.search(
            data=vec,                    # 查询向量
            anns_field="vector",         # 向量字段名
            param=search_params,         # 搜索参数
            limit=top_k,                 # 返回前k条
            output_fields=["question", "answer", "category", "source"],  # 返回的字段
        )

        matches = []
        if results:
            for hits in results:
                for hit in hits:
                    if hit.score >= score_threshold:  # 只保留超过阈值的结果
                        entity = hit.entity
                        # 兼容不同版本pymilvus获取字段的方式
                        fields = entity.get("fields") if hasattr(entity, 'get') else {}
                        if not fields:
                            fields = {}
                            for fn in ["question", "answer", "category", "source"]:
                                try:
                                    fields[fn] = entity.get(fn)
                                except:
                                    fields[fn] = ""

                        matches.append({
                            "question": fields.get("question", "") or "",
                            "answer": fields.get("answer", "") or "",
                            "category": fields.get("category", "") or "",
                            "source": fields.get("source", "") or "",
                            "score": round(hit.score, 4),  # 保留4位小数的相似度分数
                        })
        return matches
    except Exception as e:
        logger.warning(f"向量搜索失败: {e}")
        return None  # 返回None表示失败，让调用方降级到jieba搜索
