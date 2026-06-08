"""LLM 调用 -- OpenAI 兼容接口，支持图片输入 + 流式输出。

本模块提供 LLMClient 类，封装对 OpenAI 兼容 API 的调用逻辑。
支持普通对话和流式输出两种模式，同时支持图片输入。
所有关键步骤均通过 print() 输出调试日志，前缀为 [LLM]，便于排查问题。
"""

import base64
import json
import time
import requests
from typing import List, Dict, Optional, Generator


def _mask_key(key: str) -> str:
    """对 API Key 进行脱敏处理，仅显示前 6 位和后 4 位，中间用 **** 替代。

    例如: sk-abc123xyz789 -> sk-abc****789
    """
    if len(key) <= 10:
        return key[:3] + "****" + key[-1:]
    return key[:6] + "****" + key[-4:]


class LLMClient:
    """调用 OpenAI 兼容接口的 LLM 客户端。

    通过 requests 直接发送 HTTP 请求，不依赖 openai SDK。
    支持普通对话 (chat) 和流式输出 (stream) 两种调用方式。
    """

    def __init__(self, api_key: str, api_base: str, model: str, timeout: float = 30):
        """初始化 LLM 客户端，保存连接参数。

        参数:
            api_key:  API 密钥，用于鉴权
            api_base: API 基础地址，如 https://api.openai.com/v1
            model:    模型名称，如 gpt-4o, gpt-3.5-turbo
            timeout:  请求超时时间（秒），默认 30 秒
        """
        self.api_key = api_key
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.timeout = timeout
        # 打印初始化信息，API Key 脱敏显示
        print(f"[LLM] 初始化客户端 | api_base={self.api_base} | model={self.model} | timeout={self.timeout}s | api_key={_mask_key(self.api_key)}")

    def _build_messages(self, prompt, system_prompt="", history=None, image_base64=""):
        """构建发送给 LLM 的消息列表。

        根据参数组装消息数组：
        1. 如果有 system_prompt，添加系统消息
        2. 如果有 history，追加历史对话
        3. 如果有图片 (image_base64)，将用户消息构建为多模态格式（文本+图片）
        4. 否则，用户消息为纯文本

        参数:
            prompt:        用户输入的文本
            system_prompt: 系统提示词，用于设定模型角色和行为
            history:       历史对话列表，格式为 [{"role": "user/assistant", "content": "..."}]
            image_base64:  图片的 base64 编码，非空时启用图片输入

        返回:
            构建好的消息列表，可直接传入 API
        """
        messages = []

        # 添加系统提示词（如果提供）
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # 追加历史对话记录
        if history:
            messages.extend(history)

        # 构建用户消息：根据是否有图片选择不同格式
        if image_base64:
            # 多模态消息：包含文本和图片两个部分
            user_content = [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                },
            ]
        else:
            # 纯文本消息
            user_content = prompt

        messages.append({"role": "user", "content": user_content})
        return messages

    def _get_headers(self):
        """构造 HTTP 请求头，包含鉴权信息。

        返回:
            包含 Authorization 和 Content-Type 的字典
        """
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def chat(
        self,
        prompt: str,
        system_prompt: str = "",
        history: List[Dict] = None,
        image_base64: str = "",
        temperature: float = 0.3,
        max_tokens: int = 256,
    ) -> str:
        """发送非流式对话请求，等待完整响应后返回。

        流程：
        1. 构建消息列表
        2. 发送 POST 请求到 /chat/completions
        3. 解析响应并返回生成的文本

        参数:
            prompt:        用户输入文本
            system_prompt: 系统提示词
            history:       历史对话列表
            image_base64:  图片 base64 编码（可选）
            temperature:   温度参数，控制生成随机性，默认 0.3
            max_tokens:    最大生成 token 数，默认 2048

        返回:
            LLM 生成的文本响应
        """
        # 构建消息
        messages = self._build_messages(prompt, system_prompt, history, image_base64)

        # 记录请求开始信息：prompt 长度、历史条数、是否包含图片
        has_image = bool(image_base64)
        history_len = len(history) if history else 0
        print(f"[LLM] chat 开始 | prompt长度={len(prompt)} | 历史条数={history_len} | 包含图片={has_image}")

        # 构建请求体
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        # 记录请求开始时间
        start_time = time.time()

        try:
            # 发送 POST 请求到 OpenAI 兼容接口
            resp = requests.post(
                f"{self.api_base}/chat/completions",
                headers=self._get_headers(),
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()

            # 解析响应内容
            result = resp.json()["choices"][0]["message"]["content"].strip()

            # 计算响应耗时并记录成功日志
            elapsed = time.time() - start_time
            print(f"[LLM] chat 成功 | 响应长度={len(result)} | 耗时={elapsed:.2f}s")

            return result

        except Exception as e:
            # 记录失败日志，包含错误信息
            elapsed = time.time() - start_time
            print(f"[LLM] chat 失败 | 错误={e} | 耗时={elapsed:.2f}s")
            return f"LLM 调用失败: {e}"

    def stream(
        self,
        prompt: str,
        system_prompt: str = "",
        history: List[Dict] = None,
        image_base64: str = "",
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> Generator[str, None, None]:
        """流式对话请求，逐 chunk 返回生成的文本片段。

        通过 SSE (Server-Sent Events) 协议接收流式响应。
        每次 yield 一个文本片段，调用方需要循环消费。

        参数:
            prompt:        用户输入文本
            system_prompt: 系统提示词
            history:       历史对话列表
            image_base64:  图片 base64 编码（可选）
            temperature:   温度参数，默认 0.3
            max_tokens:    最大生成 token 数，默认 2048

        产出:
            逐个 yield 生成的文本片段
        """
        # 构建消息
        messages = self._build_messages(prompt, system_prompt, history, image_base64)

        # 记录流式请求开始信息
        has_image = bool(image_base64)
        history_len = len(history) if history else 0
        print(f"[LLM] stream 开始 | prompt长度={len(prompt)} | 历史条数={history_len} | 包含图片={has_image}")

        # 构建请求体，stream=True 开启流式输出
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }

        # 用于统计收到的 chunk 数量
        chunk_count = 0

        try:
            # 发送流式 POST 请求
            resp = requests.post(
                f"{self.api_base}/chat/completions",
                headers=self._get_headers(),
                json=payload,
                timeout=self.timeout,
                stream=True,
            )
            resp.raise_for_status()

            # 逐行读取 SSE 流
            for line in resp.iter_lines():
                if not line:
                    continue
                line = line.decode("utf-8")

                # SSE 协议中数据以 "data: " 前缀开头
                if line.startswith("data: "):
                    data = line[6:]

                    # "[DONE]" 表示流结束
                    if data.strip() == "[DONE]":
                        break

                    try:
                        # 解析 JSON 数据，提取 delta 中的 content
                        chunk = json.loads(data)
                        choices = chunk.get("choices", [])
                        if not choices:
                            continue
                        delta = choices[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            chunk_count += 1
                            yield content
                    except json.JSONDecodeError:
                        # JSON 解析失败时跳过该行
                        continue

            # 记录流式完成日志，显示 chunk 数量
            print(f"[LLM] stream 完成 | chunk数量={chunk_count}")

        except Exception as e:
            # 记录流式请求失败日志
            print(f"[LLM] stream 失败 | 错误={e} | 已接收chunk={chunk_count}")
            yield f"LLM 调用失败: {e}"
