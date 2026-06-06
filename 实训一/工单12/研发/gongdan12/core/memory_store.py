"""Short-term conversation memory backed by Redis."""
import json
import re
from typing import Dict, List


class RedisMemoryStore:
    """Store recent chat messages in Redis lists, one list per chat session."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 6379,
        db: int = 0,
        prefix: str = "rag:memory",
        ttl_seconds: int = 604800,
    ):
        self.prefix = prefix.rstrip(":")
        self.ttl_seconds = ttl_seconds
        self._client = None
        self._enabled = False

        try:
            import redis

            self._client = redis.Redis(
                host=host,
                port=port,
                db=db,
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=2,
            )
            self._client.ping()
            self._enabled = True
            print(f"[MEMORY] RedisMemoryStore ready: {host}:{port}/{db}, prefix={self.prefix}")
        except Exception as e:
            print(f"[MEMORY] RedisMemoryStore disabled: {e}")

    @property
    def is_ready(self) -> bool:
        return self._enabled and self._client is not None

    def _session_id(self, session_id: str) -> str:
        session_id = (session_id or "default").strip()[:128]
        session_id = re.sub(r"\s+", "_", session_id)
        return session_id or "default"

    def key(self, session_id: str) -> str:
        return f"{self.prefix}:{self._session_id(session_id)}"

    def get_history(self, session_id: str, max_turns: int = 10) -> List[Dict]:
        """Return recent messages in OpenAI chat format."""
        if not self.is_ready:
            return []

        max_messages = max(1, max_turns * 2)
        key = self.key(session_id)
        try:
            raw_items = self._client.lrange(key, -max_messages, -1)
            messages = []
            for item in raw_items:
                try:
                    msg = json.loads(item)
                except json.JSONDecodeError:
                    continue
                if msg.get("role") in {"user", "assistant"} and msg.get("content"):
                    messages.append({"role": msg["role"], "content": msg["content"]})
            print(f"[MEMORY] Loaded Redis history: key={key}, messages={len(messages)}")
            return messages
        except Exception as e:
            print(f"[MEMORY] Load Redis history failed: {e}")
            return []

    def get_exact_answer(self, session_id: str, question: str, max_turns: int = 10) -> str:
        """Return the latest assistant answer after the same user question."""
        if not self.is_ready or not question:
            return ""

        question_key = question.strip()
        history = self.get_history(session_id, max_turns)
        for i in range(len(history) - 2, -1, -1):
            current = history[i]
            next_msg = history[i + 1] if i + 1 < len(history) else {}
            if (
                current.get("role") == "user"
                and current.get("content", "").strip() == question_key
                and next_msg.get("role") == "assistant"
                and next_msg.get("content")
            ):
                answer = next_msg["content"].strip()
                print(f"[MEMORY] Exact Redis answer hit: key={self.key(session_id)}", flush=True)
                return answer
        return ""

    def delete_exact_turn(self, session_id: str, question: str) -> int:
        """Delete the latest user/assistant pair for an exact question."""
        if not self.is_ready or not question:
            return 0

        question_key = question.strip()
        key = self.key(session_id)
        try:
            raw_items = self._client.lrange(key, 0, -1)
            if not raw_items:
                return 0

            messages = []
            for item in raw_items:
                try:
                    messages.append(json.loads(item))
                except json.JSONDecodeError:
                    messages.append({"role": "", "content": item})

            delete_at = -1
            for i in range(len(messages) - 2, -1, -1):
                current = messages[i]
                next_msg = messages[i + 1] if i + 1 < len(messages) else {}
                if (
                    current.get("role") == "user"
                    and current.get("content", "").strip() == question_key
                    and next_msg.get("role") == "assistant"
                ):
                    delete_at = i
                    break

            if delete_at < 0:
                return 0

            del raw_items[delete_at:delete_at + 2]
            ttl = self._client.ttl(key)
            pipe = self._client.pipeline()
            pipe.delete(key)
            if raw_items:
                pipe.rpush(key, *raw_items)
                if ttl and ttl > 0:
                    pipe.expire(key, ttl)
            pipe.execute()
            print(f"[MEMORY] Deleted Redis turn: key={key}, question={question_key[:60]}", flush=True)
            return 1
        except Exception as e:
            print(f"[MEMORY] Delete Redis turn failed: {e}", flush=True)
            return 0

    def append_turn(self, session_id: str, question: str, answer: str, max_turns: int = 10):
        """Append one user/assistant turn and keep only the latest max_turns."""
        if not self.is_ready:
            return

        key = self.key(session_id)
        max_messages = max(1, max_turns * 2)
        items = [
            json.dumps({"role": "user", "content": question}, ensure_ascii=False),
            json.dumps({"role": "assistant", "content": answer}, ensure_ascii=False),
        ]
        try:
            pipe = self._client.pipeline()
            pipe.rpush(key, *items)
            pipe.ltrim(key, -max_messages, -1)
            if self.ttl_seconds > 0:
                pipe.expire(key, self.ttl_seconds)
            pipe.execute()
            print(f"[MEMORY] Updated Redis history: key={key}, max_messages={max_messages}")
        except Exception as e:
            print(f"[MEMORY] Update Redis history failed: {e}")
