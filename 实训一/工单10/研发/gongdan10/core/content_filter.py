"""Configurable blocked-word detection."""
from typing import List, Tuple


class ContentFilter:
    """Check user text before retrieval and generation."""

    def __init__(self, enabled: bool = False, words=None, response: str = ""):
        self.enabled = enabled
        self.words = [w.strip() for w in (words or []) if w and w.strip()]
        self.response = response or "问题包含屏蔽词，已拦截。"

    def check(self, text: str) -> Tuple[bool, List[str]]:
        if not self.enabled or not text:
            return False, []
        lowered = text.lower()
        matched = [word for word in self.words if word.lower() in lowered]
        return bool(matched), matched
