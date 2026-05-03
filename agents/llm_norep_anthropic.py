"""
agents/llm_norep_anthropic.py
LLM-NoRep agent backed by Anthropic Claude (paid).

Requires: pip install anthropic
Set ANTHROPIC_API_KEY in .env
"""
from __future__ import annotations
import os
from typing import Optional

from agents.base_agent import BaseAgent
from agents.prompts import (
    BUYER_SYSTEM, SELLER_SYSTEM,
    build_buyer_user_message, build_seller_user_message,
)
from core.schema import Scenario
from core.dialogue import DialogueHistory

DEFAULT_MODEL = "claude-haiku-4-5-20251001"   # cost-efficient for experiments


class LLMNoRepAnthropicAgent(BaseAgent):
    """LLM-NoRep agent using Anthropic Claude."""

    def __init__(self, role: str, model: str = DEFAULT_MODEL, temperature: float = 0.7):
        super().__init__(role)
        self.model = model
        self.temperature = temperature
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError:
                raise ImportError("Install anthropic: pip install anthropic")
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not set in environment")
            self._client = anthropic.Anthropic(api_key=api_key)
        return self._client

    def act(
        self,
        scenario: Scenario,
        history: DialogueHistory,
        rep_score: Optional[float],
        opponent_answer: Optional[str],
    ) -> str:
        system_prompt = BUYER_SYSTEM if self.role == "buyer" else SELLER_SYSTEM

        if self.role == "buyer":
            user_msg = build_buyer_user_message(
                scenario, history, rep_score=None, opponent_answer=opponent_answer
            )
        else:
            user_msg = build_seller_user_message(
                scenario, history, rep_score=None, opponent_answer=opponent_answer
            )

        try:
            client = self._get_client()
            response = client.messages.create(
                model=self.model,
                max_tokens=512,
                system=system_prompt,
                messages=[{"role": "user", "content": user_msg}],
            )
            return response.content[0].text
        except Exception as e:
            return f"THOUGHT: Anthropic error: {e}\nTALK: I need a moment.\nACTION: QUIT"
