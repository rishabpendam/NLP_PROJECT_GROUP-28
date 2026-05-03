"""
agents/llm_norep_groq.py
LLM-NoRep agent backed by Groq (free tier — llama-3.3-70b-versatile).

Requires: pip install groq
Set GROQ_API_KEY in .env
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


class LLMNoRepGroqAgent(BaseAgent):
    """LLM-NoRep agent using Groq. Reputation is never passed to the prompt."""

    def __init__(self, role: str, model: str = "llama-3.3-70b-versatile", temperature: float = 0.7):
        super().__init__(role)
        self.model = model
        self.temperature = temperature
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from groq import Groq
            except ImportError:
                raise ImportError("Install groq: pip install groq")
            api_key = os.environ.get("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY not set in environment")
            self._client = Groq(api_key=api_key)
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
            chat = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                ],
                temperature=self.temperature,
                max_tokens=512,
            )
            return chat.choices[0].message.content
        except Exception as e:
            return f"THOUGHT: Groq error: {e}\nTALK: I need a moment.\nACTION: QUIT"
