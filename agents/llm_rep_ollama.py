"""
agents/llm_rep_ollama.py
LLM-Rep agent backed by Ollama (fully local, no API key needed).

This is the REPUTATION-AWARE agent (Section 4.3 — LLM-Rep):
  - Receives private parameters, dialogue history, AND the opponent's reputation score.
  - Uses the reputation signal to calibrate trust and negotiation aggression.
"""
from __future__ import annotations
from typing import Optional
import requests

from agents.base_agent import BaseAgent
from agents.prompts import (
    BUYER_SYSTEM, SELLER_SYSTEM,
    build_buyer_user_message, build_seller_user_message,
)
from core.schema import Scenario
from core.dialogue import DialogueHistory

OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "llama3"


class LLMRepOllamaAgent(BaseAgent):
    """
    LLM-Rep agent using Ollama.
    Passes the opponent's reputation score to the prompt when rep_visible=True.
    """

    def __init__(self, role: str, model: str = DEFAULT_MODEL, temperature: float = 0.7):
        super().__init__(role)
        self.model = model
        self.temperature = temperature

    def act(
        self,
        scenario: Scenario,
        history: DialogueHistory,
        rep_score: Optional[float],
        opponent_answer: Optional[str],
    ) -> str:
        system_prompt = BUYER_SYSTEM if self.role == "buyer" else SELLER_SYSTEM

        # Rep agent DOES pass rep_score (if scenario says it should be visible)
        effective_rep = rep_score if scenario.rep_visible else None

        if self.role == "buyer":
            user_msg = build_buyer_user_message(
                scenario, history,
                rep_score=effective_rep,
                opponent_answer=opponent_answer,
                include_rep_reasoning=True,
            )
        else:
            user_msg = build_seller_user_message(
                scenario, history,
                rep_score=effective_rep,
                opponent_answer=opponent_answer,
                include_rep_reasoning=True,
            )

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
            "stream": False,
            "options": {"temperature": self.temperature},
        }

        try:
            response = requests.post(OLLAMA_URL, json=payload, timeout=120)
            response.raise_for_status()
            data = response.json()
            return data["message"]["content"]
        except Exception as e:
            return f"THOUGHT: Error calling Ollama: {e}\nTALK: I need a moment.\nACTION: QUIT"
