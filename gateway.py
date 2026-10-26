"""
gateway.py

LLM gateway built on LiteLLM Router + langchain-litellm:
  - automatic retries and fallbacks (primary -> fallback model)
  - per-deployment rpm/tpm limits
  - per-user request rate limit and per-user daily cost budget
"""

import logging
import os
import threading
import time
from collections import defaultdict, deque
from datetime import date

import litellm
from langchain_litellm import ChatLiteLLMRouter
from litellm import Router

logger = logging.getLogger(__name__)

AWS_REGION = os.getenv("AWS_REGION", "ap-southeast-2")
MODEL_ID = os.getenv("BEDROCK_MODEL_ID")

# Full LiteLLM model string, e.g. "bedrock/converse/<other-model-id>"
# or "openai/gpt-4o-mini". Leave empty to run without a fallback.
FALLBACK_MODEL = os.getenv("FALLBACK_MODEL", "")

PRIMARY_MODEL = f"bedrock/converse/{MODEL_ID}"

# Per-deployment limits (enforced by the router)
DEPLOYMENT_RPM = int(os.getenv("DEPLOYMENT_RPM", "60"))
DEPLOYMENT_TPM = int(os.getenv("DEPLOYMENT_TPM", "100000"))

# Per-user limits (enforced by UserLimiter)
USER_RPM = int(os.getenv("USER_RPM", "5"))
USER_DAILY_BUDGET_USD = float(os.getenv("USER_DAILY_BUDGET_USD", "0.50"))
SESSION_MAX_REQUESTS = int(os.getenv("SESSION_MAX_REQUESTS", "4"))  # total per session
# Used only if LiteLLM has no price for the model
FALLBACK_PRICE_IN_PER_1K = float(os.getenv("COST_INPUT_PER_1K", "0.003"))
FALLBACK_PRICE_OUT_PER_1K = float(os.getenv("COST_OUTPUT_PER_1K", "0.015"))


class LimitExceeded(Exception):
    """Raised when a user exceeds the rate limit or daily budget."""

class UserLimiter:
    """Per-session rate limit, total request cap and daily cost budget."""

    def __init__(self, rpm: int, daily_budget_usd: float, max_total: int):
        self.rpm = rpm
        self.daily_budget = daily_budget_usd
        self.max_total = max_total
        self._hits = defaultdict(deque)
        self._total = defaultdict(int)
        self._spend = defaultdict(lambda: (date.today(), 0.0))
        self._lock = threading.Lock()

    def check(self, user_id: str) -> None:
        now = time.monotonic()
        with self._lock:
            if self._total[user_id] >= self.max_total:
                raise LimitExceeded(
                    f"Demo limit reached ({self.max_total} questions per session)."
                )

            hits = self._hits[user_id]
            while hits and now - hits[0] > 60:
                hits.popleft()
            if len(hits) >= self.rpm:
                wait = int(60 - (now - hits[0])) + 1
                raise LimitExceeded(
                    f"Too many requests ({self.rpm}/minute). "
                    f"Please try again in {wait}s."
                )

            day, spent = self._spend[user_id]
            if day != date.today():
                day, spent = date.today(), 0.0
                self._spend[user_id] = (day, spent)
            if spent >= self.daily_budget:
                raise LimitExceeded("Session usage limit reached.")

            hits.append(now)
            self._total[user_id] += 1

    def add_cost(self, user_id: str, usd: float) -> None:
        with self._lock:
            day, spent = self._spend[user_id]
            if day != date.today():
                day, spent = date.today(), 0.0
            self._spend[user_id] = (day, spent + usd)


limiter = UserLimiter(USER_RPM, USER_DAILY_BUDGET_USD, SESSION_MAX_REQUESTS)

def _count_tokens(text: str) -> int:
    try:
        return litellm.token_counter(model=PRIMARY_MODEL, text=text)
    except Exception:
        return max(1, len(text) // 4)


def record_usage(user_id: str, prompt_text: str, output_text: str) -> float:
    """Estimate the cost of one call and charge it to the user."""
    prompt_tokens = _count_tokens(prompt_text)
    completion_tokens = _count_tokens(output_text)
    try:
        in_cost, out_cost = litellm.cost_per_token(
            model=PRIMARY_MODEL,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        cost = in_cost + out_cost
    except Exception:
        cost = (
            prompt_tokens * FALLBACK_PRICE_IN_PER_1K
            + completion_tokens * FALLBACK_PRICE_OUT_PER_1K
        ) / 1000

    limiter.add_cost(user_id, cost)
    logger.info("user=%s tokens_in=%d tokens_out=%d cost=$%.5f",
                user_id, prompt_tokens, completion_tokens, cost)
    return cost


def _build_router() -> Router:
    model_list = [{
        "model_name": "primary",
        "litellm_params": {
            "model": PRIMARY_MODEL,
            "aws_region_name": AWS_REGION,
            "rpm": DEPLOYMENT_RPM,
            "tpm": DEPLOYMENT_TPM,
        },
    }]
    fallbacks = []

    if FALLBACK_MODEL:
        model_list.append({
            "model_name": "fallback",
            "litellm_params": {
                "model": FALLBACK_MODEL,
                "rpm": DEPLOYMENT_RPM,
                "tpm": DEPLOYMENT_TPM,
            },
        })
        fallbacks = [{"primary": ["fallback"]}]

    return Router(
        model_list=model_list,
        fallbacks=fallbacks,
        num_retries=2,
        timeout=60,
        allowed_fails=3,      # failures per minute before a deployment cools down
        cooldown_time=30,     # seconds
    )


def get_llm() -> ChatLiteLLMRouter:
    return ChatLiteLLMRouter(
        router=_build_router(),
        model_name="primary",
        temperature=0,
        max_tokens=1000,
        streaming=True,
    )