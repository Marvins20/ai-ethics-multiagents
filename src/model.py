import logging

from langchain_anthropic import ChatAnthropic
from anthropic import RateLimitError, APIStatusError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

from .config import settings

logger = logging.getLogger(__name__)

_retry = retry(
    retry=retry_if_exception_type((RateLimitError, APIStatusError)),
    wait=wait_exponential(multiplier=2, min=15, max=120),
    stop=stop_after_attempt(5),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)


class _InvokeWrapper:
    """Wraps a LangChain chain to add tenacity retry to its invoke() call."""

    def __init__(self, chain):
        self._chain = chain
        self.invoke = _retry(chain.invoke)


class _RetryModel:
    """
    Transparent wrapper around ChatAnthropic that injects retry logic into
    every invoke() / with_structured_output() call without requiring changes
    in the agent files.

    Retry policy: up to 5 attempts, exponential backoff 15s → 120s.
    """

    def __init__(self, base: ChatAnthropic):
        self._base = base
        self.invoke = _retry(base.invoke)

    def with_structured_output(self, schema, **kwargs):
        return _InvokeWrapper(self._base.with_structured_output(schema, **kwargs))

    def bind_tools(self, tools, **kwargs):
        bound = self._base.bind_tools(tools, **kwargs)
        return _InvokeWrapper(bound)


_base = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    api_key=settings.claude_api_key,
    temperature=0,
    max_tokens=8192,
)
model = _RetryModel(_base)
