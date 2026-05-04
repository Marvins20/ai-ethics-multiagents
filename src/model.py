import logging

from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

from .config import settings

logger = logging.getLogger(__name__)

_RETRYABLE_NAMES = frozenset({"RateLimitError", "APIStatusError", "InternalServerError"})


def _is_retryable(exc: BaseException) -> bool:
    return type(exc).__name__ in _RETRYABLE_NAMES


_retry = retry(
    retry=retry_if_exception(_is_retryable),
    wait=wait_exponential(multiplier=2, min=15, max=120),
    stop=stop_after_attempt(5),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)


class _InvokeWrapper:
    def __init__(self, chain):
        self._chain = chain
        self.invoke = _retry(chain.invoke)


class _RetryModel:
    """
    Provider-agnostic wrapper that injects retry logic into invoke() and
    with_structured_output(). Switch providers via LLM_PROVIDER env var.
    """

    def __init__(self, base):
        self._base = base
        self.invoke = _retry(base.invoke)

    def with_structured_output(self, schema, **kwargs):
        return _InvokeWrapper(self._base.with_structured_output(schema, **kwargs))

    def bind_tools(self, tools, **kwargs):
        return _InvokeWrapper(self._base.bind_tools(tools, **kwargs))


def _build_base():
    provider = settings.llm_provider.lower()

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.llm_model or "Qwen/Qwen3-27B",
            api_key=settings.llm_api_key or "not-needed",
            base_url=settings.llm_base_url or None,
            temperature=0,
            max_tokens=8192,
        )

    # default: anthropic
    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(
        model=settings.llm_model or "claude-haiku-4-5-20251001",
        api_key=settings.claude_api_key,
        temperature=0,
        max_tokens=8192,
    )


model = _RetryModel(_build_base())
