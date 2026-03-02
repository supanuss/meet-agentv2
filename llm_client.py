"""LLM access layer using LangChain chat models with optional fallback."""

from __future__ import annotations

import json
import re
import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from pipeline_utils import PipelineConfig, PipelineError, fill_template
from prompts import JSON_REPAIR_SYS, JSON_REPAIR_USR


def clean_json_text(text: str) -> str:
    t = text.replace("\ufeff", "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE).strip()
        t = re.sub(r"\s*```$", "", t).strip()
    return t


def _try_decode_json_fragment(text: str) -> tuple[Any, str] | None:
    s = text.strip()
    if not s:
        return None

    try:
        return json.loads(s), s
    except Exception:
        pass

    decoder = json.JSONDecoder()
    for i, ch in enumerate(s):
        if ch not in "[{":
            continue
        try:
            obj, end = decoder.raw_decode(s[i:])
        except json.JSONDecodeError:
            continue
        return obj, s[i : i + end]
    return None


def extract_json_candidate(text: str) -> str | None:
    direct = _try_decode_json_fragment(text)
    if direct is not None:
        _, candidate = direct
        return candidate

    for block in re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", text, flags=re.IGNORECASE):
        parsed = _try_decode_json_fragment(block)
        if parsed is not None:
            _, candidate = parsed
            return candidate

    cleaned = clean_json_text(text)
    parsed = _try_decode_json_fragment(cleaned)
    if parsed is not None:
        _, candidate = parsed
        return candidate

    return None


def validate_keys(obj: dict[str, Any], required_keys: list[str], label: str) -> None:
    missing = [k for k in required_keys if k not in obj]
    if missing:
        raise PipelineError(f"{label} missing keys: {missing}")


def parse_json_or_raise(text: str, label: str) -> dict[str, Any]:
    candidate = extract_json_candidate(text)
    if not candidate:
        raise PipelineError(f"{label}: no valid JSON object found")
    data = json.loads(candidate)
    if isinstance(data, str):
        nested = extract_json_candidate(data)
        if nested:
            data = json.loads(nested)
    if isinstance(data, list):
        dict_items = [item for item in data if isinstance(item, dict)]
        if dict_items:
            # Some models return a list of objects; salvage the most complete object.
            data = max(dict_items, key=lambda item: len(item.keys()))
    if not isinstance(data, dict):
        raise PipelineError(f"{label}: JSON root must be object")
    return data


def is_token_limit_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    token_markers = [
        "token",
        "context length",
        "maximum context length",
        "context window",
        "max_tokens",
        "too many tokens",
        "prompt is too long",
        "exceeds the context",
    ]
    return any(m in msg for m in token_markers)


def shrink_prompt_text(text: str) -> str:
    if len(text) <= 12000:
        return text
    keep_head = int(len(text) * 0.55)
    keep_tail = int(len(text) * 0.35)
    if keep_head + keep_tail >= len(text):
        return text
    marker = "\n\n[...TRUNCATED_FOR_TOKEN_LIMIT...]\n\n"
    return text[:keep_head] + marker + text[-keep_tail:]


def _message_to_text(message: Any) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                txt = item.get("text") or item.get("content")
                if isinstance(txt, str):
                    parts.append(txt)
        return "\n".join(parts).strip()
    return str(content or "")


def _normalize_openai_base_url(base_url: str) -> str:
    text = str(base_url or "").strip().rstrip("/")
    if not text:
        return text
    if text.endswith("/v1"):
        return text
    return text + "/v1"


class LLMClient:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.typhoon_llm: ChatOpenAI | None = None
        self.ollama_chat_llm: ChatOllama | None = None
        self.vllm_chat_llm: ChatOpenAI | None = None
        self.embedder: Any | None = None
        self.call_log: list[dict[str, Any]] = []
        self._fallback_notice_printed = False
        self._fallback_only_mode = False
        self._ollama_sdk_client: Any | None = None
        self._ollama_direct_mode = False
        self._ollama_direct_mode_notice_printed = False

        if cfg.typhoon_api_key and cfg.typhoon_base_url:
            self.typhoon_llm = ChatOpenAI(
                model=cfg.typhoon_model,
                api_key=cfg.typhoon_api_key,
                base_url=cfg.typhoon_base_url,
                temperature=0.1,
                max_tokens=max(int(cfg.typhoon_max_tokens or 8192), 256),
                timeout=cfg.llm_timeout_sec,
            )

        if cfg.chat_fallback_provider == "ollama":
            chat_kwargs: dict[str, Any] = {
                "model": cfg.ollama_chat_model,
                "base_url": cfg.ollama_base_url,
                "temperature": 0.1,
            }
            # Pass timeout down to Ollama SDK client when supported.
            try:
                self.ollama_chat_llm = ChatOllama(
                    **chat_kwargs,
                    client_kwargs={"timeout": cfg.llm_timeout_sec},
                )
            except TypeError:
                self.ollama_chat_llm = ChatOllama(**chat_kwargs)
        elif cfg.chat_fallback_provider == "vllm":
            self.vllm_chat_llm = ChatOpenAI(
                model=cfg.vllm_chat_model,
                api_key=cfg.vllm_api_key or "EMPTY",
                base_url=_normalize_openai_base_url(cfg.vllm_base_url),
                temperature=0.1,
                max_tokens=max(int(cfg.typhoon_max_tokens or 8192), 256),
                timeout=cfg.llm_timeout_sec,
            )
        else:
            raise PipelineError(
                f"Unsupported CHAT_FALLBACK_PROVIDER={cfg.chat_fallback_provider!r}; "
                "allowed values: ollama, vllm"
            )

        if cfg.embedding_provider == "ollama":
            self.embedder = OllamaEmbeddings(
                model=cfg.ollama_embed_model,
                base_url=cfg.ollama_base_url,
            )
        elif cfg.embedding_provider == "vllm":
            self.embedder = OpenAIEmbeddings(
                model=cfg.vllm_embed_model,
                api_key=cfg.vllm_api_key or "EMPTY",
                base_url=_normalize_openai_base_url(cfg.vllm_base_url),
            )
        else:
            raise PipelineError(
                f"Unsupported EMBEDDING_PROVIDER={cfg.embedding_provider!r}; "
                "allowed values: ollama, vllm"
            )

        self._fallback_only_mode = self.typhoon_llm is None

        if not self.typhoon_llm and not cfg.allow_ollama_chat_fallback:
            raise PipelineError(
                "Typhoon is unavailable and chat fallback is disabled "
                "(set ALLOW_CHAT_FALLBACK=true for fallback mode)."
            )

    def _providers_in_order(self) -> list[str]:
        providers: list[str] = []
        if self.typhoon_llm:
            providers.append("typhoon")
        if self.cfg.allow_ollama_chat_fallback:
            providers.append(self.cfg.chat_fallback_provider)
        return providers

    def _invoke_typhoon(self, system: str, user: str, json_mode: bool) -> str:
        if not self.typhoon_llm:
            raise PipelineError("Typhoon provider is not configured")

        dynamic_max_tokens = max(int(self.cfg.typhoon_max_tokens or 8192), 256)
        dynamic_user = user
        last_exc: Exception | None = None

        for _ in range(5):
            try:
                model = self.typhoon_llm.bind(max_tokens=dynamic_max_tokens)
                if json_mode:
                    model = model.bind(response_format={"type": "json_object"})
                resp = model.invoke(
                    [
                        SystemMessage(content=system),
                        HumanMessage(content=dynamic_user),
                    ]
                )
                content = _message_to_text(resp)
                if not content:
                    raise PipelineError("Typhoon returned empty response")
                return content
            except Exception as exc:
                last_exc = exc
                if not is_token_limit_error(exc):
                    raise
                if dynamic_max_tokens > 1024:
                    dynamic_max_tokens = max(1024, dynamic_max_tokens // 2)
                    continue
                shrunk = shrink_prompt_text(dynamic_user)
                if shrunk != dynamic_user:
                    dynamic_user = shrunk
                    continue
                break

        raise PipelineError(f"Typhoon token-limit handling exhausted: {last_exc}")

    def _invoke_ollama(self, system: str, user: str, json_mode: bool = False) -> str:
        if self._ollama_direct_mode:
            return self._invoke_ollama_via_sdk(system, user, json_mode=json_mode)
        if not self.ollama_chat_llm:
            raise PipelineError("Ollama chat model is not configured")
        model = self.ollama_chat_llm
        invoke_options = {
            "temperature": 0.1,
            "num_predict": max(256, int(self.cfg.ollama_num_predict or 4096)),
        }
        try:
            model = model.bind(options=invoke_options)
        except Exception:
            pass
        if json_mode:
            # Ask Ollama runtime to bias structured JSON output.
            model = model.bind(format="json")

        try:
            resp = model.invoke(
                [
                    SystemMessage(content=system),
                    HumanMessage(content=user),
                ]
            )
            content = _message_to_text(resp)
            if not content:
                raise PipelineError("Ollama returned empty response")
            return content
        except Exception as exc:
            # Some langchain_ollama/ollama version combinations pass `num_predict`
            # directly into Client.chat(), which is rejected by newer Ollama SDKs.
            if "unexpected keyword argument 'num_predict'" in str(exc):
                self._ollama_direct_mode = True
                if not self._ollama_direct_mode_notice_printed:
                    print("⚠️ Ollama adapter mismatch detected; switching to direct SDK calls.")
                    self._ollama_direct_mode_notice_printed = True
                return self._invoke_ollama_via_sdk(system, user, json_mode=json_mode)
            raise

    def _invoke_ollama_via_sdk(self, system: str, user: str, json_mode: bool = False) -> str:
        if self._ollama_sdk_client is None:
            try:
                from ollama import Client as OllamaClient
            except Exception as exc:
                raise PipelineError(f"Ollama SDK is unavailable: {exc}") from exc
            try:
                self._ollama_sdk_client = OllamaClient(
                    host=self.cfg.ollama_base_url,
                    timeout=self.cfg.llm_timeout_sec,
                )
            except TypeError:
                self._ollama_sdk_client = OllamaClient(host=self.cfg.ollama_base_url)

        payload: dict[str, Any] = {
            "model": self.cfg.ollama_chat_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": max(256, int(self.cfg.ollama_num_predict or 4096)),
            },
        }
        if json_mode:
            payload["format"] = "json"

        try:
            resp = self._ollama_sdk_client.chat(**payload)
        except Exception as exc:
            raise PipelineError(f"Ollama SDK call failed: {exc}") from exc

        content = ""
        if isinstance(resp, dict):
            message = resp.get("message") or {}
            content = str(message.get("content") or "").strip()
        else:
            message = getattr(resp, "message", None)
            if isinstance(message, dict):
                content = str(message.get("content") or "").strip()
            else:
                content = str(getattr(message, "content", "") or "").strip()

        if not content:
            raise PipelineError("Ollama SDK returned empty response")
        return content

    def _invoke_vllm(self, system: str, user: str, json_mode: bool = False) -> str:
        if not self.vllm_chat_llm:
            raise PipelineError("vLLM chat model is not configured")
        messages = [
            SystemMessage(content=system),
            HumanMessage(content=user),
        ]
        model = self.vllm_chat_llm
        if json_mode:
            try:
                model = model.bind(response_format={"type": "json_object"})
            except Exception:
                pass
        resp = model.invoke(messages)
        content = _message_to_text(resp)
        if not content:
            raise PipelineError("vLLM returned empty response")
        return content

    def _invoke_by_provider(self, provider: str, system: str, user: str, json_mode: bool) -> str:
        if provider == "typhoon":
            return self._invoke_typhoon(system, user, json_mode=json_mode)
        if provider == "ollama":
            return self._invoke_ollama(system, user, json_mode=json_mode)
        if provider == "vllm":
            return self._invoke_vllm(system, user, json_mode=json_mode)
        raise PipelineError(f"Unknown provider: {provider}")

    def _maybe_log_fallback_notice(self, provider: str) -> None:
        if provider == "typhoon" or self._fallback_notice_printed:
            return
        if self._fallback_only_mode:
            if provider == "ollama":
                print("ℹ️ Ollama-only mode active (Typhoon not configured).")
            else:
                print("ℹ️ vLLM-only mode active (Typhoon not configured).")
        else:
            if provider == "ollama":
                print(
                    "⚠️ Using Ollama chat fallback (quality may differ from Typhoon). "
                    "Set ALLOW_CHAT_FALLBACK=false for Typhoon-only mode."
                )
            else:
                print(
                    "⚠️ Using vLLM chat fallback (quality may differ from Typhoon). "
                    "Set ALLOW_CHAT_FALLBACK=false for Typhoon-only mode."
                )
        self._fallback_notice_printed = True

    def _repair_json(self, broken: str, required_keys: list[str], tag: str) -> dict[str, Any] | None:
        try:
            parsed = parse_json_or_raise(broken, f"{tag}/repair-heuristic")
            if required_keys:
                validate_keys(parsed, required_keys, f"{tag}/repair-heuristic")
            return parsed
        except Exception:
            pass

        repair_prompt = fill_template(
            JSON_REPAIR_USR,
            REQUIRED_KEYS=json.dumps(required_keys, ensure_ascii=False),
            BROKEN_JSON=broken[:60000],
        )

        for provider in self._providers_in_order():
            start = time.time()
            try:
                raw = self._invoke_by_provider(provider, JSON_REPAIR_SYS, repair_prompt, json_mode=True)
                parsed = parse_json_or_raise(raw, f"{tag}/repair-{provider}")
                validate_keys(parsed, required_keys, f"{tag}/repair-{provider}")
                self.call_log.append(
                    {
                        "tag": tag,
                        "provider": provider,
                        "phase": "repair",
                        "chat_fallback_used": provider != "typhoon",
                        "success": True,
                        "latency_sec": round(time.time() - start, 3),
                    }
                )
                return parsed
            except Exception as exc:
                self.call_log.append(
                    {
                        "tag": tag,
                        "provider": provider,
                        "phase": "repair",
                        "chat_fallback_used": provider != "typhoon",
                        "success": False,
                        "latency_sec": round(time.time() - start, 3),
                        "error": str(exc),
                    }
                )
        return None

    def call(
        self,
        system: str,
        user: str,
        *,
        json_mode: bool,
        required_keys: list[str] | None,
        tag: str,
    ) -> dict[str, Any] | str:
        last_error: Exception | None = None
        required = required_keys or []

        for attempt in range(1, self.cfg.llm_max_retries + 1):
            for provider in self._providers_in_order():
                start = time.time()
                raw = ""
                try:
                    self._maybe_log_fallback_notice(provider)
                    raw = self._invoke_by_provider(provider, system, user, json_mode=json_mode)

                    if not json_mode:
                        self.call_log.append(
                            {
                                "tag": tag,
                                "provider": provider,
                                "attempt": attempt,
                                "chat_fallback_used": provider != "typhoon",
                                "success": True,
                                "latency_sec": round(time.time() - start, 3),
                            }
                        )
                        return raw

                    parsed = parse_json_or_raise(raw, tag)
                    if required:
                        validate_keys(parsed, required, tag)

                    self.call_log.append(
                        {
                            "tag": tag,
                            "provider": provider,
                            "attempt": attempt,
                            "chat_fallback_used": provider != "typhoon",
                            "success": True,
                            "latency_sec": round(time.time() - start, 3),
                        }
                    )
                    return parsed
                except Exception as exc:
                    last_error = exc
                    self.call_log.append(
                        {
                            "tag": tag,
                            "provider": provider,
                            "attempt": attempt,
                            "chat_fallback_used": provider != "typhoon",
                            "success": False,
                            "latency_sec": round(time.time() - start, 3),
                            "error": str(exc),
                            "raw_preview": (raw[:300] if raw else ""),
                        }
                    )
                    if json_mode and raw:
                        repaired = self._repair_json(raw, required, tag)
                        if repaired is not None:
                            if required:
                                validate_keys(repaired, required, tag)
                            return repaired

        raise PipelineError(f"{tag} failed after retries: {last_error}")

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not self.embedder:
            raise PipelineError("Embedding model is not configured")
        try:
            vectors = self.embedder.embed_documents(texts)
            return [list(v) for v in vectors]
        except Exception as exc:
            msg = str(exc)
            if self.cfg.embedding_provider == "vllm" and ("404" in msg or "Not Found" in msg):
                raise PipelineError(
                    "Embedding failed: vLLM endpoint does not expose /v1/embeddings for the configured model. "
                    "Use EMBEDDING_PROVIDER=ollama or deploy a vLLM embedding endpoint."
                ) from exc
            raise PipelineError(f"Embedding failed: {exc}") from exc
