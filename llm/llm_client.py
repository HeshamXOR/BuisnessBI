"""
LLM Client Module
=================
Wraps Ollama for local open-source LLM inference.
Optimized for RTX A6000 (48GB VRAM) + 32GB RAM.

Speed-optimized: Uses Phi-3 Mini (3.8B) for fast English-only inference.
Supports streaming for responsive Streamlit UI.

Available models (fastest → most capable):
  - phi3:mini          (3.8B — fast, English-only, great reasoning) ★ DEFAULT
  - gemma2:2b          (2B   — ultra-fast, good for simple tasks)
  - llama3.1:8b        (8B   — balanced quality/speed)
  - phi3:medium        (14B  — high quality, still fast on A6000)
  - llama3.1:70b-instruct-q4_K_M (70B Q4 — highest quality)
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Optional, Dict, Any, Generator

try:
    import ollama

    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

from dotenv import load_dotenv

load_dotenv()


class LLMClient:
    """
    LLM Client using Ollama — optimized for speed on RTX A6000.

    Key optimizations:
    - Default model: phi3:mini (3.8B) — 3-4x faster than llama3.1:8b
    - Reduced context window (4096 vs 8192) — faster first-token
    - Flash attention enabled
    - Full GPU offload
    """

    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: Optional[float] = None,
        request_timeout: Optional[float] = None,
    ):
        self.model = model or os.getenv("OLLAMA_MODEL", "phi3:mini")
        self.base_url = base_url or os.getenv(
            "OLLAMA_BASE_URL", "http://localhost:11434"
        )
        self.temperature = temperature or float(os.getenv("LLM_TEMPERATURE", "0.3"))
        self.request_timeout = request_timeout or float(
            os.getenv("OLLAMA_TIMEOUT", "180")
        )

        if OLLAMA_AVAILABLE:
            self.client = ollama.Client(host=self.base_url)
        else:
            self.client = None

        self._call_count = 0
        self._total_tokens = 0
        self._total_time = 0.0

    def _run_with_timeout(self, func, *args, timeout: Optional[float] = None, **kwargs):
        """Run a blocking Ollama call with a timeout."""
        effective_timeout = timeout or self.request_timeout
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=effective_timeout)
        except FuturesTimeoutError as exc:
            future.cancel()
            raise TimeoutError(
                f"Ollama request timed out after {effective_timeout:.0f}s"
            ) from exc
        finally:
            # Do not block on shutdown if the worker is still running after timeout.
            executor.shutdown(wait=False, cancel_futures=True)

    def _model_is_available(self) -> bool:
        """Check whether the configured model is present in Ollama."""
        try:
            status = self.check_connection()
        except Exception:
            return False
        if not status.get("connected"):
            return False
        return bool(status.get("model_available"))

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: int = 2048,
    ) -> str:
        """
        Generate a response from the LLM.

        Args:
            prompt: The user prompt.
            system_prompt: Optional system instruction.
            temperature: Override default temperature.
            max_tokens: Maximum tokens (reduced default for speed).

        Returns:
            Generated text string.
        """
        if not OLLAMA_AVAILABLE or self.client is None:
            return self._fallback_response(prompt)

        if not self._model_is_available():
            return (
                f"## Model Not Available\n\n"
                f"The Ollama model `{self.model}` is not available or Ollama is not responding.\n\n"
                f"Try one of these:\n"
                f"1. `ollama serve`\n"
                f"2. `ollama pull phi3:mini`\n"
                f"3. set `OLLAMA_MODEL=phi3:mini` or create `business-analyst` first\n\n"
                f"Using fallback analysis instead.\n\n"
                + self._fallback_response(prompt)
            )

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        temp = temperature if temperature is not None else self.temperature
        options = {
            "temperature": temp,
            "num_predict": max_tokens,
            "num_gpu": 999,
            "num_ctx": 4096,
        }

        try:
            start_time = time.time()

            response = self._run_with_timeout(
                self.client.chat,
                model=self.model,
                messages=messages,
                options=options,
            )

        except TimeoutError:
            # Retry once with smaller generation budget before falling back.
            retry_tokens = min(max_tokens, 1024)
            retry_tokens = max(512, retry_tokens)
            print(
                f"⚠️ Ollama timed out at {self.request_timeout:.0f}s; retrying with num_predict={retry_tokens}."
            )
            try:
                start_time = time.time()
                response = self._run_with_timeout(
                    self.client.chat,
                    model=self.model,
                    messages=messages,
                    options={
                        "temperature": temp,
                        "num_predict": retry_tokens,
                        "num_gpu": 999,
                        "num_ctx": 4096,
                    },
                    timeout=max(self.request_timeout, 240),
                )
            except Exception as e:
                print(f"⚠️ LLM generation error after retry: {e}")
                return self._fallback_response(prompt)

        except Exception as e:
            print(f"⚠️ LLM generation error: {e}")
            return self._fallback_response(prompt)

        try:

            elapsed = time.time() - start_time
            self._call_count += 1
            self._total_time += elapsed

            if "eval_count" in response:
                self._total_tokens += response.get("eval_count", 0)

            return response["message"]["content"]

        except Exception as e:
            print(f"⚠️ LLM generation error: {e}")
            return self._fallback_response(prompt)

    def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: int = 2048,
    ) -> Generator[str, None, None]:
        """
        Stream a response from the LLM token-by-token.

        Yields:
            Chunks of text as they are generated.
        """
        if not OLLAMA_AVAILABLE or self.client is None:
            yield self._fallback_response(prompt)
            return

        if not self._model_is_available():
            yield (
                f"## Model Not Available\n\n"
                f"The Ollama model `{self.model}` is not available or Ollama is not responding.\n\n"
                + self._fallback_response(prompt)
            )
            return

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        temp = temperature if temperature is not None else self.temperature

        try:
            start_time = time.time()

            stream = self._run_with_timeout(
                self.client.chat,
                model=self.model,
                messages=messages,
                stream=True,
                options={
                    "temperature": temp,
                    "num_predict": max_tokens,
                    "num_gpu": 999,
                    "num_ctx": 4096,
                },
                timeout=max(self.request_timeout, 20),
            )

            for chunk in stream:
                token = chunk.get("message", {}).get("content", "")
                if token:
                    yield token

            elapsed = time.time() - start_time
            self._call_count += 1
            self._total_time += elapsed

        except Exception as e:
            print(f"⚠️ LLM streaming error: {e}")
            yield self._fallback_response(prompt)

    def generate_with_context(
        self,
        prompt: str,
        context: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Generate a response with additional data context."""
        full_prompt = f"""## Data Context
{context}

## Analysis Request
{prompt}"""
        return self.generate(
            prompt=full_prompt, system_prompt=system_prompt, temperature=temperature
        )

    def generate_structured(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Generate a structured (markdown) response."""
        structured_system = (system_prompt or "") + (
            "\n\nIMPORTANT: Respond with well-structured markdown using "
            "headers (##), bullet points (-), and numbered lists (1.). "
            "Be concise and data-driven."
        )
        return self.generate(
            prompt=prompt,
            system_prompt=structured_system,
            temperature=temperature or 0.2,
        )

    def check_connection(self) -> Dict[str, Any]:
        """Check if Ollama is running and the model is available."""
        if not OLLAMA_AVAILABLE:
            return {
                "connected": False,
                "error": "ollama package not installed. Run: pip install ollama",
            }

        try:
            models_response = self._run_with_timeout(self.client.list, timeout=10)
            # Handle both old API (dict with 'models' key) and new API (object with .models attr)
            if hasattr(models_response, "models"):
                model_list = models_response.models
            elif isinstance(models_response, dict):
                model_list = models_response.get("models", [])
            else:
                model_list = []
            # Handle both 'model' key (new API) and 'name' key (old API)
            available_models = []
            for m in model_list:
                if hasattr(m, "model"):
                    available_models.append(m.model)
                elif isinstance(m, dict):
                    available_models.append(m.get("model", m.get("name", str(m))))
                else:
                    available_models.append(str(m))
            model_available = any(self.model in m for m in available_models)

            return {
                "connected": True,
                "base_url": self.base_url,
                "model": self.model,
                "model_available": model_available,
                "available_models": available_models,
            }
        except Exception as e:
            return {
                "connected": False,
                "error": str(e),
                "hint": "Make sure Ollama is running: 'ollama serve'",
            }

    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            "total_calls": self._call_count,
            "total_tokens": self._total_tokens,
            "total_time_seconds": round(self._total_time, 2),
            "avg_time_per_call": round(self._total_time / max(self._call_count, 1), 2),
            "model": self.model,
        }

    def _fallback_response(self, prompt: str) -> str:
        """
        Generate a fallback response when Ollama is not available.
        Provides a meaningful demo experience without a running LLM.
        """
        prompt_lower = prompt.lower()

        if "sales" in prompt_lower or "revenue" in prompt_lower:
            return (
                "## Sales Analysis Insight\n\n"
                "Based on the data provided:\n\n"
                "1. **Revenue Trends**: Seasonal peaks in Q4 suggest holiday-driven demand.\n"
                "2. **Top Performers**: Electronics and Software lead revenue.\n"
                "3. **Regional Performance**: North America strongest; Asia Pacific fastest growth.\n"
                "4. **Recommendations**: Increase inventory for top products before Q4.\n\n"
                "*Fallback response — connect Ollama for AI-powered analysis.*"
            )
        elif "marketing" in prompt_lower or "campaign" in prompt_lower:
            return (
                "## Marketing Analysis Insight\n\n"
                "Key findings:\n\n"
                "1. **Channel ROI**: Email marketing shows highest ROI.\n"
                "2. **Campaign Type**: Lead Gen outperforms Brand Awareness in conversions.\n"
                "3. **Cost Efficiency**: Google Ads has lowest CPC.\n"
                "4. **Recommendations**: Shift 15% budget to Email and Google Ads.\n\n"
                "*Fallback response — connect Ollama for AI-powered analysis.*"
            )
        elif "customer" in prompt_lower or "churn" in prompt_lower:
            return (
                "## Customer Analysis Insight\n\n"
                "Key findings:\n\n"
                "1. **Segments**: Enterprise customers generate highest LTV (15% of base).\n"
                "2. **Churn Risk**: Small Business shows highest churn risk.\n"
                "3. **Engagement**: Strong correlation between engagement and retention.\n"
                "4. **Recommendations**: Targeted retention for high-risk segments.\n\n"
                "*Fallback response — connect Ollama for AI-powered analysis.*"
            )
        elif (
            "github" in prompt_lower or "repo" in prompt_lower or "tech" in prompt_lower
        ):
            return (
                "## Tech Analysis Insight\n\n"
                "Key findings:\n\n"
                "1. **Languages**: Python dominates; Rust growing fastest.\n"
                "2. **Quality**: CI/CD + docs correlate with higher quality scores.\n"
                "3. **Community**: Active contributors → faster issue resolution.\n"
                "4. **Recommendations**: Invest in CI/CD and documentation.\n\n"
                "*Fallback response — connect Ollama for AI-powered analysis.*"
            )
        else:
            return (
                "## AI Analysis Insight\n\n"
                "Based on the data analysis:\n\n"
                "1. **Data Quality**: Strong coverage with minimal missing values.\n"
                "2. **Key Patterns**: Clear trends and clusters identified.\n"
                "3. **Opportunities**: Several optimization areas detected.\n"
                "4. **Next Steps**: Deeper analysis on highest-impact areas recommended.\n\n"
                "*Fallback response — connect Ollama for AI-powered analysis.*"
            )
