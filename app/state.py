"""Shared in-memory state for the Dash app."""

from typing import Any, Dict, Optional

import pandas as pd


_DATASETS: Dict[str, pd.DataFrame] = {}
_LAST_AI_INSIGHT: Optional[Dict[str, Any]] = None
_LAST_MULTI_AGENT_RESULT: Optional[Dict[str, Any]] = None
_LAST_RECOMMENDATIONS: Optional[str] = None


def _clear_derived_outputs() -> None:
    global _LAST_AI_INSIGHT, _LAST_MULTI_AGENT_RESULT, _LAST_RECOMMENDATIONS
    _LAST_AI_INSIGHT = None
    _LAST_MULTI_AGENT_RESULT = None
    _LAST_RECOMMENDATIONS = None


def get_datasets() -> Dict[str, pd.DataFrame]:
    return _DATASETS


def get_dataset(name: str) -> Optional[pd.DataFrame]:
    return _DATASETS.get(name)


def set_datasets(datasets: Dict[str, pd.DataFrame]) -> None:
    global _DATASETS
    _DATASETS = dict(datasets)
    _clear_derived_outputs()


def set_dataset(name: str, df: pd.DataFrame) -> None:
    _DATASETS[name] = df
    _clear_derived_outputs()


def clear_datasets() -> None:
    _DATASETS.clear()
    _clear_derived_outputs()


def has_datasets() -> bool:
    return bool(_DATASETS)


def get_last_ai_insight() -> Optional[Dict[str, Any]]:
    return _LAST_AI_INSIGHT


def set_last_ai_insight(payload: Dict[str, Any]) -> None:
    global _LAST_AI_INSIGHT
    _LAST_AI_INSIGHT = payload


def get_last_multi_agent_result() -> Optional[Dict[str, Any]]:
    return _LAST_MULTI_AGENT_RESULT


def set_last_multi_agent_result(payload: Dict[str, Any]) -> None:
    global _LAST_MULTI_AGENT_RESULT
    _LAST_MULTI_AGENT_RESULT = payload


def get_last_recommendations() -> Optional[str]:
    return _LAST_RECOMMENDATIONS


def set_last_recommendations(text: str) -> None:
    global _LAST_RECOMMENDATIONS
    _LAST_RECOMMENDATIONS = text
