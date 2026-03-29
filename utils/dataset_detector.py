"""
Dataset Detector Module
========================
Automatically classifies ANY CSV dataset by analyzing column names, data types,
and value patterns. Generates dynamic KPIs and chart recommendations.

Replaces hardcoded dataset type detection — works with any business CSV.
"""

import re
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple


# ─── Column Pattern Definitions ────────────────────────────────────

COLUMN_PATTERNS = {
    "sales": {
        "strong": ["revenue", "sales", "order", "invoice", "transaction"],
        "medium": ["product", "quantity", "units_sold", "discount", "profit", "cost",
                    "unit_price", "total_price", "order_id", "sku"],
        "weak": ["region", "store", "channel", "category"]
    },
    "marketing": {
        "strong": ["campaign", "ctr", "impression", "click", "conversion", "ad_spend",
                    "marketing", "cpc", "cpm"],
        "medium": ["channel", "roi", "reach", "engagement", "bounce_rate", "spend",
                    "ad_group", "keyword"],
        "weak": ["views", "likes", "shares", "audience"]
    },
    "customers": {
        "strong": ["customer", "churn", "lifetime_value", "ltv", "satisfaction",
                    "nps", "subscriber", "member"],
        "medium": ["segment", "retention", "purchase_frequency", "loyalty",
                    "account_age", "support_ticket", "engagement_score"],
        "weak": ["age", "gender", "location", "signup"]
    },
    "survey": {
        "strong": ["nps", "net_promoter", "csat", "survey", "response", "rating",
                    "feedback", "satisfaction", "recommend"],
        "medium": ["sentiment", "comment", "score", "promoter", "detractor",
                    "touchpoint", "experience", "resolution"],
        "weak": ["channel", "region", "product", "team", "agent"]
    },
    "financial": {
        "strong": ["balance", "debit", "credit", "account", "ledger", "journal",
                    "asset", "liability", "equity", "dividend"],
        "medium": ["interest", "tax", "expense", "income", "budget", "forecast",
                    "payable", "receivable", "cash_flow"],
        "weak": ["amount", "fiscal", "quarter", "annual"]
    },
    "hr": {
        "strong": ["employee", "salary", "department", "hire_date", "job_title",
                    "payroll", "leave", "attendance"],
        "medium": ["performance", "manager", "position", "bonus", "overtime",
                    "headcount", "turnover", "tenure"],
        "weak": ["team", "office", "role", "skills"]
    },
    "inventory": {
        "strong": ["stock", "warehouse", "sku", "inventory", "reorder",
                    "supplier", "procurement"],
        "medium": ["quantity", "unit_cost", "shelf_life", "batch", "bin",
                    "backorder", "lead_time"],
        "weak": ["item", "weight", "dimension", "barcode"]
    },
    "tech": {
        "strong": ["repo", "stars", "forks", "commit", "pull_request",
                    "issue", "contributor", "code_quality"],
        "medium": ["language", "framework", "ci_cd", "documentation",
                    "test_coverage", "bug", "feature"],
        "weak": ["version", "release", "dependency", "license"]
    }
}

# Semantic date column detection
DATE_PATTERNS = [
    "date", "time", "timestamp", "created", "updated", "modified",
    "start", "end", "period", "month", "year", "day", "week", "quarter"
]

# Monetary column detection
MONETARY_PATTERNS = [
    "revenue", "cost", "price", "spend", "budget", "profit", "income",
    "expense", "salary", "amount", "total", "fee", "payment", "balance",
    "value", "ltv", "lifetime_value", "roi", "cpc", "cpm"
]

SURVEY_SCORE_PATTERNS = [
    "nps", "nps_score", "net_promoter", "csat", "satisfaction", "rating", "score"
]

# Rate / percentage column detection
RATE_PATTERNS = [
    "rate", "ratio", "pct", "percent", "proportion", "margin",
    "ctr", "conversion", "churn", "retention", "growth", "yield",
    "bounce", "attrition", "utilization"
]

# Funnel stage patterns (ordered high→low)
FUNNEL_PATTERNS = [
    ["impressions", "views", "reach", "visits"],
    ["clicks", "click", "sessions", "engagements"],
    ["leads", "signups", "sign_up", "registrations"],
    ["conversions", "conversion", "purchases", "orders", "sales"],
]


def resolve_column(df, candidates: list, strict: bool = False):
    """Fuzzy-match a column name from a list of candidates.

    Tries exact match first, then substring containment.
    Returns the original DataFrame column name or None.
    """
    columns_lower = {c.lower().strip(): c for c in df.columns}
    # Exact match pass
    for candidate in candidates:
        if candidate.lower() in columns_lower:
            return columns_lower[candidate.lower()]
    if strict:
        return None
    # Substring containment pass
    for candidate in candidates:
        for col_lower, col_orig in columns_lower.items():
            if candidate.lower() in col_lower:
                return col_orig
    return None


class DatasetDetector:
    """
    Automatically analyzes and classifies any CSV dataset.

    Detects:
    - Dataset category (sales, marketing, customers, etc.)
    - Date columns
    - Monetary/numeric columns
    - Categorical columns
    - Appropriate KPIs
    - Recommended chart types
    """

    def __init__(self, df: pd.DataFrame, name: str = "dataset"):
        """
        Initialize detector with a DataFrame.

        Args:
            df: The DataFrame to analyze.
            name: Optional human-readable name for this dataset.
        """
        self.df = df
        self.name = name
        self._detected_type: Optional[str] = None
        self._secondary_type: Optional[str] = None
        self._confidence: float = 0.0
        self._column_roles: Dict[str, str] = {}
        self._date_columns: List[str] = []
        self._numeric_columns: List[str] = []
        self._categorical_columns: List[str] = []
        self._monetary_columns: List[str] = []
        self._rate_columns: List[str] = []

        # Run detection
        self._analyze()

    def _analyze(self):
        """Run full column analysis and type detection."""
        self._detect_column_types()
        self._detect_dataset_type()

    def _detect_column_types(self):
        """Classify each column as date, numeric, categorical, or monetary."""
        for col in self.df.columns:
            col_lower = col.lower().strip()

            # Date detection
            if self._is_date_column(col):
                self._date_columns.append(col)
                self._column_roles[col] = "date"
                continue

            # Numeric detection (exclude boolean columns — they break max()-min())
            if pd.api.types.is_numeric_dtype(self.df[col]) and not pd.api.types.is_bool_dtype(self.df[col]):
                self._numeric_columns.append(col)

                # Check if monetary
                if any(p in col_lower for p in MONETARY_PATTERNS):
                    self._monetary_columns.append(col)
                    self._column_roles[col] = "monetary"
                # Check if rate / percentage
                elif any(p in col_lower for p in RATE_PATTERNS):
                    self._rate_columns.append(col)
                    self._column_roles[col] = "rate"
                else:
                    self._column_roles[col] = "numeric"
            else:
                self._categorical_columns.append(col)
                self._column_roles[col] = "categorical"

    def _is_date_column(self, col: str) -> bool:
        """Check if a column is a date column by name or content."""
        col_lower = col.lower().strip()
        series = self.df[col].dropna()
        if series.empty:
            return False

        explicit_time_name = any(
            p in col_lower for p in ["date", "timestamp", "datetime", "created", "updated", "epoch", "unix"]
        )

        if pd.api.types.is_datetime64_any_dtype(self.df[col]):
            return True

        # Numeric columns are often IDs/durations; only treat as date if explicitly named and plausible.
        if pd.api.types.is_numeric_dtype(self.df[col]) and not pd.api.types.is_bool_dtype(self.df[col]):
            if not explicit_time_name:
                return False
            sample_num = pd.to_numeric(series.head(50), errors="coerce").dropna()
            if sample_num.empty:
                return False

            parsed = pd.to_datetime(sample_num, unit="s", errors="coerce")
            if parsed.notna().sum() < max(3, int(len(sample_num) * 0.7)):
                parsed = pd.to_datetime(sample_num, unit="ms", errors="coerce")

            parsed = parsed.dropna()
            if parsed.empty:
                return False

            # Ensure plausible year range and enough variation.
            if parsed.dt.year.between(1990, 2100).mean() < 0.8:
                return False
            return parsed.nunique() >= 3

        # Check if values can be parsed as dates (sample first 20 non-null)
        if self.df[col].dtype == "object":
            sample = series.head(40)
            if len(sample) > 0:
                try:
                    parsed = pd.to_datetime(sample, format="mixed", dayfirst=False, errors="coerce")
                    parsed_valid = parsed.dropna()
                    parse_ratio = parsed_valid.shape[0] / max(1, len(sample))

                    # Need strong parse confidence and enough variation to avoid false positives.
                    if parse_ratio >= 0.8 and parsed_valid.nunique() >= 3:
                        return True

                    # Only allow weaker parse if name strongly suggests datetime semantics.
                    if explicit_time_name and parse_ratio >= 0.6 and parsed_valid.nunique() >= 3:
                        return True
                except (ValueError, TypeError):
                    pass

        # Weak name-only date detection for non-numeric columns.
        if any(p in col_lower for p in DATE_PATTERNS) and not pd.api.types.is_numeric_dtype(self.df[col]):
            return True

        return False

    def _compute_type_scores(self) -> Dict[str, int]:
        """Score each dataset type against column names."""
        col_names_lower = [c.lower().strip() for c in self.df.columns]
        scores = {}
        for dtype, patterns in COLUMN_PATTERNS.items():
            score = 0
            for col in col_names_lower:
                for p in patterns.get("strong", []):
                    if p in col:
                        score += 3
                for p in patterns.get("medium", []):
                    if p in col:
                        score += 2
                for p in patterns.get("weak", []):
                    if p in col:
                        score += 1
            scores[dtype] = score
        return scores

    def _detect_dataset_type(self):
        """Detect primary and secondary dataset categories."""
        scores = self._compute_type_scores()

        if scores:
            sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            best_type, best_score = sorted_scores[0]
            total_cols = len(self.df.columns)

            self._confidence = min(1.0, best_score / (total_cols * 1.5))

            if best_score >= 2:
                self._detected_type = best_type
            else:
                self._detected_type = "generic"
                self._confidence = 0.3

            # Secondary type — if runner-up is within 60% of primary score
            if len(sorted_scores) >= 2:
                sec_type, sec_score = sorted_scores[1]
                if sec_score >= 2 and sec_score >= best_score * 0.6:
                    self._secondary_type = sec_type
        else:
            self._detected_type = "generic"
            self._confidence = 0.1

    @property
    def detected_type(self) -> str:
        """Get the detected dataset type."""
        return self._detected_type or "generic"

    @property
    def confidence(self) -> float:
        """Get detection confidence (0-1)."""
        return self._confidence

    @property
    def secondary_type(self) -> Optional[str]:
        """Second-best dataset type (if close to primary)."""
        return self._secondary_type

    @property
    def date_columns(self) -> List[str]:
        return self._date_columns

    @property
    def numeric_columns(self) -> List[str]:
        return self._numeric_columns

    @property
    def categorical_columns(self) -> List[str]:
        return self._categorical_columns

    @property
    def monetary_columns(self) -> List[str]:
        return self._monetary_columns

    @property
    def rate_columns(self) -> List[str]:
        return self._rate_columns

    def get_primary_metric(self) -> Optional[str]:
        """Get the most likely primary metric column (monetary > largest numeric)."""
        if self._monetary_columns:
            return self._monetary_columns[0]
        if self._numeric_columns:
            # Return the numeric column with greatest range (skip booleans)
            ranges = {}
            for col in self._numeric_columns:
                try:
                    col_range = float(self.df[col].max()) - float(self.df[col].min())
                    ranges[col] = col_range
                except (TypeError, ValueError):
                    continue
            if ranges:
                return max(ranges, key=ranges.get)
        return None

    def get_primary_category(self) -> Optional[str]:
        """Get the most likely category/group column."""
        for col in self._categorical_columns:
            nunique = self.df[col].nunique()
            # Good category: 2-50 unique values
            if 2 <= nunique <= 50:
                return col
        return self._categorical_columns[0] if self._categorical_columns else None

    def get_primary_date(self) -> Optional[str]:
        """Get the most likely primary date column."""
        return self._date_columns[0] if self._date_columns else None

    def _find_column_by_patterns(
        self, patterns: List[str], within: Optional[List[str]] = None
    ) -> Optional[str]:
        """Find the first column whose name contains any provided pattern."""
        columns = within if within is not None else list(self.df.columns)
        for col in columns:
            col_lower = col.lower().strip()
            if any(p in col_lower for p in patterns):
                return col
        return None

    def compute_auto_kpis(self) -> Dict[str, Any]:
        """
        Automatically compute KPIs based on detected column types.
        Works with ANY dataset structure.
        """
        kpis = {
            "total_records": len(self.df),
            "total_columns": len(self.df.columns),
            "detected_type": self.detected_type,
            "detection_confidence": f"{self.confidence:.0%}"
        }

        missing_cells = int(self.df.isna().sum().sum())
        total_cells = max(1, int(self.df.shape[0] * self.df.shape[1]))
        kpis["missing_cells"] = missing_cells
        kpis["data_completeness_pct"] = round((1 - (missing_cells / total_cells)) * 100, 2)

        # Monetary KPIs
        for col in self._monetary_columns[:4]:
            label = col.replace("_", " ").title()
            kpis[f"total_{col}"] = round(self.df[col].sum(), 2)
            kpis[f"avg_{col}"] = round(self.df[col].mean(), 2)
            kpis[f"median_{col}"] = round(self.df[col].median(), 2)

        # Numeric KPIs (non-monetary)
        non_monetary_numeric = [c for c in self._numeric_columns
                                if c not in self._monetary_columns]
        for col in non_monetary_numeric[:4]:
            kpis[f"avg_{col}"] = round(self.df[col].mean(), 2)
            kpis[f"max_{col}"] = round(self.df[col].max(), 2)

        # NPS / survey KPIs (dataset-agnostic by column pattern)
        nps_col = self._find_column_by_patterns(["nps", "nps_score", "net_promoter"])
        if nps_col and pd.api.types.is_numeric_dtype(self.df[nps_col]):
            nps_scores = pd.to_numeric(self.df[nps_col], errors="coerce").dropna()
            bounded_scores = nps_scores[(nps_scores >= 0) & (nps_scores <= 10)]
            if not bounded_scores.empty:
                promoter_pct = round((bounded_scores >= 9).mean() * 100, 2)
                detractor_pct = round((bounded_scores <= 6).mean() * 100, 2)
                passive_pct = round(
                    ((bounded_scores >= 7) & (bounded_scores <= 8)).mean() * 100, 2
                )
                kpis["nps_score_mean"] = round(bounded_scores.mean(), 2)
                kpis["nps_index"] = round(promoter_pct - detractor_pct, 2)
                kpis["promoter_pct"] = promoter_pct
                kpis["passive_pct"] = passive_pct
                kpis["detractor_pct"] = detractor_pct

        rating_col = self._find_column_by_patterns(
            SURVEY_SCORE_PATTERNS, within=non_monetary_numeric
        )
        if rating_col and pd.api.types.is_numeric_dtype(self.df[rating_col]):
            ratings = pd.to_numeric(self.df[rating_col], errors="coerce").dropna()
            if not ratings.empty:
                kpis[f"avg_{rating_col}"] = round(ratings.mean(), 2)
                # Heuristic positive threshold for common rating scales.
                if ratings.max() <= 5:
                    positive_pct = round((ratings >= 4).mean() * 100, 2)
                elif ratings.max() <= 10:
                    positive_pct = round((ratings >= 8).mean() * 100, 2)
                else:
                    positive_pct = round((ratings >= ratings.quantile(0.75)).mean() * 100, 2)
                kpis[f"positive_{rating_col}_pct"] = positive_pct

        # Categorical KPIs
        for col in self._categorical_columns[:3]:
            kpis[f"unique_{col}"] = int(self.df[col].nunique())
            top_val = self.df[col].mode().iloc[0] if len(self.df[col].mode()) > 0 else "N/A"
            kpis[f"top_{col}"] = str(top_val)

        # Date range
        if self._date_columns:
            date_col = self._date_columns[0]
            try:
                dates = pd.to_datetime(self.df[date_col])
                kpis["date_range_start"] = str(dates.min().date())
                kpis["date_range_end"] = str(dates.max().date())
            except Exception:
                pass

        return kpis

    def get_chart_recommendations(self) -> List[Dict[str, Any]]:
        """
        Recommend appropriate chart types based on detected schema.
        Multi-metric: explores multiple column combinations.

        Returns:
            List of dicts with 'type', 'x', 'y', 'title' (and optional extra keys).
        """
        charts: List[Dict[str, Any]] = []
        primary_metric = self.get_primary_metric()
        primary_category = self.get_primary_category()
        primary_date = self.get_primary_date()
        nps_col = self._find_column_by_patterns(["nps", "nps_score", "net_promoter"])

        # ── helpers ──────────────────────────────────────────────
        def _enough_num(col, min_pts=5, min_uniq=2):
            if not col or col not in self.df.columns:
                return False
            v = pd.to_numeric(self.df[col], errors="coerce").dropna()
            return len(v) >= min_pts and v.nunique() >= min_uniq

        def _enough_cat(col, min_pts=5, min_uniq=2):
            if not col or col not in self.df.columns:
                return False
            v = self.df[col].dropna().astype(str)
            return len(v) >= min_pts and v.nunique() >= min_uniq

        def _enough_date(col, min_pts=5, min_uniq=3):
            if not col or col not in self.df.columns:
                return False
            p = pd.to_datetime(self.df[col], errors="coerce").dropna()
            return len(p) >= min_pts and p.nunique() >= min_uniq

        def _label(col):
            return col.replace("_", " ").title() if col else ""

        def add(chart):
            sig = (chart.get("type"), chart.get("x"), chart.get("y"))
            for c in charts:
                if (c.get("type"), c.get("x"), c.get("y")) == sig:
                    return
            charts.append(chart)

        # Build ranked numeric candidates
        num_cands = []
        for col in self._numeric_columns:
            v = pd.to_numeric(self.df[col], errors="coerce").dropna()
            if len(v) >= 8 and v.nunique() >= 3:
                num_cands.append((col, float(v.std()) if v.nunique() > 1 else 0.0))
        num_cands.sort(key=lambda x: x[1], reverse=True)

        # Usable categories (2-50 unique values)
        good_cats = [c for c in self._categorical_columns
                     if 2 <= self.df[c].nunique() <= 50]

        # Metrics to iterate: monetary first, then top numeric
        iter_metrics = list(dict.fromkeys(
            self._monetary_columns[:3]
            + [c for c, _ in num_cands[:4] if c not in self._monetary_columns]
        ))

        # ── 1. NPS / survey histogram ────────────────────────────
        if _enough_num(nps_col, 6, 3):
            add({"type": "histogram", "x": nps_col, "y": None,
                 "title": f"Distribution of {_label(nps_col)}"})

        # ── 2. Time-series lines (multi-metric) ─────────────────
        for date_col in self._date_columns[:2]:
            if not _enough_date(date_col):
                continue
            for metric in iter_metrics[:3]:
                if _enough_num(metric):
                    add({"type": "line", "x": date_col, "y": metric,
                         "title": f"{_label(metric)} Over Time"})

        # ── 3. Bar charts (multi-category × multi-metric) ───────
        for cat in good_cats[:3]:
            for metric in iter_metrics[:2]:
                if _enough_num(metric):
                    add({"type": "bar", "x": cat, "y": metric,
                         "title": f"{_label(metric)} by {_label(cat)}"})

        # ── 4. Stacked area (composition over time) ─────────────
        if primary_date and _enough_date(primary_date) and good_cats:
            best_cat = good_cats[0]
            metric_for_stack = iter_metrics[0] if iter_metrics else None
            if metric_for_stack and _enough_num(metric_for_stack):
                add({"type": "stacked_area", "x": primary_date,
                     "y": metric_for_stack, "color": best_cat,
                     "title": f"{_label(metric_for_stack)} Composition Over Time"})

        # ── 5. Distribution histograms ───────────────────────────
        for metric in iter_metrics[:2]:
            if _enough_num(metric, 8, 3):
                add({"type": "histogram", "x": metric, "y": None,
                     "title": f"Distribution of {_label(metric)}"})

        # ── 6. Scatter plots (top numeric pairs) ────────────────
        if len(num_cands) >= 2:
            pairs_added = 0
            for i in range(len(num_cands)):
                for j in range(i + 1, len(num_cands)):
                    if pairs_added >= 2:
                        break
                    x_c, y_c = num_cands[i][0], num_cands[j][0]
                    add({"type": "scatter", "x": x_c, "y": y_c,
                         "title": f"{_label(x_c)} vs {_label(y_c)}"})
                    pairs_added += 1
                if pairs_added >= 2:
                    break

        # ── 7. Pie chart ─────────────────────────────────────────
        if _enough_cat(primary_category):
            add({"type": "pie", "x": primary_category,
                 "y": primary_metric or "count",
                 "title": f"{_label(primary_category)} Distribution"})

        # ── 8. Box plot (numeric by category) ────────────────────
        if good_cats and num_cands:
            add({"type": "box", "x": good_cats[0], "y": num_cands[0][0],
                 "title": f"{_label(num_cands[0][0])} by {_label(good_cats[0])}"})

        # ── 9. Correlation heatmap (3+ numeric columns) ─────────
        if len(self._numeric_columns) >= 3:
            add({"type": "heatmap", "x": None, "y": None,
                 "columns": self._numeric_columns[:12],
                 "title": "Correlation Between Numeric Variables"})

        # ── 10. Treemap (hierarchical categories) ────────────────
        if len(good_cats) >= 2 and iter_metrics:
            add({"type": "treemap",
                 "path": good_cats[:2], "y": iter_metrics[0],
                 "title": f"{_label(iter_metrics[0])} by {_label(good_cats[0])} → {_label(good_cats[1])}"})

        # ── 11. Funnel chart (marketing conversion) ──────────────
        funnel_cols = []
        for stage_patterns in FUNNEL_PATTERNS:
            found = self._find_column_by_patterns(stage_patterns,
                                                   within=self._numeric_columns)
            if found:
                funnel_cols.append(found)
        if len(funnel_cols) >= 2:
            add({"type": "funnel", "columns": funnel_cols,
                 "title": "Conversion Funnel"})

        # ── 12. Waterfall (financial) ────────────────────────────
        if self.detected_type == "financial" and iter_metrics:
            if good_cats:
                add({"type": "waterfall", "x": good_cats[0],
                     "y": iter_metrics[0],
                     "title": f"{_label(iter_metrics[0])} Waterfall"})

        return charts

    def get_analysis_context(self) -> str:
        """
        Generate a comprehensive context string for LLM analysis.
        Works with ANY dataset.
        """
        lines = [
            f"Dataset: {self.name}",
            f"Detected Type: {self.detected_type} (confidence: {self.confidence:.0%})",
            f"Shape: {self.df.shape[0]} rows × {self.df.shape[1]} columns",
            "",
            "Column Classification:",
        ]

        if self._date_columns:
            lines.append(f"  Date columns: {', '.join(self._date_columns)}")
        if self._monetary_columns:
            lines.append(f"  Monetary columns: {', '.join(self._monetary_columns)}")
        non_mon = [c for c in self._numeric_columns if c not in self._monetary_columns]
        if non_mon:
            lines.append(f"  Numeric columns: {', '.join(non_mon)}")
        if self._categorical_columns:
            lines.append(f"  Categorical columns: {', '.join(self._categorical_columns)}")

        lines.append("")
        lines.append("Key Metrics:")
        kpis = self.compute_auto_kpis()
        for k, v in kpis.items():
            if k not in ["total_records", "total_columns", "detected_type", "detection_confidence"]:
                label = k.replace("_", " ").title()
                if isinstance(v, float) and v > 1000:
                    lines.append(f"  - {label}: {v:,.2f}")
                else:
                    lines.append(f"  - {label}: {v}")

        lines.append("")
        lines.append("Sample Data:")
        lines.append(self.df.head(8).to_string(index=False))

        lines.append("")
        lines.append("Descriptive Statistics:")
        lines.append(self.df.describe().round(2).to_string())

        return "\n".join(lines)

    def get_detection_summary(self) -> Dict[str, Any]:
        """Get a complete summary of what was detected."""
        return {
            "name": self.name,
            "detected_type": self.detected_type,
            "secondary_type": self.secondary_type,
            "confidence": self.confidence,
            "shape": {"rows": len(self.df), "columns": len(self.df.columns)},
            "date_columns": self._date_columns,
            "monetary_columns": self._monetary_columns,
            "rate_columns": self._rate_columns,
            "numeric_columns": self._numeric_columns,
            "categorical_columns": self._categorical_columns,
            "primary_metric": self.get_primary_metric(),
            "primary_category": self.get_primary_category(),
            "primary_date": self.get_primary_date(),
            "recommended_charts": len(self.get_chart_recommendations())
        }


def detect_dataset(df: pd.DataFrame, name: str = "dataset") -> DatasetDetector:
    """Convenience function to create a DatasetDetector."""
    return DatasetDetector(df, name)


def detect_all_datasets(datasets: Dict[str, pd.DataFrame]) -> Dict[str, DatasetDetector]:
    """Detect types for all datasets in a dictionary."""
    return {name: DatasetDetector(df, name) for name, df in datasets.items()}
