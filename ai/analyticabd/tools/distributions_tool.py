from .base_tool import BaseTool
import pandas as pd
import numpy as np


class DistributionsTool(BaseTool):
    def get_description(self):
        return "Analyze variable distributions: summary stats, skewness/kurtosis, and normality hints for numeric columns."

    def get_parameters_schema(self):
        return {
            "top_n": {"type": "integer", "default": 6},
            "include_histograms": {"type": "boolean", "default": False},
            "bins": {"type": "integer", "default": 20},
        }

    def execute(self, parameters=None):
        params = parameters or {}
        top_n = int(params.get("top_n", 6))
        include_hist = bool(params.get("include_histograms", False))
        bins = int(params.get("bins", 20))

        cached = self.cache_manager.get_cached_tool_results(self.tool_name, self.dataset_id, params)
        if cached:
            return cached

        self.load_dataset()

        numeric_cols = self.dataset.select_dtypes(include=[np.number]).columns.tolist()
        result = {
            "overview": {
                "total_numeric": len(numeric_cols),
            },
            "variables": {},
            "interpretation": "",
        }

        # Choose top_n numeric columns by variance (descending)
        ranked_cols = sorted(
            numeric_cols,
            key=lambda c: float(self.dataset[c].dropna().var()) if len(self.dataset[c].dropna()) > 0 else -1,
            reverse=True,
        )[: top_n]

        interpretations = []
        for col in ranked_cols:
            series = self.dataset[col].dropna()
            if series.empty:
                continue
            mean = float(series.mean())
            std = float(series.std()) if not np.isnan(series.std()) else None
            skew = float(series.skew()) if not np.isnan(series.skew()) else None
            kurt = float(series.kurtosis()) if not np.isnan(series.kurtosis()) else None
            var_info = {
                "count": int(series.count()),
                "mean": mean,
                "std": std,
                "min": float(series.min()),
                "q25": float(series.quantile(0.25)),
                "median": float(series.median()),
                "q75": float(series.quantile(0.75)),
                "max": float(series.max()),
                "skewness": skew,
                "kurtosis": kurt,
                "distribution_type": "normal" if (skew is not None and abs(skew) < 0.5) else "skewed",
            }

            if include_hist:
                try:
                    counts, bin_edges = np.histogram(series.values, bins=bins)
                    var_info["histogram"] = {
                        "bins": [float(b) for b in bin_edges.tolist()],
                        "counts": [int(c) for c in counts.tolist()],
                    }
                except Exception:
                    pass

            result["variables"][str(col)] = var_info

            if skew is not None:
                if abs(skew) >= 1:
                    interpretations.append(f"{col} is highly skewed (skew={skew:.2f}).")
                elif abs(skew) >= 0.5:
                    interpretations.append(f"{col} is moderately skewed (skew={skew:.2f}).")

        if not interpretations:
            result["interpretation"] = "Most numeric variables appear approximately normal (low skew)."
        else:
            result["interpretation"] = " ".join(interpretations[:6])

        self.cache_manager.cache_tool_results(self.tool_name, self.dataset_id, params, result)
        return result


