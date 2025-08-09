from .base_tool import BaseTool
import pandas as pd
import numpy as np


class OutliersTool(BaseTool):
    def get_description(self):
        return "Detect outliers in numeric columns using the IQR method; returns counts, percentages, and bounds."

    def get_parameters_schema(self):
        return {
            "iqr_multiplier": {"type": "number", "default": 1.5},
            "top_n": {"type": "integer", "default": 12},
        }

    def execute(self, parameters=None):
        params = parameters or {}
        k = float(params.get("iqr_multiplier", 1.5))
        top_n = int(params.get("top_n", 12))

        cached = self.cache_manager.get_cached_tool_results(self.tool_name, self.dataset_id, params)
        if cached:
            return cached

        self.load_dataset()
        numeric_cols = self.dataset.select_dtypes(include=[np.number]).columns.tolist()

        results = {}
        ranking = []
        for col in numeric_cols:
            series = pd.to_numeric(self.dataset[col], errors='coerce').dropna()
            if len(series) == 0:
                continue
            q1 = float(series.quantile(0.25))
            q3 = float(series.quantile(0.75))
            iqr = q3 - q1
            if iqr <= 0:
                lower = q1
                upper = q3
                mask = pd.Series([False] * len(series), index=series.index)
            else:
                lower = q1 - k * iqr
                upper = q3 + k * iqr
                mask = (series < lower) | (series > upper)
            count = int(mask.sum())
            pct = float(count / max(1, len(series)))
            results[str(col)] = {
                "count": count,
                "percentage": pct,
                "lower_bound": float(lower),
                "upper_bound": float(upper),
            }
            ranking.append((col, count))

        ranking.sort(key=lambda x: x[1], reverse=True)
        interpretation = (
            "; ".join([f"{c}: {n} outliers" for c, n in ranking[:5]]) if ranking else "No outliers detected."
        )

        payload = {
            "outliers": results,
            "rank": ranking[:top_n],
            "interpretation": interpretation,
            "iqr_multiplier": k,
        }

        self.cache_manager.cache_tool_results(self.tool_name, self.dataset_id, params, payload)
        return payload


