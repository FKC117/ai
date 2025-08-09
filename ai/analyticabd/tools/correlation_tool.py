from .base_tool import BaseTool
import pandas as pd
import numpy as np


class CorrelationTool(BaseTool):
    def get_description(self):
        return "Compute correlation matrix, identify strong correlations, and provide brief interpretations."

    def get_parameters_schema(self):
        return {
            "threshold": {"type": "number", "default": 0.7},
            "method": {"type": "string", "enum": ["pearson", "spearman"], "default": "pearson"}
        }

    def execute(self, parameters=None):
        params = parameters or {}
        threshold = float(params.get("threshold", 0.7))
        method = params.get("method", "pearson")

        cached_results = self.cache_manager.get_cached_tool_results(
            self.tool_name, self.dataset_id, params
        )
        if cached_results:
            return cached_results

        self.load_dataset()

        numeric_cols = self.dataset.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            result = {"matrix": {}, "strong_correlations": [], "interpretation": "Not enough numeric variables for correlation."}
            self.cache_manager.cache_tool_results(self.tool_name, self.dataset_id, params, result)
            return result

        corr = self.dataset[numeric_cols].corr(method=method)

        strong = []
        for i in range(len(corr.columns)):
            for j in range(i + 1, len(corr.columns)):
                r = corr.iloc[i, j]
                if pd.notna(r) and abs(r) >= threshold:
                    strong.append({
                        "variable1": str(corr.columns[i]),
                        "variable2": str(corr.columns[j]),
                        "correlation": float(r),
                    })

        # Simple interpretation
        if strong:
            pairs = ", ".join([f"{s['variable1']}↔{s['variable2']} (r={s['correlation']:.2f})" for s in strong[:5]])
            interpretation = f"Top strong correlations: {pairs}."
        else:
            interpretation = "No strong correlations detected at the chosen threshold."

        result = {
            "matrix": {str(r): {str(c): (None if pd.isna(v) else float(v)) for c, v in corr.loc[r].items()} for r in corr.index},
            "strong_correlations": strong,
            "interpretation": interpretation,
            "method": method,
            "threshold": threshold,
        }

        self.cache_manager.cache_tool_results(self.tool_name, self.dataset_id, params, result)
        return result


