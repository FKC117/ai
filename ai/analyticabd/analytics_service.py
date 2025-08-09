from typing import Optional, Any
import os
import pandas as pd
import numpy as np

from .models import UserDataset


def read_dataset_file(dataset: UserDataset) -> pd.DataFrame:
    """Read a dataset file into a pandas DataFrame with basic type detection.
    Raises FileNotFoundError or ValueError for unsupported types.
    """
    file_path = dataset.file.path
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    file_extension = os.path.splitext(dataset.file.name)[1].lower()

    if file_extension == ".csv":
        return pd.read_csv(file_path)
    if file_extension in (".xlsx", ".xls"):
        return pd.read_excel(file_path)
    if file_extension == ".json":
        return pd.read_json(file_path)

    raise ValueError(f"Unsupported file type: {file_extension}")


def create_comprehensive_summary_table(summary_stats: dict) -> str:
    """Create a comprehensive HTML table for summary statistics."""
    try:
        var_items = list(summary_stats.get('variable_summary', {}).items())
        if not var_items:
            return ""

        html_parts = []
        html_parts.append('<table style="border-collapse: collapse; width: 100%; margin: 10px 0; font-family: Inter, sans-serif;">')
        html_parts.append('<thead>')
        html_parts.append('<tr style="background-color: #1a365d; color: white;">')
        html_parts.append('<th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Variable</th>')
        html_parts.append('<th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Type</th>')
        html_parts.append('<th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Count</th>')
        html_parts.append('<th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Missing %</th>')
        html_parts.append('<th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Mean</th>')
        html_parts.append('<th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Std Dev</th>')
        html_parts.append('<th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Min</th>')
        html_parts.append('<th style="border: 1px solid #ddd; padding: 8px; text-align: left;">25th %</th>')
        html_parts.append('<th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Median</th>')
        html_parts.append('<th style="border: 1px solid #ddd; padding: 8px; text-align: left;">75th %</th>')
        html_parts.append('<th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Max</th>')
        html_parts.append('<th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Quality Score</th>')
        html_parts.append('</tr>')
        html_parts.append('</thead>')
        html_parts.append('<tbody>')

        for idx, (name, meta) in enumerate(var_items):
            dq_info = summary_stats.get('data_quality', {}).get(name, {})
            missing_pct = dq_info.get('missing_percentage', 0)
            quality_score = dq_info.get('quality_score', 1.0)
            bg_color = '#f7fafc' if idx % 2 == 1 else '#ffffff'

            if meta.get('type') == 'numeric':
                mean_val = meta.get('mean', 'N/A')
                std_val = meta.get('std', 'N/A')
                min_val = meta.get('min', 'N/A')
                q25_val = meta.get('q25', 'N/A')
                median_val = meta.get('median', 'N/A')
                q75_val = meta.get('q75', 'N/A')
                max_val = meta.get('max', 'N/A')

                if isinstance(mean_val, (int, float)): mean_val = f"{mean_val:.2f}"
                if isinstance(std_val, (int, float)): std_val = f"{std_val:.2f}"
                if isinstance(min_val, (int, float)): min_val = f"{min_val:.2f}"
                if isinstance(q25_val, (int, float)): q25_val = f"{q25_val:.2f}"
                if isinstance(median_val, (int, float)): median_val = f"{median_val:.2f}"
                if isinstance(q75_val, (int, float)): q75_val = f"{q75_val:.2f}"
                if isinstance(max_val, (int, float)): max_val = f"{max_val:.2f}"

                html_parts.append(f'<tr style="background-color: {bg_color};">')
                html_parts.append(f'<td style="border: 1px solid #ddd; padding: 8px; font-weight: bold;">{name}</td>')
                html_parts.append(f'<td style="border: 1px solid #ddd; padding: 8px;">numeric</td>')
                html_parts.append(f'<td style="border: 1px solid #ddd; padding: 8px;">{meta.get("count", "N/A")}</td>')
                html_parts.append(f'<td style="border: 1px solid #ddd; padding: 8px;">{missing_pct:.1%}</td>')
                html_parts.append(f'<td style="border: 1px solid #ddd; padding: 8px;">{mean_val}</td>')
                html_parts.append(f'<td style="border: 1px solid #ddd; padding: 8px;">{std_val}</td>')
                html_parts.append(f'<td style="border: 1px solid #ddd; padding: 8px;">{min_val}</td>')
                html_parts.append(f'<td style="border: 1px solid #ddd; padding: 8px;">{q25_val}</td>')
                html_parts.append(f'<td style="border: 1px solid #ddd; padding: 8px;">{median_val}</td>')
                html_parts.append(f'<td style="border: 1px solid #ddd; padding: 8px;">{q75_val}</td>')
                html_parts.append(f'<td style="border: 1px solid #ddd; padding: 8px;">{max_val}</td>')
                html_parts.append(f'<td style="border: 1px solid #ddd; padding: 8px;">{quality_score:.2f}</td>')
                html_parts.append('</tr>')
            else:
                unique_count = meta.get('unique_count', 'N/A')
                most_common = meta.get('most_common', 'N/A')
                most_common_count = meta.get('most_common_count', 'N/A')

                html_parts.append(f'<tr style="background-color: {bg_color};">')
                html_parts.append(f'<td style="border: 1px solid #ddd; padding: 8px; font-weight: bold;">{name}</td>')
                html_parts.append(f'<td style="border: 1px solid #ddd; padding: 8px;">categorical</td>')
                html_parts.append(f'<td style="border: 1px solid #ddd; padding: 8px;">{meta.get("count", "N/A")}</td>')
                html_parts.append(f'<td style="border: 1px solid #ddd; padding: 8px;">{missing_pct:.1%}</td>')
                html_parts.append(f'<td style="border: 1px solid #ddd; padding: 8px;" colspan="6">{unique_count} unique values</td>')
                html_parts.append(f'<td style="border: 1px solid #ddd; padding: 8px;" colspan="2">Most common: {most_common} ({most_common_count})</td>')
                html_parts.append(f'<td style="border: 1px solid #ddd; padding: 8px;">{quality_score:.2f}</td>')
                html_parts.append('</tr>')

        html_parts.append('</tbody>')
        html_parts.append('</table>')
        return '\n'.join(html_parts)
    except Exception:
        return ""


def get_summary_statistics_data(dataset_id: int) -> dict:
    """Compute summary stats and attach comprehensive HTML table."""
    dataset = UserDataset.objects.get(id=dataset_id)
    df = read_dataset_file(dataset)

    summary_stats: dict = {
        'dataset_overview': {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            # Convert dtypes (which include ObjectDType) to strings for JSON serialization
            'data_types': {str(col): str(dtype) for col, dtype in df.dtypes.to_dict().items()},
            'sample_data': df.head(5).to_dict('records'),
        },
        'variable_summary': {},
        'data_quality': {},
        'distribution_insights': {},
        'correlation_matrix': {},
        'outlier_analysis': {},
        'missing_data_analysis': {},
        'comprehensive_table': {},
    }

    # Variables
    for col in df.columns:
        col_data = df[col]
        if pd.api.types.is_numeric_dtype(col_data):
            non_null = col_data.dropna()
            if len(non_null) == 0:
                summary_stats['variable_summary'][col] = {
                    'type': 'numeric', 'count': col_data.count(), 'mean': None, 'std': None,
                    'min': None, 'q25': None, 'median': None, 'q75': None, 'max': None,
                    'skewness': None, 'kurtosis': None,
                }
            else:
                summary_stats['variable_summary'][col] = {
                    'type': 'numeric',
                    'count': col_data.count(),
                    'mean': float(non_null.mean()),
                    'std': float(non_null.std()),
                    'min': float(non_null.min()),
                    'q25': float(non_null.quantile(0.25)),
                    'median': float(non_null.median()),
                    'q75': float(non_null.quantile(0.75)),
                    'max': float(non_null.max()),
                    'skewness': float(non_null.skew()),
                    'kurtosis': float(non_null.kurtosis()),
                }
        else:
            non_null = col_data.dropna()
            value_counts = non_null.value_counts()
            summary_stats['variable_summary'][col] = {
                'type': 'categorical',
                'count': col_data.count(),
                'unique_count': col_data.nunique(),
                'most_common': str(value_counts.index[0]) if len(value_counts) > 0 else None,
                'most_common_count': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
            }

    # Data quality
    for col in df.columns:
        missing_pct = df[col].isnull().sum() / max(len(df), 1)
        summary_stats['data_quality'][col] = {
            'missing_percentage': float(missing_pct),
            'completeness': float(1 - missing_pct),
            'quality_score': float(1 - missing_pct if missing_pct < 0.1 else 0.5),
        }

    # Correlations
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 1:
        corr = df[numeric_cols].corr()
        # Convert correlation matrix to pure Python floats (avoid numpy types for JSON)
        matrix: dict = {}
        for row in corr.index:
            row_dict: dict = {}
            for col in corr.columns:
                val = corr.at[row, col]
                row_dict[str(col)] = None if pd.isna(val) else float(val)
            matrix[str(row)] = row_dict
        summary_stats['correlation_matrix'] = {
            'matrix': matrix,
            'strong_correlations': [],
        }
        for i in range(len(corr.columns)):
            for j in range(i + 1, len(corr.columns)):
                corr_value = corr.iloc[i, j]
                if abs(corr_value) >= 0.7:
                    summary_stats['correlation_matrix']['strong_correlations'].append(
                        {
                            'variable1': str(corr.columns[i]),
                            'variable2': str(corr.columns[j]),
                            'correlation': float(corr_value),
                        }
                    )

    # Outlier analysis (IQR) for numeric columns
    outlier_analysis: dict = {}
    for col in numeric_cols:
        col_data = pd.to_numeric(df[col], errors='coerce').dropna()
        if len(col_data) == 0:
            continue
        q1 = float(col_data.quantile(0.25))
        q3 = float(col_data.quantile(0.75))
        iqr = q3 - q1
        if iqr <= 0:
            lower = q1
            upper = q3
            mask = pd.Series([False] * len(col_data), index=col_data.index)
        else:
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            mask = (col_data < lower) | (col_data > upper)
        count = int(mask.sum())
        percentage = float(count / max(1, len(col_data)))
        outlier_analysis[str(col)] = {
            'count': count,
            'percentage': percentage,
            'lower_bound': float(lower),
            'upper_bound': float(upper),
        }
    summary_stats['outlier_analysis'] = outlier_analysis

    # Comprehensive HTML table
    summary_stats['comprehensive_table'] = create_comprehensive_summary_table(summary_stats)

    # Ensure all values are JSON serializable (convert numpy types, NaN -> None)
    def make_json_safe(value: Any) -> Any:
        # Scalars
        if isinstance(value, (np.integer, )):
            return int(value)
        if isinstance(value, (np.floating, )):
            # Convert NaN/Inf to None or finite float
            if pd.isna(value) or np.isinf(value):
                return None
            return float(value)
        if isinstance(value, (np.bool_, )):
            return bool(value)
        if value is None:
            return None
        # Containers
        if isinstance(value, dict):
            return {str(k): make_json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [make_json_safe(v) for v in value]
        # Pandas NA
        try:
            if pd.isna(value):
                return None
        except Exception:
            pass
        # Fallback: cast to str for unsupported types
        return value

    return make_json_safe(summary_stats)
