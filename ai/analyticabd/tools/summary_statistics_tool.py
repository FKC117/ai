from .base_tool import BaseTool
import pandas as pd
import numpy as np

class SummaryStatisticsTool(BaseTool):
    def get_description(self):
        return "Generate comprehensive summary statistics including descriptive stats, data quality metrics, and distribution insights"
    
    def get_parameters_schema(self):
        return {
            "include_visualizations": {"type": "boolean", "default": True},
            "outlier_threshold": {"type": "float", "default": 1.5},
            "missing_threshold": {"type": "float", "default": 0.1}
        }
    
    def execute(self, parameters=None):
        cached_results = self.cache_manager.get_cached_tool_results(
            self.tool_name, self.dataset_id, parameters
        )
        if cached_results:
            return cached_results
        
        self.load_dataset()
        
        summary = {
            'dataset_overview': self._get_dataset_overview(),
            'variable_summary': self._get_variable_summary(),
            'data_quality': self._get_data_quality(parameters),
            'distribution_insights': self._get_distribution_insights(parameters),
            'correlation_matrix': self._get_correlation_matrix(),
            'outlier_analysis': self._get_outlier_analysis(parameters),
            'missing_data_analysis': self._get_missing_data_analysis(),
            'statistical_tests': self._get_statistical_tests()
        }
        
        self.cache_manager.cache_tool_results(
            self.tool_name, self.dataset_id, parameters, summary
        )
        
        return summary
    
    def _get_dataset_overview(self):
        return {
            'total_rows': len(self.dataset),
            'total_columns': len(self.dataset.columns),
            'memory_usage': self.dataset.memory_usage(deep=True).sum(),
            'data_types': self.dataset.dtypes.to_dict(),
            'sample_data': self.dataset.head(5).to_dict('records')
        }
    
    def _get_variable_summary(self):
        summary = {}
        for col in self.dataset.columns:
            col_data = self.dataset[col]
            if pd.api.types.is_numeric_dtype(col_data):
                summary[col] = {
                    'type': 'numeric',
                    'count': col_data.count(),
                    'mean': col_data.mean(),
                    'std': col_data.std(),
                    'min': col_data.min(),
                    'q25': col_data.quantile(0.25),
                    'median': col_data.median(),
                    'q75': col_data.quantile(0.75),
                    'max': col_data.max(),
                    'skewness': col_data.skew(),
                    'kurtosis': col_data.kurtosis()
                }
            else:
                summary[col] = {
                    'type': 'categorical',
                    'count': col_data.count(),
                    'unique_count': col_data.nunique(),
                    'most_common': col_data.mode().iloc[0] if not col_data.mode().empty else None,
                    'most_common_count': col_data.value_counts().iloc[0] if not col_data.empty else 0
                }
        return summary
    
    def _get_data_quality(self, parameters):
        missing_threshold = parameters.get('missing_threshold', 0.1) if parameters else 0.1
        
        quality_metrics = {}
        for col in self.dataset.columns:
            missing_pct = self.dataset[col].isnull().sum() / len(self.dataset)
            quality_metrics[col] = {
                'missing_percentage': missing_pct,
                'completeness': 1 - missing_pct,
                'quality_score': 1 - missing_pct if missing_pct < missing_threshold else 0.5,
                'issues': []
            }
            
            if missing_pct > missing_threshold:
                quality_metrics[col]['issues'].append('High missing data')
            
            if pd.api.types.is_numeric_dtype(self.dataset[col]):
                Q1 = self.dataset[col].quantile(0.25)
                Q3 = self.dataset[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = ((self.dataset[col] < (Q1 - 1.5 * IQR)) | 
                           (self.dataset[col] > (Q3 + 1.5 * IQR))).sum()
                quality_metrics[col]['outliers_count'] = outliers
                quality_metrics[col]['outliers_percentage'] = outliers / len(self.dataset)
        
        return quality_metrics
    
    def _get_distribution_insights(self, parameters):
        insights = {}
        for col in self.dataset.columns:
            if pd.api.types.is_numeric_dtype(self.dataset[col]):
                col_data = self.dataset[col].dropna()
                if len(col_data) > 0:
                    insights[col] = {
                        'distribution_type': 'normal' if abs(col_data.skew()) < 0.5 else 'skewed',
                        'skewness': col_data.skew(),
                        'kurtosis': col_data.kurtosis(),
                        'range': col_data.max() - col_data.min(),
                        'coefficient_of_variation': col_data.std() / col_data.mean() if col_data.mean() != 0 else 0
                    }
        return insights
    
    def _get_correlation_matrix(self):
        numeric_cols = self.dataset.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            correlation_matrix = self.dataset[numeric_cols].corr()
            return {
                'matrix': correlation_matrix.to_dict(),
                'strong_correlations': self._find_strong_correlations(correlation_matrix)
            }
        return {'matrix': {}, 'strong_correlations': []}
    
    def _find_strong_correlations(self, corr_matrix, threshold=0.7):
        strong_correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) >= threshold:
                    strong_correlations.append({
                        'variable1': corr_matrix.columns[i],
                        'variable2': corr_matrix.columns[j],
                        'correlation': corr_value
                    })
        return strong_correlations
    
    def _get_outlier_analysis(self, parameters):
        outlier_threshold = parameters.get('outlier_threshold', 1.5) if parameters else 1.5
        outliers = {}
        
        for col in self.dataset.select_dtypes(include=[np.number]).columns:
            col_data = self.dataset[col].dropna()
            if len(col_data) > 0:
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - outlier_threshold * IQR
                upper_bound = Q3 + outlier_threshold * IQR
                
                outlier_indices = ((col_data < lower_bound) | (col_data > upper_bound))
                outliers[col] = {
                    'count': outlier_indices.sum(),
                    'percentage': outlier_indices.sum() / len(col_data),
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound,
                    'outlier_values': col_data[outlier_indices].tolist()
                }
        
        return outliers
    
    def _get_missing_data_analysis(self):
        missing_analysis = {}
        for col in self.dataset.columns:
            missing_count = self.dataset[col].isnull().sum()
            missing_analysis[col] = {
                'missing_count': missing_count,
                'missing_percentage': missing_count / len(self.dataset),
                'completeness': 1 - (missing_count / len(self.dataset))
            }
        return missing_analysis
    
    def _get_statistical_tests(self):
        tests = {}
        numeric_cols = self.dataset.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            col_data = self.dataset[col].dropna()
            if len(col_data) > 30:
                from scipy import stats
                shapiro_stat, shapiro_p = stats.shapiro(col_data)
                tests[col] = {
                    'normality_test': {
                        'test_name': 'Shapiro-Wilk',
                        'statistic': shapiro_stat,
                        'p_value': shapiro_p,
                        'is_normal': shapiro_p > 0.05
                    }
                }
        
        return tests
