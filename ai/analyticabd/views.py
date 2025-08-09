from django.shortcuts import render, redirect
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from .forms import CustomUserCreationForm, CustomAuthenticationForm
from .models import UserDataset, DatasetVariable, AnalysisSession, AnalysisInteraction, UserPreference, AnalysisHistory, UserWarningPreference, ReportDocument, ReportSection, ChatMessage, DatasetUIState
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from io import BytesIO
from django.http import HttpResponse
import re
from bs4 import BeautifulSoup, NavigableString

try:
    from docx import Document
    from docx.shared import Inches, Pt
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import RGBColor
    DOCX_AVAILABLE = True
except Exception:
    DOCX_AVAILABLE = False

from .session_manager import (
    set_current_dataset_and_session,
    get_current_session_id_and_ui_state,
    save_dataset_ui_state,
    resolve_session_for,
    get_chat_history_for_session,
    get_user_quota_status,
    get_user_billing_summary as build_billing_summary,
)
from .analytics_service import get_summary_statistics_data
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from .ai.cache_manager import CacheManager

# Create your views here.
def home(request):
    return render(request, 'homepage.html')

def account_management(request):
    return render(request, 'account_management.html')

@login_required
def analytics(request):
    return render(request, 'analytics_dashboard.html')

def getting_started(request):
    return render(request, 'getting_started.html')

def pricing_plans(request):
    return render(request, 'pricing_plans.html')

def signup(request):
    if request.method == 'POST':
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            messages.success(request, 'Account created successfully!')
            # Get the 'next' parameter to redirect to the originally intended page
            next_url = request.GET.get('next')
            if next_url:
                return redirect(next_url)
            else:
                return redirect('homepage')
        else:
            messages.error(request, 'Please correct the errors below.')
    else:
        form = CustomUserCreationForm()
    
    return render(request, 'signup.html', {'form': form})

def signin(request):
    if request.method == 'POST':
        form = CustomAuthenticationForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)
                messages.success(request, f'Welcome back, {user.username}!')
                # Get the 'next' parameter to redirect to the originally intended page
                next_url = request.GET.get('next')
                if next_url:
                    return redirect(next_url)
                else:
                    return redirect('homepage')
    else:
        form = CustomAuthenticationForm()
    
    return render(request, 'signin.html', {'form': form})

def signout(request):
    logout(request)
    messages.success(request, 'You have been logged out successfully.')
    return redirect('homepage')

@login_required
def profile(request):
    return render(request, 'profile.html')

@login_required
@csrf_exempt
@require_http_methods(["POST"])
def upload_dataset(request):
    """Handle dataset upload and process it for summary statistics"""
    try:
        if 'file' not in request.FILES:
            return JsonResponse({'error': 'No file uploaded'}, status=400)
        
        uploaded_file = request.FILES['file']
        
        # Validate file type
        allowed_extensions = ['.csv', '.xlsx', '.xls', '.json']
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        if file_extension not in allowed_extensions:
            return JsonResponse({'error': 'Invalid file type. Please upload CSV, Excel, or JSON files.'}, status=400)
        
        # Validate file size (50MB limit)
        if uploaded_file.size > 50 * 1024 * 1024:
            return JsonResponse({'error': 'File size must be less than 50MB'}, status=400)
        
        # Read the file based on its type
        try:
            if file_extension == '.csv':
                df = pd.read_csv(uploaded_file)
            elif file_extension in ['.xlsx', '.xls']:
                df = pd.read_excel(uploaded_file)
            elif file_extension == '.json':
                df = pd.read_json(uploaded_file)
            else:
                return JsonResponse({'error': 'Unsupported file format'}, status=400)
        except Exception as e:
            return JsonResponse({'error': f'Error reading file: {str(e)}'}, status=400)
        
        # Calculate basic statistics
        rows, cols = df.shape
        numeric_cols = df.select_dtypes(include=[np.number]).shape[1]
        categorical_cols = df.select_dtypes(include=['object']).shape[1]
        
        # Calculate missing values percentage
        missing_percentage = (df.isnull().sum().sum() / (rows * cols)) * 100 if rows * cols > 0 else 0
        
        # Calculate duplicate rows
        duplicate_rows = df.duplicated().sum()
        
        # Create UserDataset object
        dataset = UserDataset.objects.create(
            user=request.user,
            name=uploaded_file.name,
            file=uploaded_file,
            file_size=uploaded_file.size,
            rows=rows,
            columns=cols,
            numeric_columns=numeric_cols,
            categorical_columns=categorical_cols,
            missing_values_percentage=missing_percentage,
            duplicate_rows=duplicate_rows
        )
        
        # Process each variable and create DatasetVariable objects
        for column in df.columns:
            try:
                col_data = df[column]
                missing_count = col_data.isnull().sum()
                
                # Determine data type
                if pd.api.types.is_numeric_dtype(col_data):
                    data_type = 'numeric'
                    
                    # Check if column has any non-null values
                    if col_data.notna().sum() == 0:
                        # All values are null/empty
                        DatasetVariable.objects.create(
                            dataset=dataset,
                            name=column,
                            data_type=data_type,
                            count=len(col_data),
                            missing_count=len(col_data),
                            mean=None,
                            std_dev=None,
                            min_value=None,
                            q25=None,
                            median=None,
                            q75=None,
                            max_value=None,
                            outliers_count=0
                        )
                        continue
                    
                    # Calculate numeric statistics only for non-null values
                    non_null_data = col_data.dropna()
                    if len(non_null_data) == 0:
                        # All values are null after dropping
                        DatasetVariable.objects.create(
                            dataset=dataset,
                            name=column,
                            data_type=data_type,
                            count=len(col_data),
                            missing_count=len(col_data),
                            mean=None,
                            std_dev=None,
                            min_value=None,
                            q25=None,
                            median=None,
                            q75=None,
                            max_value=None,
                            outliers_count=0
                        )
                        continue
                    
                    # Calculate numeric statistics
                    mean_val = non_null_data.mean()
                    std_val = non_null_data.std()
                    min_val = non_null_data.min()
                    q25_val = non_null_data.quantile(0.25)
                    median_val = non_null_data.median()
                    q75_val = non_null_data.quantile(0.75)
                    max_val = non_null_data.max()
                    
                    # Calculate outliers (using IQR method) only if we have enough data
                    outliers = 0
                    if len(non_null_data) > 1 and std_val > 0:
                        Q1 = non_null_data.quantile(0.25)
                        Q3 = non_null_data.quantile(0.75)
                        IQR = Q3 - Q1
                        if IQR > 0:
                            outliers = ((non_null_data < (Q1 - 1.5 * IQR)) | (non_null_data > (Q3 + 1.5 * IQR))).sum()
                    
                    DatasetVariable.objects.create(
                        dataset=dataset,
                        name=column,
                        data_type=data_type,
                        count=len(col_data),
                        missing_count=missing_count,
                        mean=mean_val,
                        std_dev=std_val,
                        min_value=min_val,
                        q25=q25_val,
                        median=median_val,
                        q75=q75_val,
                        max_value=max_val,
                        outliers_count=outliers
                    )
                else:
                    data_type = 'categorical'
                    # Calculate categorical statistics
                    non_null_data = col_data.dropna()
                    value_counts = non_null_data.value_counts()
                    unique_count = len(value_counts)
                    most_common_value = value_counts.index[0] if len(value_counts) > 0 else None
                    most_common_count = value_counts.iloc[0] if len(value_counts) > 0 else 0
                    
                    DatasetVariable.objects.create(
                        dataset=dataset,
                        name=column,
                        data_type=data_type,
                        count=len(col_data),
                        missing_count=missing_count,
                        unique_values=unique_count,
                        most_common_value=most_common_value,
                        most_common_count=most_common_count
                    )
            except Exception as e:
                # Log the error but continue processing other columns
                print(f"Error processing column {column}: {str(e)}")
                continue
        
        # Update outliers count for the dataset
        total_outliers = sum([
            var.outliers_count for var in dataset.variables.filter(data_type='numeric')
        ])
        dataset.outliers_count = total_outliers
        dataset.save()
        
        # Create or update user preferences to set this as current dataset
        user_pref, created = UserPreference.objects.get_or_create(user=request.user)
        user_pref.current_dataset = dataset
        user_pref.save()
        
        # Create analysis session for this upload
        session = AnalysisSession.objects.create(
            user=request.user,
            dataset=dataset,
            session_name=f"Analysis of {uploaded_file.name}"
        )
        
        # Record the upload interaction
        AnalysisInteraction.objects.create(
            session=session,
            interaction_type='upload',
            description=f"Uploaded dataset: {uploaded_file.name} with {rows} rows and {cols} columns",
            metadata={
                'file_size': uploaded_file.size,
                'file_type': file_extension,
                'rows': rows,
                'columns': cols,
                'numeric_columns': numeric_cols,
                'categorical_columns': categorical_cols
            }
        )
        
        return JsonResponse({
            'success': True,
            'dataset_id': dataset.id,
            'name': dataset.name,
            'rows': rows,
            'columns': cols,
            'file_size': f"{uploaded_file.size / (1024 * 1024):.1f} MB",
            'message': f'Dataset "{uploaded_file.name}" uploaded successfully!',
            'session_id': session.id
        })
        
    except Exception as e:
        return JsonResponse({'error': f'Error processing file: {str(e)}'}, status=500)

@login_required
def get_summary_statistics(request, dataset_id):
    """Get summary statistics for a specific dataset"""
    try:
        dataset = UserDataset.objects.get(id=dataset_id, user=request.user)
        variables = dataset.variables.all()
        
        # Prepare variable summary data using pandas describe approach
        variable_summary = []
        for var in variables:
            if var.data_type == 'numeric':
                # Format numeric values properly, handling null values
                variable_summary.append({
                    'name': var.name,
                    'type': 'Numeric',
                    'count': var.count,
                    'mean': round(var.mean, 2) if var.mean is not None and not pd.isna(var.mean) else 'N/A',
                    'std_dev': round(var.std_dev, 2) if var.std_dev is not None and not pd.isna(var.std_dev) else 'N/A',
                    'min': round(var.min_value, 2) if var.min_value is not None and not pd.isna(var.min_value) else 'N/A',
                    'q25': round(var.q25, 2) if var.q25 is not None and not pd.isna(var.q25) else 'N/A',
                    'median': round(var.median, 2) if var.median is not None and not pd.isna(var.median) else 'N/A',
                    'q75': round(var.q75, 2) if var.q75 is not None and not pd.isna(var.q75) else 'N/A',
                    'max': round(var.max_value, 2) if var.max_value is not None and not pd.isna(var.max_value) else 'N/A'
                })
            else:
                variable_summary.append({
                    'name': var.name,
                    'type': 'Categorical',
                    'count': var.count,
                    'mean': 'N/A',
                    'std_dev': 'N/A',
                    'min': 'N/A',
                    'q25': 'N/A',
                    'median': 'N/A',
                    'q75': 'N/A',
                    'max': 'N/A'
                })
        
        # Calculate additional metrics with better null handling
        numeric_vars = variables.filter(data_type='numeric')
        skewed_vars = sum([1 for var in numeric_vars if var.std_dev and var.std_dev > 0 and not pd.isna(var.std_dev)])
        normal_dist_vars = len(numeric_vars) - skewed_vars
        high_variance_vars = sum([1 for var in numeric_vars if var.std_dev and var.std_dev > var.mean * 0.5 and not pd.isna(var.std_dev) and not pd.isna(var.mean)])
        zero_variance_vars = sum([1 for var in numeric_vars if var.std_dev == 0 or pd.isna(var.std_dev)])
        
        summary_data = {
            'dataset_overview': {
                'total_rows': dataset.rows,
                'total_columns': dataset.columns,
                'numeric_columns': dataset.numeric_columns
            },
            'data_quality': {
                'missing_values': f"{dataset.missing_values_percentage:.1f}%",
                'duplicate_rows': dataset.duplicate_rows,
                'outliers_count': dataset.outliers_count,
                'data_completeness': f"{100 - dataset.missing_values_percentage:.1f}%"
            },
            'distribution_insights': {
                'skewed_variables': skewed_vars,
                'normal_distribution': normal_dist_vars,
                'high_variance': high_variance_vars,
                'zero_variance': zero_variance_vars
            },
            'variable_summary': variable_summary
        }
        
        # Record this interaction if there's an active session
        try:
            active_session = AnalysisSession.objects.filter(
                user=request.user, 
                dataset=dataset, 
                is_active=True
            ).latest('updated_at')
            
            AnalysisInteraction.objects.create(
                session=active_session,
                interaction_type='summary_stats',
                description=f"Viewed summary statistics for dataset: {dataset.name}",
                metadata={
                    'total_rows': dataset.rows,
                    'total_columns': dataset.columns,
                    'numeric_columns': dataset.numeric_columns,
                    'missing_percentage': dataset.missing_values_percentage,
                    'outliers_count': dataset.outliers_count
                }
            )
        except AnalysisSession.DoesNotExist:
            pass  # No active session, that's okay
        
        print(f"Debug: Sending summary data for dataset {dataset_id}")
        print(f"Debug: Variable summary has {len(variable_summary)} variables")
        for var in variable_summary[:3]:  # Print first 3 variables for debugging
            print(f"Debug: Variable {var['name']} - {var['type']} - Mean: {var['mean']}")
        
        return JsonResponse(summary_data)
        
    except UserDataset.DoesNotExist:
        return JsonResponse({'error': 'Dataset not found'}, status=404)
    except Exception as e:
        print(f"Error in get_summary_statistics: {str(e)}")
        return JsonResponse({'error': f'Error generating summary: {str(e)}'}, status=500)


@login_required
def get_dataset_analysis(request, dataset_id):
    """Return full analysis including correlation matrix for the given dataset."""
    try:
        # Ensure the dataset belongs to the user
        _ = UserDataset.objects.get(id=dataset_id, user=request.user)
        stats = get_summary_statistics_data(dataset_id)
        return JsonResponse(stats)
    except UserDataset.DoesNotExist:
        return JsonResponse({'error': 'Dataset not found'}, status=404)
    except Exception as e:
        return JsonResponse({'error': f'Error generating analysis: {str(e)}'}, status=500)


@login_required
def correlation_heatmap(request, dataset_id):
    """Return a seaborn heatmap PNG for the dataset's correlation matrix."""
    try:
        dataset = UserDataset.objects.get(id=dataset_id, user=request.user)
        # Build DataFrame
        import pandas as pd
        import numpy as np
        from .analytics_service import read_dataset_file

        # Try cache first
        cache = CacheManager()
        cache_key = cache.generate_cache_key('img:correlation_heatmap', {'dataset_id': dataset_id})
        cached = cache.get_cached_response(cache_key)
        if cached:
            return HttpResponse(cached, content_type='image/png')

        df = read_dataset_file(dataset)
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) < 2:
            # Render a simple placeholder figure
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.text(0.5, 0.5, 'Not enough numeric columns for correlation',
                    ha='center', va='center')
            ax.axis('off')
        else:
            corr = df[numeric_cols].corr()
            # Plot heatmap
            fig, ax = plt.subplots(
                figsize=(max(6, len(numeric_cols) * 0.5), max(4, len(numeric_cols) * 0.4)),
                constrained_layout=True,
            )
            sns.heatmap(
                corr,
                vmin=-1, vmax=1, center=0,
                cmap='RdBu_r', annot=False, square=False,
                cbar_kws={'shrink': 0.75},
                ax=ax
            )
            ax.set_title(f'Correlation Heatmap: {dataset.name}', fontsize=10)

        from io import BytesIO
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
        buf.seek(0)
        png_bytes = buf.getvalue()
        cache.set_cached_response(cache_key, png_bytes)
        return HttpResponse(png_bytes, content_type='image/png')
    except UserDataset.DoesNotExist:
        return JsonResponse({'error': 'Dataset not found'}, status=404)
    except Exception as e:
        return JsonResponse({'error': f'Error generating heatmap: {str(e)}'}, status=500)


@login_required
def distributions_image(request, dataset_id):
    """Return a PNG grid of charts (2 columns):
    - Numeric columns: Histogram + KDE
    - Categorical columns: Countplot
    The grid is dynamically sized by rows; frontend can scroll if large.
    """
    try:
        dataset = UserDataset.objects.get(id=dataset_id, user=request.user)
        from .analytics_service import read_dataset_file
        import numpy as np
        import pandas as pd

        top = int(request.GET.get('top', 12))
        bins = int(request.GET.get('bins', 20))

        cache = CacheManager()
        query = {
            'dataset_id': dataset_id,
            'top': top,
            'bins': bins,
        }
        cache_key = cache.generate_cache_key('img:distributions', query)
        cached = cache.get_cached_response(cache_key)
        if cached:
            return HttpResponse(cached, content_type='image/png')

        df = read_dataset_file(dataset)
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        if len(numeric_cols) == 0 and len(categorical_cols) == 0:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.text(0.5, 0.5, 'No numeric variables to plot', ha='center', va='center')
            ax.axis('off')
        else:
            # Rank numeric by variance (desc)
            num_ranked = []
            for c in numeric_cols:
                series = pd.to_numeric(df[c], errors='coerce').dropna()
                var = float(series.var()) if len(series) > 0 else -1
                num_ranked.append((c, var))
            num_ranked.sort(key=lambda x: x[1], reverse=True)

            # Rank categorical by cardinality (desc)
            cat_ranked = []
            for c in categorical_cols:
                vc = df[c].astype('category')
                cat_ranked.append((c, int(vc.nunique())))
            cat_ranked.sort(key=lambda x: x[1], reverse=True)

            # Combine, numeric first then categorical, limit by top
            ordered_cols = [c for c, _ in num_ranked] + [c for c, _ in cat_ranked]
            ordered_cols = ordered_cols[:top]

            n = len(ordered_cols)
            cols = 2
            rows = max(1, int(np.ceil(n / cols)))
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 3.5), constrained_layout=True)
            if rows == 1 and cols == 1:
                axes = np.array([[axes]])
            elif rows == 1:
                axes = np.array([axes])
            elif cols == 1:
                axes = np.array([[ax] for ax in axes])

            idx = 0
            for r in range(rows):
                for c in range(cols):
                    ax = axes[r, c]
                    if idx < n:
                        colname = ordered_cols[idx]
                        if colname in numeric_cols:
                            series = pd.to_numeric(df[colname], errors='coerce').dropna()
                            if len(series) > 0:
                                sns.histplot(series, bins=bins, kde=True, ax=ax, color='#1a365d')
                            else:
                                ax.text(0.5, 0.5, f'{colname}\n(no data)', ha='center', va='center', fontsize=8)
                            ax.set_title(f'{colname} (Histogram + KDE)', fontsize=9)
                        else:
                            sns.countplot(x=df[colname], ax=ax, color='#1a365d')
                            ax.set_title(f'{colname} (Count)', fontsize=9)
                            for label in ax.get_xticklabels():
                                label.set_rotation(45)
                                label.set_ha('right')
                        idx += 1
                    else:
                        ax.axis('off')
            # spacing handled by constrained_layout

        from io import BytesIO
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
        buf.seek(0)
        png_bytes = buf.getvalue()
        cache.set_cached_response(cache_key, png_bytes)
        return HttpResponse(png_bytes, content_type='image/png')
    except UserDataset.DoesNotExist:
        return JsonResponse({'error': 'Dataset not found'}, status=404)
    except Exception as e:
        return JsonResponse({'error': f'Error generating distributions: {str(e)}'}, status=500)


@login_required
def outliers_image(request, dataset_id):
    """Return a PNG grid (2 columns) of boxplots for variables with highest outlier counts."""
    try:
        dataset = UserDataset.objects.get(id=dataset_id, user=request.user)
        from .analytics_service import read_dataset_file
        import numpy as np
        import pandas as pd

        top = int(request.GET.get('top', 12))
        k = float(request.GET.get('iqr', 1.5))

        cache = CacheManager()
        query = {
            'dataset_id': dataset_id,
            'top': top,
            'iqr': k,
        }
        cache_key = cache.generate_cache_key('img:outliers', query)
        cached = cache.get_cached_response(cache_key)
        if cached:
            return HttpResponse(cached, content_type='image/png')

        df = read_dataset_file(dataset)
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if len(numeric_cols) == 0:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.text(0.5, 0.5, 'No numeric variables to analyze', ha='center', va='center')
            ax.axis('off')
        else:
            # Compute outlier counts via IQR
            ranking = []
            for c in numeric_cols:
                s = pd.to_numeric(df[c], errors='coerce').dropna()
                if len(s) == 0:
                    continue
                q1 = float(s.quantile(0.25))
                q3 = float(s.quantile(0.75))
                iqr = q3 - q1
                if iqr <= 0:
                    lower = q1
                    upper = q3
                    mask = pd.Series([False] * len(s), index=s.index)
                else:
                    lower = q1 - k * iqr
                    upper = q3 + k * iqr
                    mask = (s < lower) | (s > upper)
                ranking.append((c, int(mask.sum()), lower, upper))

            if not ranking:
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.text(0.5, 0.5, 'No data for outliers', ha='center', va='center')
                ax.axis('off')
            else:
                ranking.sort(key=lambda x: x[1], reverse=True)
                cols = 2
                names = [c for c, _, _, _ in ranking[:top]]
                n = len(names)
                rows = max(1, int(np.ceil(n / cols)))
                fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 3.5), constrained_layout=True)
                if rows == 1 and cols == 1:
                    axes = np.array([[axes]])
                elif rows == 1:
                    axes = np.array([axes])
                elif cols == 1:
                    axes = np.array([[ax] for ax in axes])

                idx = 0
                for r in range(rows):
                    for c in range(cols):
                        ax = axes[r, c]
                        if idx < n:
                            colname = names[idx]
                            s = pd.to_numeric(df[colname], errors='coerce').dropna()
                            sns.boxplot(x=s, ax=ax, color='#1a365d')
                            # Draw IQR bounds lines
                            q1 = float(s.quantile(0.25))
                            q3 = float(s.quantile(0.75))
                            iqr = q3 - q1
                            if iqr > 0:
                                lower = q1 - k * iqr
                                upper = q3 + k * iqr
                                ax.axvline(lower, color='#ef4444', linestyle='--', linewidth=1)
                                ax.axvline(upper, color='#ef4444', linestyle='--', linewidth=1)
                            ax.set_title(f'{colname} (Boxplot)', fontsize=9)
                            idx += 1
                        else:
                            ax.axis('off')
                # spacing handled by constrained_layout

        from io import BytesIO
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
        buf.seek(0)
        png_bytes = buf.getvalue()
        cache.set_cached_response(cache_key, png_bytes)
        return HttpResponse(png_bytes, content_type='image/png')
    except UserDataset.DoesNotExist:
        return JsonResponse({'error': 'Dataset not found'}, status=404)
    except Exception as e:
        return JsonResponse({'error': f'Error generating outliers: {str(e)}'}, status=500)

@login_required
def get_user_datasets(request):
    """Get all datasets for the current user"""
    datasets = UserDataset.objects.filter(user=request.user, is_active=True)
    dataset_list = []
    
    for dataset in datasets:
        dataset_list.append({
            'id': dataset.id,
            'name': dataset.name,
            'rows': dataset.rows,
            'columns': dataset.columns,
            'file_size': f"{dataset.file_size / (1024 * 1024):.1f} MB",
            'uploaded_at': dataset.uploaded_at.strftime('%b %d, %Y')
        })
    
    return JsonResponse({'datasets': dataset_list})

@login_required
def get_user_state(request):
    """Get user's current state including datasets and preferences"""
    try:
        # Get user preferences
        user_pref, created = UserPreference.objects.get_or_create(user=request.user)
        
        # Get all datasets
        datasets = UserDataset.objects.filter(user=request.user, is_active=True)
        dataset_list = []
        
        for dataset in datasets:
            dataset_list.append({
                'id': dataset.id,
                'name': dataset.name,
                'rows': dataset.rows,
                'columns': dataset.columns,
                'file_size': f"{dataset.file_size / (1024 * 1024):.1f} MB",
                'uploaded_at': dataset.uploaded_at.strftime('%b %d, %Y')
            })
        
        # Get current dataset index
        current_dataset_index = -1
        if user_pref.current_dataset:
            for i, dataset in enumerate(dataset_list):
                if dataset['id'] == user_pref.current_dataset.id:
                    current_dataset_index = i
                    break
        
        # Get recent analysis sessions
        recent_sessions = AnalysisSession.objects.filter(
            user=request.user, 
            is_active=True
        )[:5]  # Last 5 sessions
        
        session_list = []
        for session in recent_sessions:
            session_list.append({
                'id': session.id,
                'name': session.session_name,
                'dataset_name': session.dataset.name,
                'created_at': session.created_at.strftime('%b %d, %Y'),
                'updated_at': session.updated_at.strftime('%b %d, %Y')
            })
        
        # Get dataset-specific UI state if current dataset exists
        current_session_id, ui_state = get_current_session_id_and_ui_state(request.user)
        
        return JsonResponse({
            'datasets': dataset_list,
            'current_dataset_index': current_dataset_index,
            'current_session_id': current_session_id,
            'ui_state': ui_state,
            'default_analysis_type': user_pref.default_analysis_type,
            'recent_sessions': session_list
        })
        
    except Exception as e:
        return JsonResponse({'error': f'Error loading user state: {str(e)}'}, status=500)

@login_required
def set_current_dataset(request):
    """Set the current dataset for the user"""
    try:
        dataset_id = request.POST.get('dataset_id')
        if not dataset_id:
            return JsonResponse({'error': 'Dataset ID is required'}, status=400)
        
        dataset, session = set_current_dataset_and_session(request.user, dataset_id)
        
        return JsonResponse({
            'success': True,
            'message': f'Current dataset set to: {dataset.name}',
            'session_id': session.id
        })
        
    except UserDataset.DoesNotExist:
        return JsonResponse({'error': 'Dataset not found'}, status=404)
    except Exception as e:
        return JsonResponse({'error': f'Error setting current dataset: {str(e)}'}, status=500)

@login_required
def record_interaction(request):
    """Record an analysis interaction for LLM context"""
    try:
        interaction_type = request.POST.get('type')
        description = request.POST.get('description')
        metadata = json.loads(request.POST.get('metadata', '{}'))
        
        # Get current session
        user_pref, created = UserPreference.objects.get_or_create(user=request.user)
        if not user_pref.current_dataset:
            return JsonResponse({'error': 'No current dataset'}, status=400)
        
        session = AnalysisSession.objects.filter(
            user=request.user,
            dataset=user_pref.current_dataset,
            is_active=True
        ).first()
        
        if not session:
            session = AnalysisSession.objects.create(
                user=request.user,
                dataset=user_pref.current_dataset,
                session_name=f"Analysis of {user_pref.current_dataset.name}"
            )
        
        # Record the interaction
        AnalysisInteraction.objects.create(
            session=session,
            interaction_type=interaction_type,
            description=description,
            metadata=metadata
        )
        
        # Record analysis history if it's an analysis type
        if interaction_type in ['summary_stats', 'linear_regression', 't_test', 'anova', 'correlation']:
            AnalysisHistory.objects.get_or_create(
                user=request.user,
                dataset=user_pref.current_dataset,
                analysis_type=interaction_type,
                defaults={'is_complete': True}
            )
        
        return JsonResponse({'success': True, 'message': 'Interaction recorded'})
        
    except Exception as e:
        return JsonResponse({'error': f'Error recording interaction: {str(e)}'}, status=500)

@login_required
def get_analysis_history(request, dataset_id):
    """Get analysis history for a specific dataset"""
    try:
        dataset = UserDataset.objects.get(id=dataset_id, user=request.user)
        analysis_history = AnalysisHistory.objects.filter(
            user=request.user,
            dataset=dataset
        ).values('analysis_type', 'created_at', 'is_complete')
        
        return JsonResponse({
            'success': True,
            'analysis_history': list(analysis_history)
        })
    except UserDataset.DoesNotExist:
        return JsonResponse({'error': 'Dataset not found'}, status=404)
    except Exception as e:
        return JsonResponse({'error': f'Error getting analysis history: {str(e)}'}, status=500)

@login_required
@csrf_exempt
@require_http_methods(["POST"])
def delete_dataset(request):
    """Delete a dataset and all associated data"""
    try:
        dataset_id = request.POST.get('dataset_id')
        if not dataset_id:
            return JsonResponse({'error': 'Dataset ID is required'}, status=400)
        
        dataset = UserDataset.objects.get(id=dataset_id, user=request.user)
        dataset_name = dataset.name
        
        # Clean up all related data before deleting the dataset
        # This will automatically delete:
        # - AnalysisSession (CASCADE)
        # - ChatMessage (CASCADE via session)
        # - AnalysisHistory (CASCADE)
        # - ReportDocument (CASCADE)
        # - ReportSection (CASCADE via document)
        # - DatasetUIState (CASCADE)
        # - DatasetVariable (CASCADE)
        
        # Update user preferences if this was the current dataset
        user_pref, created = UserPreference.objects.get_or_create(user=request.user)
        if user_pref.current_dataset and user_pref.current_dataset.id == int(dataset_id):
            user_pref.current_dataset = None
            user_pref.save()
        
        # Delete the dataset (cascade will handle related objects)
        dataset.delete()
        
        return JsonResponse({
            'success': True,
            'message': f'Dataset "{dataset_name}" deleted successfully'
        })
        
    except UserDataset.DoesNotExist:
        return JsonResponse({'error': 'Dataset not found'}, status=404)
    except Exception as e:
        return JsonResponse({'error': f'Error deleting dataset: {str(e)}'}, status=500)

@login_required
@csrf_exempt
@require_http_methods(["POST"])
def update_warning_preferences(request):
    """Update user warning preferences"""
    try:
        warning_type = request.POST.get('warning_type')
        show_warning = request.POST.get('show_warning', 'true').lower() == 'true'
        
        user_pref, created = UserWarningPreference.objects.get_or_create(user=request.user)
        
        if warning_type == 'delete':
            user_pref.show_delete_warning = show_warning
        elif warning_type == 'multiselect':
            user_pref.show_multiselect_help = show_warning
        
        user_pref.save()
        
        return JsonResponse({
            'success': True,
            'message': 'Warning preferences updated'
        })
        
    except Exception as e:
        return JsonResponse({'error': f'Error updating preferences: {str(e)}'}, status=500)

@login_required
def get_warning_preferences(request):
    """Get user warning preferences"""
    try:
        user_pref, created = UserWarningPreference.objects.get_or_create(user=request.user)
        
        return JsonResponse({
            'success': True,
            'show_delete_warning': user_pref.show_delete_warning,
            'show_multiselect_help': user_pref.show_multiselect_help
        })
        
    except Exception as e:
        return JsonResponse({'error': f'Error getting preferences: {str(e)}'}, status=500)

@login_required
@csrf_exempt
@require_http_methods(["POST"])
def save_ui_state(request):
    """Save UI state to database"""
    try:
        ui_state = json.loads(request.POST.get('ui_state', '{}'))
        dataset_id = request.POST.get('dataset_id')
        
        if not dataset_id:
            return JsonResponse({'error': 'dataset_id is required'}, status=400)
        
        save_dataset_ui_state(request.user, int(dataset_id), ui_state)
        
        return JsonResponse({
            'success': True,
            'message': 'UI state saved successfully'
        })
        
    except UserDataset.DoesNotExist:
        return JsonResponse({'error': 'Dataset not found'}, status=404)
    except Exception as e:
        return JsonResponse({'error': f'Error saving UI state: {str(e)}'}, status=500)

@login_required
@csrf_exempt
@require_http_methods(["POST"])
def send_chat_message(request):
    """Handle AI chat messages and store conversation history"""
    print("=== BACKEND AI CHAT DEBUG START ===")
    try:
        data = json.loads(request.body)
        message = data.get('message', '').strip()
        dataset_id = data.get('dataset_id')
        session_id = data.get('session_id')
        
        print(f"Received request - Message: '{message[:50]}...' | Dataset ID: {dataset_id} | Session ID: {session_id}")
        
        if not message or not dataset_id:
            print("❌ Validation failed - missing message or dataset_id")
            return JsonResponse({'error': 'Message and dataset_id are required'}, status=400)
        
        print(f"✅ Validation passed - processing dataset {dataset_id}")
        dataset = UserDataset.objects.get(id=dataset_id, user=request.user)
        print(f"✅ Dataset found: {dataset.name} ({dataset.rows} rows, {dataset.columns} columns)")
        
        # Get or create session
        session = None
        if session_id:
            try:
                session = AnalysisSession.objects.get(id=int(session_id), user=request.user, dataset=dataset)
                print(f"✅ Found existing session: {session.id}")
            except (ValueError, AnalysisSession.DoesNotExist) as e:
                print(f"⚠️ Session not found with ID {session_id}: {e}")
                pass
        
        if not session:
            session = AnalysisSession.objects.filter(user=request.user, dataset=dataset, is_active=True).first()
            if session:
                print(f"✅ Found active session: {session.id}")
            else:
                session = AnalysisSession.objects.create(
                    user=request.user,
                    dataset=dataset,
                    session_name=f"Chat Session for {dataset.name}"
                )
                print(f"✅ Created new session: {session.id}")
        
        # Store user message
        user_message = ChatMessage.objects.create(
            session=session,
            message_type='user',
            content=message
        )
        print(f"✅ Stored user message: {user_message.id}")
        
        # Get summary statistics for context
        print("📊 Getting summary statistics (EDA components)...")
        summary_data = get_summary_statistics_data(dataset_id)
        var_sum = summary_data.get('variable_summary', {}) or {}
        dist_ins = summary_data.get('distribution_insights', {}) or {}
        corr_info = summary_data.get('correlation_matrix', {}) or {}
        corr_matrix = corr_info.get('matrix', {}) or {}
        corr_strong = corr_info.get('strong_correlations', []) or []
        out_info = summary_data.get('outlier_analysis', {}) or {}
        print(f"✅ Summary data retrieved - variable_summary: {len(var_sum)} variables")
        print(f"✅ Distribution insights passed to AI - keys: {len(dist_ins)}")
        print(f"✅ Correlation matrix passed to AI - variables: {len(corr_matrix)}, strong pairs: {len(corr_strong)}")
        print(f"✅ Outlier analysis passed to AI - variables analyzed: {len(out_info)}")
        
        # Prepare context for AI
        context = {
            'dataset_name': dataset.name,
            'dataset_rows': dataset.rows,
            'dataset_columns': dataset.columns,
            'summary_statistics': summary_data,
            'user_message': message,
            'dataset_info': {
                'rows': dataset.rows,
                'columns': dataset.columns,
                'name': dataset.name
            }
        }
        
        # Get previous chat messages for context
        previous_messages = ChatMessage.objects.filter(session=session).order_by('created_at')[:10]
        chat_history = []
        for msg in previous_messages:
            if msg.message_type == 'user':
                chat_history.append(f"User: {msg.content}")
            else:
                chat_history.append(f"AI: {msg.content}")
        
        context['chat_history'] = '\n'.join(chat_history[-6:])  # Last 6 messages for context
        print(f"✅ Chat history prepared - {len(chat_history)} previous messages")
        print("🧠 Passing EDA context to AI: [Summary, Distributions, Correlation, Outliers]")
        
        # Use the actual AI client
        print("🤖 Initializing AI client...")
        try:
            from .ai.llm_client import LLMClient
            llm_client = LLMClient()
            print("✅ AI client initialized successfully")
            
            # Build the full message with context
            full_message = f"""
User Query: {message}

Dataset Information:
- Name: {dataset.name}
- Rows: {dataset.rows:,}
- Columns: {dataset.columns}

Previous Chat History:
{context['chat_history']}

Please provide a comprehensive analysis based on the user's query and the dataset information provided.
"""
            print(f"📝 Sending message to AI (length: {len(full_message)} chars)")
            
            ai_response = llm_client.chat(full_message, context)
            print(f"✅ AI response received (length: {len(ai_response)} chars)")
            print(f"AI response preview: {ai_response[:200]}...")
            
        except Exception as ai_error:
            print(f"❌ AI Error: {ai_error}")
            print(f"AI Error type: {type(ai_error).__name__}")
            print(f"AI Error details: {str(ai_error)}")
            
            # Fallback to basic response if AI fails
            ai_response = f"I'm analyzing your dataset '{dataset.name}' with {dataset.rows:,} rows and {dataset.columns} columns. Here's what I found:\n\n"
            
            # Add some intelligent response based on the message
            if 'summary' in message.lower() or 'statistics' in message.lower():
                ai_response += "## Summary Statistics Overview\n\n"
                ai_response += f"- **Total Variables**: {len(summary_data.get('variable_summary', {}))}\n"
                ai_response += f"- **Numeric Variables**: {summary_data.get('dataset_overview', {}).get('numeric_columns', 0)}\n"
                ai_response += f"- **Categorical Variables**: {summary_data.get('dataset_overview', {}).get('categorical_columns', 0)}\n\n"
                
                # Add some variable insights
                var_summary = summary_data.get('variable_summary', {})
                if var_summary:
                    ai_response += "## Key Variable Insights\n\n"
                    for i, (var_name, var_data) in enumerate(list(var_summary.items())[:5]):
                        if var_data.get('type') == 'numeric':
                            ai_response += f"- **{var_name}**: Mean = {var_data.get('mean', 'N/A'):.2f}, Std = {var_data.get('std', 'N/A'):.2f}\n"
                        else:
                            ai_response += f"- **{var_name}**: {var_data.get('unique_count', 'N/A')} unique values\n"
            
            elif 'quality' in message.lower() or 'missing' in message.lower():
                ai_response += "## Data Quality Analysis\n\n"
                dq = summary_data.get('data_quality', {})
                if dq:
                    for var_name, var_stats in list(dq.items())[:5]:
                        missing_pct = var_stats.get('missing_percentage', 0)
                        ai_response += f"- **{var_name}**: {missing_pct:.1%} missing values\n"
            
            elif 'correlation' in message.lower():
                ai_response += "## Correlation Analysis\n\n"
                corr = summary_data.get('correlation_matrix', {})
                if corr and corr.get('strong_correlations'):
                    ai_response += "Strong correlations found:\n"
                    for corr_item in corr['strong_correlations'][:3]:
                        ai_response += f"- **{corr_item['variable1']}** ↔ **{corr_item['variable2']}**: {corr_item['correlation']:.3f}\n"
                else:
                    ai_response += "No strong correlations detected in the dataset.\n"
            
            else:
                ai_response += "I can help you analyze your dataset! Here are some things you can ask me about:\n\n"
                ai_response += "- **Summary statistics** and key insights\n"
                ai_response += "- **Data quality** and missing value analysis\n"
                ai_response += "- **Correlation analysis** between variables\n"
                ai_response += "- **Distribution patterns** and outliers\n"
                ai_response += "- **Recommendations** for further analysis\n\n"
                ai_response += "What specific aspect would you like me to focus on?"
            
            print(f"✅ Fallback response generated (length: {len(ai_response)} chars)")
        
        # Store AI response
        ai_message = ChatMessage.objects.create(
            session=session,
            message_type='ai',
            content=ai_response
        )
        print(f"✅ Stored AI message: {ai_message.id}")
        
        response_data = {
            'success': True,
            'response': ai_response,
            'session_id': session.id,
            'message_id': ai_message.id
        }
        print(f"✅ Sending response - Session ID: {session.id}, Message ID: {ai_message.id}")
        print("=== BACKEND AI CHAT DEBUG SUCCESS ===")
        
        return JsonResponse(response_data)
        
    except UserDataset.DoesNotExist:
        print("❌ Dataset not found")
        print("=== BACKEND AI CHAT DEBUG ERROR ===")
        return JsonResponse({'error': 'Dataset not found'}, status=404)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        print(f"Error type: {type(e).__name__}")
        print(f"Error details: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        print("=== BACKEND AI CHAT DEBUG ERROR ===")
        return JsonResponse({'error': f'Error processing message: {str(e)}'}, status=500)


@login_required
def get_chat_history(request):
    """Retrieve chat history for a specific session"""
    try:
        dataset_id = request.GET.get('dataset_id')
        session_id = request.GET.get('session_id')
        
        if not dataset_id:
            return JsonResponse({'error': 'Dataset ID is required'}, status=400)
        
        session = resolve_session_for(request.user, int(dataset_id), int(session_id) if session_id else None)
        
        if not session:
            return JsonResponse({'messages': [], 'session_id': None})
        
        # Get chat messages for this session
        messages = ChatMessage.objects.filter(session=session).order_by('created_at')
        
        chat_history = []
        for message in messages:
            chat_history.append({
                'id': message.id,
                'type': message.message_type,
                'content': message.content,
                'created_at': message.created_at.strftime('%Y-%m-%d %H:%M:%S'),
                'is_added_to_report': message.is_added_to_report,
                'report_section_id': message.report_section.id if message.report_section else None
            })
        
        return JsonResponse({
            'messages': chat_history,
            'session_id': session.id,
            'session_name': session.session_name
        })
        
    except UserDataset.DoesNotExist:
        return JsonResponse({'error': 'Dataset not found'}, status=404)
    except Exception as e:
        return JsonResponse({'error': f'Error retrieving chat history: {str(e)}'}, status=500)

def create_comprehensive_summary_table(summary_stats):
    """Deprecated local helper; use analytics_service.create_comprehensive_summary_table instead."""
    from .analytics_service import create_comprehensive_summary_table as _create
    try:
        return _create(summary_stats)
    except Exception:
        return ""


@login_required
@csrf_exempt
@require_http_methods(["POST"])
def add_to_report(request):
    """Add content to the user's report document"""
    print("=== ADD TO REPORT DEBUG START ===")
    try:
        data = json.loads(request.body)
        dataset_id = data.get('dataset_id')
        session_id = data.get('session_id')
        title = data.get('title', 'Analysis Section')
        content = data.get('content', '')
        section_type = data.get('section_type', 'general')
        metadata = data.get('metadata', {})
        message_id = data.get('message_id')
        report_data = data.get('report_data', {})  # New: tables and images data
        
        print(f"Received add_to_report request:")
        print(f"- Dataset ID: {dataset_id}")
        print(f"- Session ID: {session_id}")
        print(f"- Title: {title}")
        print(f"- Section Type: {section_type}")
        print(f"- Message ID: {message_id}")
        print(f"- Report Data: {report_data}")
        
        if not dataset_id:
            print("❌ Missing dataset_id")
            return JsonResponse({'error': 'Dataset ID is required'}, status=400)
        
        dataset = UserDataset.objects.get(id=dataset_id, user=request.user)
        print(f"✅ Dataset found: {dataset.name}")
        
        # Resolve session: accept numeric ID or fallback to active session
        resolved_session = None
        try:
            resolved_session = AnalysisSession.objects.get(id=int(session_id), user=request.user, dataset=dataset)
            print(f"✅ Found session with ID {session_id}")
        except Exception as e:
            print(f"⚠️ Session not found with ID {session_id}, error: {str(e)}")
            # Fallback to active/latest session for this user+dataset
            resolved_session = AnalysisSession.objects.filter(user=request.user, dataset=dataset, is_active=True).order_by('-updated_at').first()
            if not resolved_session:
                resolved_session = AnalysisSession.objects.create(user=request.user, dataset=dataset, session_name=f"Analysis of {dataset.name}")
                print(f"✅ Created new session with ID {resolved_session.id}")
            else:
                print(f"✅ Using fallback session with ID {resolved_session.id}")

        document, _ = ReportDocument.objects.get_or_create(user=request.user, dataset=dataset, session=resolved_session)
        print(f"✅ Using document {document.id}")

        # Determine order
        last_section = document.sections.order_by('-order').first()
        next_order = (last_section.order + 1) if last_section else 1

        # If this is an EDA bundle request, create clustered sections and skip duplicates
        is_eda_bundle = (title or '').strip().lower() in ['eda', 'eda summary', 'eda (exploratory data analysis)']
        if is_eda_bundle:
            try:
                stats = get_summary_statistics_data(dataset.id)
                filename = dataset.name
                created_sections = []

                # Helper to check duplicate by eda_subsection
                def has_section(sub):
                    try:
                        return document.sections.filter(metadata__eda_subsection=sub).exists()
                    except Exception:
                        return False

                # SUMMARY subsection
                if not has_section('summary'):
                    # Auto-generate concise markdown (reuse logic below by setting content)
                    ov = stats.get('dataset_overview', {})
                    parts = [
                        '## Dataset Overview',
                        f"Total Rows: {ov.get('total_rows', 'N/A')} | Total Columns: {ov.get('total_columns', 'N/A')}\n",
                        '## Data Quality Report'
                    ]
                    for var_name, var_stats in (stats.get('data_quality', {}) or {}).items():
                        missing_pct = var_stats.get('missing_percentage', 0)
                        completeness = var_stats.get('completeness', 0)
                        quality_score = var_stats.get('quality_score', 0)
                        parts.append(f"- **{var_name}**: Missing {missing_pct:.1%}, Completeness {completeness:.1%}, Quality Score {quality_score:.1f}")
                    content_summary = '\n'.join(parts)
                    meta_summary = {'eda_subsection': 'summary'}
                    # Add comprehensive summary table to metadata
                    if stats.get('comprehensive_table'):
                        meta_summary['report_data'] = {'tables': [{
                            'type': 'html',
                            'html': stats['comprehensive_table'],
                            'title': 'Complete Dataset Summary Table'
                        }]}
                    ReportSection.objects.create(
                        document=document,
                        order=next_order,
                        title='EDA: Summary Statistics',
                        content=content_summary,
                        section_type='summary_statistics',
                        metadata=meta_summary
                    )
                    created_sections.append('summary')
                    next_order += 1

                # DISTRIBUTIONS subsection
                if not has_section('distributions'):
                    dist_img = None
                    for img in (metadata.get('report_data', {}).get('images', []) if metadata else []):
                        if str(img.get('id', '')).startswith('eda_dist'):
                            dist_img = img.get('src')
                            break
                    if not dist_img:
                        dist_img = f"/api/analysis/{dataset.id}/distributions.png?top=12&bins=20"
                    # Build a compact table of top-variance numeric vars
                    var_html_rows = []
                    var_sum = stats.get('variable_summary', {}) or {}
                    # variable_summary may be dict of {name: meta}
                    for name, meta in var_sum.items():
                        if (meta.get('type') or '').lower() == 'numeric':
                            mean = meta.get('mean')
                            std = meta.get('std') if meta.get('std') is not None else meta.get('std_dev')
                            var_html_rows.append(
                                f"<tr><td style='border:1px solid #ddd;padding:6px;'>{name}</td><td style='border:1px solid #ddd;padding:6px;text-align:right;'>{f'{mean:.2f}' if isinstance(mean,(int,float)) else 'N/A'}</td><td style='border:1px solid #ddd;padding:6px;text-align:right;'>{f'{std:.2f}' if isinstance(std,(int,float)) else 'N/A'}</td></tr>"
                            )
                    dist_table_html = (
                        "<table style=\"border-collapse:collapse;width:100%;margin:8px 0;\"><thead><tr style=\"background:#f8fafc;\"><th style=\"border:1px solid #ddd;padding:6px;text-align:left;\">Variable</th><th style=\"border:1px solid #ddd;padding:6px;text-align:right;\">Mean</th><th style=\"border:1px solid #ddd;padding:6px;text-align:right;\">Std</th></tr></thead><tbody>"
                        + ''.join(var_html_rows[:12]) + "</tbody></table>"
                    )
                    ReportSection.objects.create(
                        document=document,
                        order=next_order,
                        title=f'EDA: Distribution of Data for {filename}',
                        content='',
                        section_type='distributions',
                        metadata={
                            'eda_subsection': 'distributions',
                            'report_data': {
                                'images': [{'type': 'url', 'src': dist_img, 'title': 'Distributions'}],
                                'tables': [{'type': 'html', 'html': dist_table_html, 'title': 'Distribution Summary'}]
                            }
                        }
                    )
                    created_sections.append('distributions')
                    next_order += 1

                # CORRELATION subsection
                if not has_section('correlation'):
                    corr_img = None
                    for img in (metadata.get('report_data', {}).get('images', []) if metadata else []):
                        if str(img.get('id', '')).startswith('eda_corr'):
                            corr_img = img.get('src')
                            break
                    if not corr_img:
                        corr_img = f"/api/analysis/{dataset.id}/heatmap.png"
                    # Build strong correlations table (if any)
                    strong = (stats.get('correlation_matrix', {}) or {}).get('strong_correlations', []) or []
                    corr_rows = []
                    for item in strong:
                        v1 = item.get('variable1'); v2 = item.get('variable2'); r = item.get('correlation')
                        corr_rows.append(
                            f"<tr><td style='border:1px solid #ddd;padding:6px;'>{v1}</td><td style='border:1px solid #ddd;padding:6px;'>{v2}</td><td style='border:1px solid #ddd;padding:6px;text-align:right;'>{f'{r:.3f}' if isinstance(r,(int,float)) else 'N/A'}</td></tr>"
                        )
                    corr_table_html = (
                        "<table style=\"border-collapse:collapse;width:100%;margin:8px 0;\"><thead><tr style=\"background:#f8fafc;\"><th style=\"border:1px solid #ddd;padding:6px;text-align:left;\">Variable 1</th><th style=\"border:1px solid #ddd;padding:6px;text-align:left;\">Variable 2</th><th style=\"border:1px solid #ddd;padding:6px;text-align:right;\">Correlation</th></tr></thead><tbody>"
                        + ''.join(corr_rows[:24]) + "</tbody></table>"
                    )
                    ReportSection.objects.create(
                        document=document,
                        order=next_order,
                        title=f'EDA: Correlation Matrix for {filename}',
                        content='',
                        section_type='correlation',
                        metadata={
                            'eda_subsection': 'correlation',
                            'report_data': {
                                'images': [{'type': 'url', 'src': corr_img, 'title': 'Correlation Heatmap'}],
                                'tables': ([{'type': 'html', 'html': corr_table_html, 'title': 'Strong Correlations'}] if corr_rows else [])
                            }
                        }
                    )
                    created_sections.append('correlation')
                    next_order += 1

                # OUTLIERS subsection
                if not has_section('outliers'):
                    out_img = None
                    for img in (metadata.get('report_data', {}).get('images', []) if metadata else []):
                        if str(img.get('id', '')).startswith('eda_out'):
                            out_img = img.get('src')
                            break
                    if not out_img:
                        out_img = f"/api/analysis/{dataset.id}/outliers.png?top=12&iqr=1.5"
                    out_info = stats.get('outlier_analysis', {}) or {}
                    out_rows = []
                    for name, meta_o in out_info.items():
                        cnt = meta_o.get('count', 0)
                        pct = meta_o.get('percentage', 0.0)
                        lb = meta_o.get('lower_bound'); ub = meta_o.get('upper_bound')
                        out_rows.append(
                            f"<tr><td style='border:1px solid #ddd;padding:6px;'>{name}</td><td style='border:1px solid #ddd;padding:6px;text-align:right;'>{cnt}</td><td style='border:1px solid #ddd;padding:6px;text-align:right;'>{pct:.1%}</td><td style='border:1px solid #ddd;padding:6px;text-align:right;'>{f'{lb:.2f}' if isinstance(lb,(int,float)) else 'N/A'}</td><td style='border:1px solid #ddd;padding:6px;text-align:right;'>{f'{ub:.2f}' if isinstance(ub,(int,float)) else 'N/A'}</td></tr>"
                        )
                    out_table_html = (
                        "<table style=\"border-collapse:collapse;width:100%;margin:8px 0;\"><thead><tr style=\"background:#f8fafc;\"><th style=\"border:1px solid #ddd;padding:6px;text-align:left;\">Variable</th><th style=\"border:1px solid #ddd;padding:6px;text-align:right;\">Outliers</th><th style=\"border:1px solid #ddd;padding:6px;text-align:right;\">%</th><th style=\"border:1px solid #ddd;padding:6px;text-align:right;\">Lower bound</th><th style=\"border:1px solid #ddd;padding:6px;text-align:right;\">Upper bound</th></tr></thead><tbody>"
                        + ''.join(sorted(out_rows, reverse=True)[:24]) + "</tbody></table>"
                    )
                    ReportSection.objects.create(
                        document=document,
                        order=next_order,
                        title=f'EDA: Outlier Analysis for {filename}',
                        content='',
                        section_type='outliers',
                        metadata={
                            'eda_subsection': 'outliers',
                            'report_data': {
                                'images': [{'type': 'url', 'src': out_img, 'title': 'Outliers (Boxplots)'}],
                                'tables': ([{'type': 'html', 'html': out_table_html, 'title': 'Outlier Summary'}] if out_rows else [])
                            }
                        }
                    )
                    created_sections.append('outliers')
                    next_order += 1

                # Mark UI state (eda_added=true)
                try:
                    # Merge flag in UI state
                    from .session_manager import save_dataset_ui_state
                    existing = {'eda_added': True}
                    save_dataset_ui_state(request.user, dataset.id, existing)
                except Exception:
                    pass

                print(f"✅ EDA sections created: {', '.join(created_sections) if created_sections else 'none (skipped duplicates)'}")
                return JsonResponse({'success': True, 'message': 'EDA added to report', 'sections_created': created_sections})
            except Exception as e:
                print(f"❌ Error creating EDA sections: {e}")
                return JsonResponse({'error': 'Error creating EDA sections'}, status=500)

        # If summary_statistics and no content provided, auto-generate a concise markdown block
        if (section_type == 'summary_statistics') and (not content):
            try:
                stats = get_summary_statistics_data(dataset.id)
                ov = stats.get('dataset_overview', {})
                content_parts = []
                content_parts.append(f"## Dataset Overview\n")
                content_parts.append(f"Total Rows: {ov.get('total_rows', 'N/A')} | Total Columns: {ov.get('total_columns', 'N/A')}\n")
                
                dq = stats.get('data_quality', {})
                if dq:
                    content_parts.append(f"## Data Quality Report\n")
                    for var_name, var_stats in dq.items():
                        missing_pct = var_stats.get('missing_percentage', 0)
                        completeness = var_stats.get('completeness', 0)
                        quality_score = var_stats.get('quality_score', 0)
                        content_parts.append(f"- **{var_name}**: Missing {missing_pct:.1%}, Completeness {completeness:.1%}, Quality Score {quality_score:.1f}\n")
                
                di = stats.get('distribution_insights', {})
                if di:
                    content_parts.append(f"## Distribution Insights\n")
                    content_parts.append(f"- Skewed Variables: {di.get('skewed_variables', 0)}\n")
                    content_parts.append(f"- Normal Distribution: {di.get('normal_distribution', 0)}\n")
                    content_parts.append(f"- High Variance: {di.get('high_variance', 0)}\n")
                    content_parts.append(f"- Zero Variance: {di.get('zero_variance', 0)}\n")
                
                # Add comprehensive summary data table (attached to report metadata below)
                content_parts.append("## Complete Dataset Summary Table\n")
                content_parts.append("A full summary table for all variables is attached to this section.")
                
                # Add correlation summary if available
                corr = stats.get('correlation_matrix', {})
                if corr and corr.get('strong_correlations'):
                    content_parts.append("\n## Correlation Summary\n")
                    content_parts.append('| Variable 1 | Variable 2 | Correlation | Strength |')
                    content_parts.append('|------------|------------|-------------|----------|')
                    for corr_item in corr['strong_correlations']:
                        corr_val = corr_item['correlation']
                        strength = 'Strong' if abs(corr_val) >= 0.8 else 'Moderate' if abs(corr_val) >= 0.6 else 'Weak'
                        content_parts.append(f"| {corr_item['variable1']} | {corr_item['variable2']} | {corr_val:.3f} | {strength} |")
                
                corr = stats.get('correlation_matrix', {})
                if corr and corr.get('strong_correlations'):
                    content_parts.append(f"\n## Strong Correlations\n")
                    for corr_item in corr['strong_correlations']:
                        content_parts.append(f"- **{corr_item['variable1']}** ↔ **{corr_item['variable2']}**: {corr_item['correlation']:.3f}\n")
                
                content = ('\n\n'.join(content_parts))
                print("✅ Generated summary statistics content")
            except Exception as e:
                print(f"❌ Error generating summary content: {e}")
                content = 'Summary statistics added.'

        # Store report data (tables and images) in metadata
        if report_data:
            metadata['report_data'] = report_data
            print(f"📊 Stored {len(report_data.get('tables', []))} tables and {len(report_data.get('images', []))} images in metadata")
        
        # Auto-extract HTML tables from AI responses if no report_data provided
        if section_type == 'ai_response' and content and not report_data:
            extracted_tables = extract_html_tables_from_content(content)
            if extracted_tables:
                if 'report_data' not in metadata:
                    metadata['report_data'] = {}
                if 'tables' not in metadata['report_data']:
                    metadata['report_data']['tables'] = []
                
                metadata['report_data']['tables'].extend(extracted_tables)
                print(f"📊 Auto-extracted {len(extracted_tables)} HTML tables from AI response")
        
        # Add comprehensive summary table to metadata if this is a summary_statistics section
        if section_type == 'summary_statistics':
            try:
                stats = get_summary_statistics_data(dataset.id)
                if stats.get('comprehensive_table'):
                    if 'report_data' not in metadata:
                        metadata['report_data'] = {}
                    if 'tables' not in metadata['report_data']:
                        metadata['report_data']['tables'] = []
                    
                    metadata['report_data']['tables'].append({
                        'type': 'html',
                        'html': stats['comprehensive_table'],
                        'title': 'Complete Dataset Summary Table'
                    })
                    print(f"📊 Added comprehensive summary table to metadata")
            except Exception as e:
                print(f"⚠️ Error adding comprehensive table to metadata: {e}")

        report_section = ReportSection.objects.create(
            document=document,
            order=next_order,
            title=title,
            content=content,
            section_type=section_type,
            metadata=metadata
        )
        print(f"✅ Created report section {report_section.id} with order {next_order}")

        if section_type == 'ai_response' and data.get('message_id'):
            try:
                message_id = data.get('message_id')
                chat_message = ChatMessage.objects.get(id=message_id, session=resolved_session)
                chat_message.is_added_to_report = True
                chat_message.report_section = report_section
                chat_message.save()
                print(f"✅ Linked chat message {message_id} to report section {report_section.id}")
            except ChatMessage.DoesNotExist:
                print(f"⚠️ Chat message {message_id} not found")

        print("=== ADD TO REPORT DEBUG SUCCESS ===")
        return JsonResponse({'success': True, 'message': 'Added to report', 'section_order': next_order})

    except UserDataset.DoesNotExist:
        print("❌ Dataset not found")
        print("=== ADD TO REPORT DEBUG ERROR ===")
        return JsonResponse({'error': 'Dataset not found'}, status=404)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        print("=== ADD TO REPORT DEBUG ERROR ===")
        return JsonResponse({'error': f'Error adding to report: {str(e)}'}, status=500)


from .report_generator import generate_report_document, extract_html_tables_from_content

@login_required
@require_http_methods(["GET"])
def download_report(request):
    """Compile the stored report sections into a nicely formatted DOCX and return as download"""
    dataset_id = request.GET.get('dataset_id')
    session_id = request.GET.get('session_id')
    if not dataset_id or not session_id:
        return JsonResponse({'error': 'dataset_id and session_id are required'}, status=400)

    return generate_report_document(request, dataset_id, session_id)

@login_required
def get_user_billing_summary(request):
    """Return billing info to show on dashboard (BDT spend only)."""
    try:
        quota = get_user_quota_status(request.user)
        return JsonResponse({
            'plan_name': quota['plan_name'],
            'amount_spent_bdt': quota['amount_spent_bdt'],
        })
    except Exception as e:
        return JsonResponse({'error': f'Error fetching billing summary: {str(e)}'}, status=500)

@login_required
def get_account_overview(request):
    """Return account info, usage and billing for the dashboard UI."""
    try:
        # Basic account info
        user = request.user
        profile = getattr(user, 'profile', None)
        account = {
            'username': user.username,
            'email': user.email,
            'first_name': user.first_name,
            'last_name': user.last_name,
            'phone_number': getattr(profile, 'phone_number', None) if profile else None,
            'country_code': getattr(profile, 'country_code', None) if profile else None,
            'joined': user.date_joined.strftime('%Y-%m-%d'),
        }

        # Billing summary
        billing = build_billing_summary(user)

        # Usage summary (tokens not exposed; only spend)
        usage = {
            'amount_spent_bdt': billing['amount_spent_bdt'],
        }

        return JsonResponse({'account': account, 'billing': billing, 'usage': usage})
    except Exception as e:
        return JsonResponse({'error': f'Error fetching account overview: {str(e)}'}, status=500)