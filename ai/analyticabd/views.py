from django.shortcuts import render, redirect
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from .forms import CustomUserCreationForm, CustomAuthenticationForm
from .models import UserDataset, DatasetVariable, AnalysisSession, AnalysisInteraction, UserPreference, AnalysisHistory, UserWarningPreference
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime

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
        
        return JsonResponse({
            'datasets': dataset_list,
            'current_dataset_index': current_dataset_index,
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
        
        dataset = UserDataset.objects.get(id=dataset_id, user=request.user)
        
        # Update user preferences
        user_pref, created = UserPreference.objects.get_or_create(user=request.user)
        user_pref.current_dataset = dataset
        user_pref.save()
        
        # Create or update analysis session
        session, created = AnalysisSession.objects.get_or_create(
            user=request.user,
            dataset=dataset,
            is_active=True,
            defaults={'session_name': f"Analysis of {dataset.name}"}
        )
        
        if not created:
            session.updated_at = datetime.now()
            session.save()
        
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
        
        # Delete the dataset (cascade will handle related objects)
        dataset.delete()
        
        # Update user preferences if this was the current dataset
        user_pref, created = UserPreference.objects.get_or_create(user=request.user)
        if user_pref.current_dataset and user_pref.current_dataset.id == int(dataset_id):
            user_pref.current_dataset = None
            user_pref.save()
        
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
def send_chat_message(request):
    """Handle AI chat messages"""
    try:
        data = json.loads(request.body)
        message = data.get('message')
        session_id = data.get('session_id')
        context = data.get('context', {})
        
        if not message:
            return JsonResponse({'error': 'Message is required'}, status=400)
        
        # Import AI components
        from .ai.tool_executor import ToolExecutor
        from .ai.conversation_manager import ConversationManager
        
        # Initialize tool executor
        tool_executor = ToolExecutor()
        
        # Process the message with AI
        result = tool_executor.process_chat_message(
            user_id=request.user.id,
            message=message,
            session_id=session_id,
            context=context
        )
        
        if result.get('error'):
            return JsonResponse({'error': result['error']}, status=500)
        
        return JsonResponse({
            'success': True,
            'response': result['response'],
            'tool_executed': result.get('tool_executed'),
            'metadata': result.get('metadata', {})
        })
        
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON data'}, status=400)
    except Exception as e:
        return JsonResponse({'error': f'Error processing chat message: {str(e)}'}, status=500)