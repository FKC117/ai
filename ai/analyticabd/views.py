from django.shortcuts import render, redirect
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from .forms import CustomUserCreationForm, CustomAuthenticationForm
from .models import UserDataset, DatasetVariable, AnalysisSession, AnalysisInteraction, UserPreference, AnalysisHistory, UserWarningPreference, ReportDocument, ReportSection, ChatMessage
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from io import BytesIO
from django.http import HttpResponse
import re
from bs4 import BeautifulSoup

try:
    from docx import Document
    from docx.shared import Inches, Pt
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import RGBColor
    DOCX_AVAILABLE = True
except Exception:
    DOCX_AVAILABLE = False

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
        print("📊 Getting summary statistics...")
        summary_data = get_summary_statistics_data(dataset_id)
        print(f"✅ Summary data retrieved - {len(summary_data.get('variable_summary', {}))} variables")
        
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
        
        dataset = UserDataset.objects.get(id=dataset_id, user=request.user)
        
        # Get session
        session = None
        if session_id:
            try:
                session = AnalysisSession.objects.get(id=int(session_id), user=request.user, dataset=dataset)
            except (ValueError, AnalysisSession.DoesNotExist):
                pass
        
        if not session:
            session = AnalysisSession.objects.filter(user=request.user, dataset=dataset, is_active=True).first()
        
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

def get_summary_statistics_data(dataset_id):
    """Get summary statistics data for a dataset"""
    try:
        dataset = UserDataset.objects.get(id=dataset_id)
        
        # Read the dataset file with better error handling
        import pandas as pd
        import os
        
        # Check if file exists
        if not os.path.exists(dataset.file.path):
            raise Exception(f"Dataset file not found: {dataset.file.path}")
        
        # Determine file type and read accordingly
        file_extension = os.path.splitext(dataset.file.name)[1].lower()
        
        try:
            if file_extension == '.csv':
                df = pd.read_csv(dataset.file.path)
            elif file_extension in ['.xlsx', '.xls']:
                df = pd.read_excel(dataset.file.path)
            elif file_extension == '.json':
                df = pd.read_json(dataset.file.path)
            else:
                # Try CSV as default
                df = pd.read_csv(dataset.file.path)
        except Exception as e:
            raise Exception(f"Error reading file {dataset.file.name}: {str(e)}")
        
        # Calculate summary statistics
        summary_stats = {
            'dataset_overview': {
                'total_rows': len(df),
                'total_columns': len(df.columns),
                'data_types': df.dtypes.to_dict(),
                'sample_data': df.head(5).to_dict('records')
            },
            'variable_summary': {},
            'data_quality': {},
            'distribution_insights': {},
            'correlation_matrix': {},
            'outlier_analysis': {},
            'missing_data_analysis': {},
            'comprehensive_table': {}  # New field for comprehensive table data
        }
        
        # Calculate variable summary with better error handling
        for col in df.columns:
            try:
                col_data = df[col]
                if pd.api.types.is_numeric_dtype(col_data):
                    # Handle numeric data
                    non_null_data = col_data.dropna()
                    if len(non_null_data) == 0:
                        # All values are null
                        summary_stats['variable_summary'][col] = {
                            'type': 'numeric',
                            'count': col_data.count(),
                            'mean': None,
                            'std': None,
                            'min': None,
                            'q25': None,
                            'median': None,
                            'q75': None,
                            'max': None,
                            'skewness': None,
                            'kurtosis': None
                        }
                    else:
                        summary_stats['variable_summary'][col] = {
                            'type': 'numeric',
                            'count': col_data.count(),
                            'mean': float(non_null_data.mean()),
                            'std': float(non_null_data.std()),
                            'min': float(non_null_data.min()),
                            'q25': float(non_null_data.quantile(0.25)),
                            'median': float(non_null_data.median()),
                            'q75': float(non_null_data.quantile(0.75)),
                            'max': float(non_null_data.max()),
                            'skewness': float(non_null_data.skew()),
                            'kurtosis': float(non_null_data.kurtosis())
                        }
                else:
                    # Handle categorical data
                    non_null_data = col_data.dropna()
                    value_counts = non_null_data.value_counts()
                    
                    summary_stats['variable_summary'][col] = {
                        'type': 'categorical',
                        'count': col_data.count(),
                        'unique_count': col_data.nunique(),
                        'most_common': str(value_counts.index[0]) if len(value_counts) > 0 else None,
                        'most_common_count': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0
                    }
            except Exception as e:
                print(f"Error processing column {col}: {str(e)}")
                # Add basic info for failed columns
                summary_stats['variable_summary'][col] = {
                    'type': 'unknown',
                    'count': len(df),
                    'error': str(e)
                }
        
        # Calculate data quality metrics
        for col in df.columns:
            try:
                missing_pct = df[col].isnull().sum() / len(df)
                summary_stats['data_quality'][col] = {
                    'missing_percentage': float(missing_pct),
                    'completeness': float(1 - missing_pct),
                    'quality_score': float(1 - missing_pct if missing_pct < 0.1 else 0.5)
                }
            except Exception as e:
                print(f"Error calculating quality metrics for {col}: {str(e)}")
                summary_stats['data_quality'][col] = {
                    'missing_percentage': 0.0,
                    'completeness': 1.0,
                    'quality_score': 1.0
                }
        
        # Calculate correlation matrix for numeric columns
        try:
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 1:
                correlation_matrix = df[numeric_cols].corr()
                summary_stats['correlation_matrix'] = {
                    'matrix': correlation_matrix.to_dict(),
                    'strong_correlations': []
                }
                
                # Find strong correlations
                for i in range(len(correlation_matrix.columns)):
                    for j in range(i+1, len(correlation_matrix.columns)):
                        corr_value = correlation_matrix.iloc[i, j]
                        if abs(corr_value) >= 0.7:
                            summary_stats['correlation_matrix']['strong_correlations'].append({
                                'variable1': str(correlation_matrix.columns[i]),
                                'variable2': str(correlation_matrix.columns[j]),
                                'correlation': float(corr_value)
                            })
        except Exception as e:
            print(f"Error calculating correlations: {str(e)}")
            summary_stats['correlation_matrix'] = {
                'matrix': {},
                'strong_correlations': []
            }
        
        # Create comprehensive HTML table for summary statistics
        summary_stats['comprehensive_table'] = create_comprehensive_summary_table(summary_stats)
        
        return summary_stats
        
    except Exception as e:
        print(f"Error in get_summary_statistics_data: {str(e)}")
        raise Exception(f"Error calculating summary statistics: {str(e)}")

def create_comprehensive_summary_table(summary_stats):
    """Create a comprehensive HTML table for summary statistics"""
    try:
        var_items = list(summary_stats.get('variable_summary', {}).items())
        if not var_items:
            return ""
        
        # Start building HTML table
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
            # Get data quality info
            dq_info = summary_stats.get('data_quality', {}).get(name, {})
            missing_pct = dq_info.get('missing_percentage', 0)
            quality_score = dq_info.get('quality_score', 1.0)
            
            # Determine row background color
            bg_color = '#f7fafc' if idx % 2 == 1 else '#ffffff'
            
            if meta.get('type') == 'numeric':
                mean_val = meta.get('mean', 'N/A')
                std_val = meta.get('std', 'N/A')
                min_val = meta.get('min', 'N/A')
                q25_val = meta.get('q25', 'N/A')
                median_val = meta.get('median', 'N/A')
                q75_val = meta.get('q75', 'N/A')
                max_val = meta.get('max', 'N/A')
                
                # Format numeric values
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
        
    except Exception as e:
        print(f"Error creating comprehensive summary table: {str(e)}")
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
                
                # Add comprehensive summary data table for all section types
                content_parts.append(f"## Complete Dataset Summary Table\n")
                var_items = list(stats.get('variable_summary', {}).items())
                if var_items:
                    # Create a comprehensive table with all statistics
                    table_rows = ['| Variable | Type | Count | Missing % | Mean | Std Dev | Min | 25th % | Median | 75th % | Max | Skewness | Kurtosis | Quality Score |']
                    table_rows.append('|----------|------|-------|-----------|------|---------|-----|---------|--------|---------|-----|----------|----------|--------------|')
                    
                    for name, meta in var_items:
                        # Get data quality info
                        dq_info = stats.get('data_quality', {}).get(name, {})
                        missing_pct = dq_info.get('missing_percentage', 0)
                        quality_score = dq_info.get('quality_score', 1.0)
                        
                        if meta.get('type') == 'numeric':
                            mean_val = meta.get('mean', 'N/A')
                            std_val = meta.get('std', 'N/A')
                            min_val = meta.get('min', 'N/A')
                            q25_val = meta.get('q25', 'N/A')
                            median_val = meta.get('median', 'N/A')
                            q75_val = meta.get('q75', 'N/A')
                            max_val = meta.get('max', 'N/A')
                            skewness = meta.get('skewness', 'N/A')
                            kurtosis = meta.get('kurtosis', 'N/A')
                            
                            # Format numeric values
                            if isinstance(mean_val, (int, float)): mean_val = f"{mean_val:.2f}"
                            if isinstance(std_val, (int, float)): std_val = f"{std_val:.2f}"
                            if isinstance(min_val, (int, float)): min_val = f"{min_val:.2f}"
                            if isinstance(q25_val, (int, float)): q25_val = f"{q25_val:.2f}"
                            if isinstance(median_val, (int, float)): median_val = f"{median_val:.2f}"
                            if isinstance(q75_val, (int, float)): q75_val = f"{q75_val:.2f}"
                            if isinstance(max_val, (int, float)): max_val = f"{max_val:.2f}"
                            if isinstance(skewness, (int, float)): skewness = f"{skewness:.2f}"
                            if isinstance(kurtosis, (int, float)): kurtosis = f"{kurtosis:.2f}"
                            
                            row = f"| {name} | numeric | {meta.get('count', 'N/A')} | {missing_pct:.1%} | {mean_val} | {std_val} | {min_val} | {q25_val} | {median_val} | {q75_val} | {max_val} | {skewness} | {kurtosis} | {quality_score:.2f} |"
                        else:
                            unique_count = meta.get('unique_count', 'N/A')
                            most_common = meta.get('most_common', 'N/A')
                            most_common_count = meta.get('most_common_count', 'N/A')
                            row = f"| {name} | categorical | {meta.get('count', 'N/A')} | {missing_pct:.1%} | {unique_count} unique | {most_common} ({most_common_count}) |  |  |  |  |  |  |  | {quality_score:.2f} |"
                        table_rows.append(row)
                    
                    content_parts.append('\n'.join(table_rows))
                    
                    # Add correlation summary if available
                    corr = stats.get('correlation_matrix', {})
                    if corr and corr.get('strong_correlations'):
                        content_parts.append(f"\n## Correlation Summary\n")
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


@login_required
@require_http_methods(["GET"])
def download_report(request):
    """Compile the stored report sections into a nicely formatted DOCX and return as download"""
    try:
        if not DOCX_AVAILABLE:
            return JsonResponse({'error': 'python-docx is not installed on the server'}, status=500)

        dataset_id = request.GET.get('dataset_id')
        session_id = request.GET.get('session_id')
        if not dataset_id or not session_id:
            return JsonResponse({'error': 'dataset_id and session_id are required'}, status=400)

        dataset = UserDataset.objects.get(id=dataset_id, user=request.user)
        # Resolve session: accept numeric ID or fallback to active session
        try:
            session = AnalysisSession.objects.get(id=int(session_id), user=request.user, dataset=dataset)
        except Exception:
            session = AnalysisSession.objects.filter(user=request.user, dataset=dataset, is_active=True).order_by('-updated_at').first()
            if not session:
                return JsonResponse({'error': 'No active session found for this dataset'}, status=404)
        document_obj = ReportDocument.objects.filter(user=request.user, dataset=dataset, session=session).first()

        if not document_obj:
            return JsonResponse({'error': 'No report content to download yet'}, status=404)

        # Build DOCX
        doc = Document()

        # Apply custom styles to match DataFlow Analytics design
        from docx.shared import RGBColor, Inches
        from docx.enum.text import WD_ALIGN_PARAGRAPH
        from docx.oxml.shared import OxmlElement, qn

        # Title with DataFlow Analytics branding
        title_paragraph = doc.add_paragraph()
        title_run = title_paragraph.add_run("DataFlow Analytics Report")
        title_run.font.name = 'Inter'
        title_run.font.size = Pt(28)
        title_run.font.bold = True
        title_run.font.color.rgb = RGBColor(26, 54, 93)  # --color-primary
        title_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        
        # Subtitle
        subtitle_paragraph = doc.add_paragraph()
        subtitle_run = subtitle_paragraph.add_run("Advanced Data Analysis & Insights")
        subtitle_run.font.name = 'Inter'
        subtitle_run.font.size = Pt(14)
        subtitle_run.font.color.rgb = RGBColor(113, 128, 150)  # --color-text-secondary
        subtitle_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER

        # Add a decorative line
        doc.add_paragraph('')

        # Dataset Information Card
        info_paragraph = doc.add_paragraph()
        info_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        
        # Dataset name with primary color
        dataset_name_run = info_paragraph.add_run(f"Dataset: {dataset.name}")
        dataset_name_run.font.name = 'Inter'
        dataset_name_run.font.size = Pt(16)
        dataset_name_run.font.bold = True
        dataset_name_run.font.color.rgb = RGBColor(26, 54, 93)  # --color-primary
        
        info_paragraph.add_run("\n")
        
        # Dataset stats with secondary color
        stats_run = info_paragraph.add_run(f"Rows: {dataset.rows:,} | Columns: {dataset.columns} | Size: {dataset.file_size or 'N/A'}")
        stats_run.font.name = 'Inter'
        stats_run.font.size = Pt(12)
        stats_run.font.color.rgb = RGBColor(56, 178, 172)  # --color-secondary
        
        info_paragraph.add_run("\n")
        
        # Upload and generation dates
        date_run = info_paragraph.add_run(f"Uploaded: {dataset.uploaded_at.strftime('%B %d, %Y at %I:%M %p')}")
        date_run.font.name = 'Inter'
        date_run.font.size = Pt(10)
        date_run.font.color.rgb = RGBColor(113, 128, 150)  # --color-text-secondary
        
        date_run = info_paragraph.add_run(f" | Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}")
        date_run.font.name = 'Inter'
        date_run.font.size = Pt(10)
        date_run.font.color.rgb = RGBColor(113, 128, 150)  # --color-text-secondary

        doc.add_paragraph('')

        # Sections - smart ordering by type first, then by order/created_at
        type_priority = {
            'summary_statistics': 10,
            'data_quality': 20,
            'correlation': 30,
            'outlier_analysis': 40,
            'visualization': 50,
            'ai_response': 90,
        }

        sections = list(document_obj.sections.all())
        sections.sort(key=lambda s: (type_priority.get(s.section_type, 80), s.order, s.created_at))

        for section in sections:
            # Section header with DataFlow styling
            heading = doc.add_heading(section.title, level=2)
            heading.alignment = WD_ALIGN_PARAGRAPH.LEFT
            
            # Style the heading
            for run in heading.runs:
                run.font.name = 'Inter'
                run.font.size = Pt(18)
                run.font.bold = True
                run.font.color.rgb = RGBColor(26, 54, 93)  # --color-primary

            content = section.content or ''
            
            # Check if section has report data (tables/images from AI responses)
            report_data = section.metadata.get('report_data', {}) if section.metadata else {}
            tables_data = report_data.get('tables', [])
            images_data = report_data.get('images', [])
            
            print(f"📄 Processing section '{section.title}' with {len(tables_data)} tables and {len(images_data)} images")
            
            # For AI responses, preserve the original order by processing content sequentially
            if section.section_type == 'ai_response':
                # First, extract and render HTML tables from the content
                try:
                    soup = BeautifulSoup(content, 'html.parser')
                    html_tables = soup.find_all('table')
                    
                    if html_tables:
                        print(f"📊 Found {len(html_tables)} HTML tables in AI response")
                        for i, html_table in enumerate(html_tables):
                            table_data = {
                                'type': 'html',
                                'html': str(html_table),
                                'title': f'Table {i+1}'
                            }
                            table_rendered = render_table_from_metadata(table_data)
                            if table_rendered:
                                print(f"✅ Successfully rendered HTML table {i+1}")
                        
                        # Remove HTML tables from content for text processing
                        for table in html_tables:
                            table.decompose()
                        
                        # Get the cleaned text content without HTML tables
                        content = soup.get_text()
                
                except Exception as e:
                    print(f"⚠️ Error processing HTML tables: {e}")
                
                # Process remaining content line-by-line in original order, matching tables from metadata when encountered
                lines = content.splitlines()
                current_paragraph = []
                table_index = 0  # Track which table from metadata we're on
                
                def render_table_from_metadata(table_data):
                    """Render a table from metadata with proper styling"""
                    try:
                        if table_data.get('type') == 'markdown':
                            # Handle markdown table
                            markdown_text = table_data.get('markdown', '')
                            table_lines = markdown_text.split('\n')
                            parsed_rows = []
                            
                            for line in table_lines:
                                if line.strip().startswith('|') and line.strip().endswith('|'):
                                    # Remove leading/trailing | and split by |
                                    cells = [cell.strip() for cell in line.strip('|').split('|')]
                                    parsed_rows.append(cells)
                            
                            if parsed_rows and len(parsed_rows) > 1:  # At least header + data
                                # Skip separator row (dashes)
                                data_rows = [row for row in parsed_rows if not all(cell.replace('-', '').replace(':', '').strip() == '' for cell in row)]
                                
                                if data_rows:
                                    doc_table = doc.add_table(rows=len(data_rows), cols=len(data_rows[0]))
                                    doc_table.style = 'Table Grid'
                                    
                                    for r_idx, row_cells in enumerate(data_rows):
                                        for c_idx, cell_text in enumerate(row_cells):
                                            if c_idx < len(doc_table.rows[r_idx].cells):
                                                doc_cell = doc_table.rows[r_idx].cells[c_idx]
                                                doc_cell.text = cell_text
                                                
                                                # Style the cell
                                                for paragraph in doc_cell.paragraphs:
                                                    for run in paragraph.runs:
                                                        run.font.name = 'Inter'
                                                        run.font.size = Pt(10)
                                                        if r_idx == 0:  # Header row
                                                            run.font.bold = True
                                                            run.font.color.rgb = RGBColor(255, 255, 255)  # White text
                                                            doc_cell._tc.get_or_add_tcPr().append(OxmlElement('w:shd'))
                                                            doc_cell._tc.get_or_add_tcPr().find(qn('w:shd')).set(qn('w:fill'), '1a365d')  # Primary color
                                                        else:
                                                            run.font.color.rgb = RGBColor(45, 55, 72)  # --color-text-primary
                                                            if r_idx % 2 == 1:  # Alternating rows
                                                                doc_cell._tc.get_or_add_tcPr().append(OxmlElement('w:shd'))
                                                                doc_cell._tc.get_or_add_tcPr().find(qn('w:shd')).set(qn('w:fill'), 'f7fafc')  # --color-background
                                    
                                    # Auto-fit column widths
                                    for column in doc_table.columns:
                                        column.width = Inches(1.5)
                                    
                                    doc.add_paragraph('')  # Add space after table
                                    print(f"✅ Added markdown table from metadata successfully")
                                    return True
                                else:
                                    print(f"⚠️ No valid data rows found in markdown table")
                            else:
                                print(f"⚠️ Invalid markdown table format")
                        
                        elif table_data.get('type') == 'html':
                            # Handle HTML table with enhanced styling
                            from bs4 import BeautifulSoup
                            soup = BeautifulSoup(table_data['html'], 'html.parser')
                            table = soup.find('table')
                            
                            if table:
                                # Create a new table in the document
                                rows = table.find_all('tr')
                                if rows:
                                    # Determine max columns across all rows
                                    max_cols = max(len(row.find_all(['td', 'th'])) for row in rows)
                                    doc_table = doc.add_table(rows=len(rows), cols=max_cols)
                                    doc_table.style = 'Table Grid'
                                    
                                    for r_idx, row in enumerate(rows):
                                        cells = row.find_all(['td', 'th'])
                                        for c_idx, cell in enumerate(cells):
                                            if c_idx < len(doc_table.rows[r_idx].cells):
                                                doc_cell = doc_table.rows[r_idx].cells[c_idx]
                                                doc_cell.text = cell.get_text(strip=True)
                                                
                                                # Enhanced styling
                                                for paragraph in doc_cell.paragraphs:
                                                    for run in paragraph.runs:
                                                        run.font.name = 'Inter'
                                                        run.font.size = Pt(9)  # Slightly smaller for better fit
                                                        
                                                        # Check if this is a header cell (th tag or first row)
                                                        is_header = cell.name == 'th' or r_idx == 0
                                                        
                                                        if is_header:
                                                            run.font.bold = True
                                                            run.font.color.rgb = RGBColor(255, 255, 255)  # White text
                                                            doc_cell._tc.get_or_add_tcPr().append(OxmlElement('w:shd'))
                                                            doc_cell._tc.get_or_add_tcPr().find(qn('w:shd')).set(qn('w:fill'), '1a365d')  # Primary color
                                                        else:
                                                            run.font.color.rgb = RGBColor(45, 55, 72)  # --color-text-primary
                                                            # Alternating row colors
                                                            if r_idx % 2 == 1:
                                                                doc_cell._tc.get_or_add_tcPr().append(OxmlElement('w:shd'))
                                                                doc_cell._tc.get_or_add_tcPr().find(qn('w:shd')).set(qn('w:fill'), 'f7fafc')  # Light background
                                    
                                    # Auto-fit column widths based on content
                                    for column in doc_table.columns:
                                        column.width = Inches(1.2)  # Slightly narrower for better fit
                                    
                                    doc.add_paragraph('')  # Add space after table
                                    print(f"✅ Added enhanced HTML table from metadata successfully")
                                    return True
                                else:
                                    print(f"⚠️ No rows found in table HTML")
                            else:
                                print(f"⚠️ No table tag found in HTML")
                        else:
                            print(f"⚠️ Unknown table type: {table_data.get('type')}")
                    except Exception as e:
                        print(f"❌ Error rendering table from metadata: {e}")
                    
                    return False
                
                def process_paragraph(paragraph_text):
                    """Process accumulated paragraph text with proper styling"""
                    if paragraph_text:
                        if paragraph_text.startswith('##'):
                            subheading = doc.add_heading(paragraph_text.replace('##', '').strip(), level=3)
                            for run in subheading.runs:
                                run.font.name = 'Inter'
                                run.font.size = Pt(14)
                                run.font.bold = True
                                run.font.color.rgb = RGBColor(56, 178, 172)  # --color-secondary
                        elif paragraph_text.startswith('- **'):
                            p = doc.add_paragraph()
                            p.style = 'List Bullet'
                            run = p.add_run(paragraph_text.replace('- **', '').replace('**', ''))
                            run.font.name = 'Inter'
                            run.font.bold = True
                            run.font.color.rgb = RGBColor(26, 54, 93)  # --color-primary
                        elif paragraph_text.startswith('- '):
                            p = doc.add_paragraph()
                            p.style = 'List Bullet'
                            run = p.add_run(paragraph_text.replace('- ', ''))
                            run.font.name = 'Inter'
                            run.font.color.rgb = RGBColor(45, 55, 72)  # --color-text-primary
                        elif paragraph_text.strip():
                            p = doc.add_paragraph(paragraph_text)
                            for run in p.runs:
                                run.font.name = 'Inter'
                                run.font.color.rgb = RGBColor(45, 55, 72)  # --color-text-primary
                
                # Process content line-by-line in original order
                for line in lines:
                    line_stripped = line.strip()
                    
                    # Check if this line is part of a markdown table
                    is_table_line = (line_stripped.startswith('|') and line_stripped.endswith('|')) or \
                                   (line_stripped.replace('-', '').replace(':', '').replace('|', '').strip() == '' and '|' in line_stripped)
                    
                    if is_table_line:
                        # Process any accumulated text first
                        if current_paragraph:
                            paragraph_text = '\n'.join(current_paragraph).strip()
                            process_paragraph(paragraph_text)
                            current_paragraph = []
                        
                        # Try to render table from metadata if available
                        if table_index < len(tables_data):
                            table_rendered = render_table_from_metadata(tables_data[table_index])
                            if table_rendered:
                                table_index += 1
                        else:
                            # Fallback: render as plain text if no metadata table available
                            p = doc.add_paragraph(line)
                            for run in p.runs:
                                run.font.name = 'Inter'
                                run.font.color.rgb = RGBColor(45, 55, 72)  # --color-text-primary
                    else:
                        # If this is a blank line, process accumulated paragraph
                        if not line_stripped and current_paragraph:
                            paragraph_text = '\n'.join(current_paragraph).strip()
                            process_paragraph(paragraph_text)
                            current_paragraph = []
                        else:
                            # Add line to current paragraph
                            current_paragraph.append(line)
                
                # Process any remaining paragraph
                if current_paragraph:
                    paragraph_text = '\n'.join(current_paragraph).strip()
                    process_paragraph(paragraph_text)
            else:
                # For non-AI responses, use the original processing logic
                # Filter out table lines from regular content
                non_table_lines = []
                filtered_table_rows = []  # Define the missing variable
                for line in lines:
                    if is_table_line(line):
                        filtered_table_rows.append(line)  # Collect table lines
                    elif line.strip():
                        non_table_lines.append(line)
                
                if filtered_table_rows:
                    parsed_rows = []
                    for row in filtered_table_rows:
                        cells = [c.strip() for c in row.strip('|').split('|')]
                        parsed_rows.append(cells)
                    if parsed_rows:
                        table = doc.add_table(rows=len(parsed_rows), cols=len(parsed_rows[0]))
                        table.style = 'Table Grid'
                        
                        for r_idx, row_cells in enumerate(parsed_rows):
                            for c_idx, cell_text in enumerate(row_cells):
                                cell = table.cell(r_idx, c_idx)
                                cell.text = str(cell_text)
                                
                                if r_idx == 0:
                                    for paragraph in cell.paragraphs:
                                        for run in paragraph.runs:
                                            run.font.name = 'Inter'
                                            run.font.bold = True
                                            run.font.color.rgb = RGBColor(255, 255, 255)  # White text
                                            run.font.size = Pt(11)
                                    cell._tc.get_or_add_tcPr().append(OxmlElement('w:shd'))
                                    cell._tc.get_or_add_tcPr().find(qn('w:shd')).set(qn('w:fill'), '1a365d')  # Primary color
                                else:
                                    for paragraph in cell.paragraphs:
                                        for run in paragraph.runs:
                                            run.font.name = 'Inter'
                                            run.font.size = Pt(10)
                                            run.font.color.rgb = RGBColor(45, 55, 72)  # --color-text-primary
                                    
                                    if r_idx % 2 == 1:
                                        cell._tc.get_or_add_tcPr().append(OxmlElement('w:shd'))
                                        cell._tc.get_or_add_tcPr().find(qn('w:shd')).set(qn('w:fill'), 'f7fafc')  # --color-background
                        
                        for column in table.columns:
                            column.width = Inches(1.5)
                    
                    if non_table_lines:
                        for line in non_table_lines:
                            line = line.strip()
                            if line.startswith('##'):
                                subheading = doc.add_heading(line.replace('##', '').strip(), level=3)
                                for run in subheading.runs:
                                    run.font.name = 'Inter'
                                    run.font.size = Pt(14)
                                    run.font.bold = True
                                    run.font.color.rgb = RGBColor(56, 178, 172)  # --color-secondary
                            elif line.startswith('- **'):
                                p = doc.add_paragraph()
                                p.style = 'List Bullet'
                                run = p.add_run(line.replace('- **', '').replace('**', ''))
                                run.font.name = 'Inter'
                                run.font.bold = True
                                run.font.color.rgb = RGBColor(26, 54, 93)  # --color-primary
                            elif line.startswith('- '):
                                p = doc.add_paragraph()
                                p.style = 'List Bullet'
                                run = p.add_run(line.replace('- ', ''))
                                run.font.name = 'Inter'
                                run.font.color.rgb = RGBColor(45, 55, 72)  # --color-text-primary
                            elif line.strip():
                                p = doc.add_paragraph(line)
                                for run in p.runs:
                                    run.font.name = 'Inter'
                                    run.font.color.rgb = RGBColor(45, 55, 72)  # --color-text-primary
                else:
                    # Process non-table content only
                    paragraphs = '\n'.join(non_table_lines).split('\n\n')
                    for para in paragraphs:
                        para = para.strip()
                        if para.startswith('##'):
                            subheading = doc.add_heading(para.replace('##', '').strip(), level=3)
                            for run in subheading.runs:
                                run.font.name = 'Inter'
                                run.font.size = Pt(14)
                                run.font.bold = True
                                run.font.color.rgb = RGBColor(56, 178, 172)  # --color-secondary
                        elif para.startswith('- **'):
                            p = doc.add_paragraph()
                            p.style = 'List Bullet'
                            run = p.add_run(para.replace('- **', '').replace('**', ''))
                            run.font.name = 'Inter'
                            run.font.bold = True
                            run.font.color.rgb = RGBColor(26, 54, 93)  # --color-primary
                        elif para.startswith('- '):
                            p = doc.add_paragraph()
                            p.style = 'List Bullet'
                            run = p.add_run(para.replace('- ', ''))
                            run.font.name = 'Inter'
                            run.font.color.rgb = RGBColor(45, 55, 72)  # --color-text-primary
                        elif para.strip():
                            p = doc.add_paragraph(para)
                            for run in p.runs:
                                run.font.name = 'Inter'
                                run.font.color.rgb = RGBColor(45, 55, 72)  # --color-text-primary

                doc.add_paragraph('')

        # Footer with DataFlow branding
        footer_paragraph = doc.add_paragraph()
        footer_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        
        footer_run = footer_paragraph.add_run("Generated by DataFlow Analytics")
        footer_run.font.name = 'Inter'
        footer_run.font.size = Pt(10)
        footer_run.font.italic = True
        footer_run.font.color.rgb = RGBColor(113, 128, 150)  # --color-text-secondary
        
        footer_paragraph.add_run(" | ")
        
        footer_run = footer_paragraph.add_run("Advanced AI-Powered Data Analysis")
        footer_run.font.name = 'Inter'
        footer_run.font.size = Pt(10)
        footer_run.font.italic = True
        footer_run.font.color.rgb = RGBColor(56, 178, 172)  # --color-secondary

        # Stream the document
        buffer = BytesIO()
        doc.save(buffer)
        buffer.seek(0)

        filename = f"DataFlow_Report_{dataset.id}_{session.id}.docx"
        response = HttpResponse(
            buffer.getvalue(),
            content_type='application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        )
        response['Content-Disposition'] = f'attachment; filename="{filename}"'
        return response

    except UserDataset.DoesNotExist:
        return JsonResponse({'error': 'Dataset not found'}, status=404)
    except AnalysisSession.DoesNotExist:
        return JsonResponse({'error': 'Session not found'}, status=404)
    except Exception as e:
        return JsonResponse({'error': f'Error generating report: {str(e)}'}, status=500)

def extract_html_tables_from_content(content):
    """Extract HTML tables from AI response content and convert them to Word format"""
    tables = []
    
    try:
        # Use BeautifulSoup to parse HTML content
        soup = BeautifulSoup(content, 'html.parser')
        
        # Find all table elements
        html_tables = soup.find_all('table')
        
        for i, table in enumerate(html_tables):
            # Extract table data
            rows = table.find_all('tr')
            if rows:
                # Determine max columns across all rows
                max_cols = max(len(row.find_all(['td', 'th'])) for row in rows)
                
                # Create table data structure
                table_data = {
                    'type': 'html',
                    'html': str(table),
                    'title': f'Table {i+1}',
                    'rows': len(rows),
                    'columns': max_cols
                }
                tables.append(table_data)
                print(f"📊 Extracted HTML table {i+1} with {len(rows)} rows and {max_cols} columns")
        
        return tables
        
    except Exception as e:
        print(f"❌ Error extracting HTML tables: {e}")
        return []