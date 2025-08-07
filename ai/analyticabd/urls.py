from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='homepage'),
    path('account/', views.account_management, name='account_management'),
    path('dashboard/', views.analytics, name='analytics_dashboard'),
    path('getting-started/', views.getting_started, name='getting_started'),
    path('pricing/', views.pricing_plans, name='pricing_plans'),
    
    # Authentication URLs
    path('signup/', views.signup, name='signup'),
    path('signin/', views.signin, name='signin'),
    path('signout/', views.signout, name='signout'),
    path('profile/', views.profile, name='profile'),
    
    # API endpoints
    path('api/upload-dataset/', views.upload_dataset, name='upload_dataset'),
    path('api/datasets/', views.get_user_datasets, name='get_user_datasets'),
    path('api/summary-statistics/<int:dataset_id>/', views.get_summary_statistics, name='get_summary_statistics'),
    path('api/user-state/', views.get_user_state, name='get_user_state'),
    path('api/set-current-dataset/', views.set_current_dataset, name='set_current_dataset'),
    path('api/record-interaction/', views.record_interaction, name='record_interaction'),
    path('api/analysis-history/<int:dataset_id>/', views.get_analysis_history, name='get_analysis_history'),
    path('api/delete-dataset/', views.delete_dataset, name='delete_dataset'),
    path('api/warning-preferences/', views.get_warning_preferences, name='get_warning_preferences'),
    path('api/update-warning-preferences/', views.update_warning_preferences, name='update_warning_preferences'),
    
    # AI Chat endpoints
    path('api/v1/chat/send_message/', views.send_chat_message, name='send_chat_message'),
]
