from django.urls import path
from . import views
from . import views_chat
from . import views_reports

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
    path('api/analysis/<int:dataset_id>/', views.get_dataset_analysis, name='get_dataset_analysis'),
    path('api/analysis/<int:dataset_id>/heatmap.png', views.correlation_heatmap, name='correlation_heatmap'),
    path('api/user-state/', views.get_user_state, name='get_user_state'),
    path('api/set-current-dataset/', views.set_current_dataset, name='set_current_dataset'),
    path('api/record-interaction/', views.record_interaction, name='record_interaction'),
    path('api/analysis-history/<int:dataset_id>/', views.get_analysis_history, name='get_analysis_history'),
    path('api/delete-dataset/', views.delete_dataset, name='delete_dataset'),
    path('api/warning-preferences/', views.get_warning_preferences, name='get_warning_preferences'),
    path('api/update-warning-preferences/', views.update_warning_preferences, name='update_warning_preferences'),
    path('api/save-ui-state/', views.save_ui_state, name='save_ui_state'),
    path('api/billing/summary/', views.get_user_billing_summary, name='get_user_billing_summary'),
    path('api/account/overview/', views.get_account_overview, name='get_account_overview'),
    
    # AI Chat endpoints
    path('api/v1/chat/send_message/', views_chat.send_chat_message, name='send_chat_message'),
    path('api/v1/chat/history/', views_chat.get_chat_history, name='get_chat_history'),
    path('api/v1/report/add/', views_reports.add_to_report, name='add_to_report'),
    path('api/v1/report/download', views_reports.download_report, name='download_report'),
]
