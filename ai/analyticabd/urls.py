from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='homepage'),
    path('account/', views.account_management, name='account_management'),
    path('dashboard/', views.analytics, name='analytics_dashboard'),
    path('getting-started/', views.getting_started, name='getting_started'),
    path('pricing/', views.pricing_plans, name='pricing'),
    path('signup/', views.signup, name='signup'),
    path('signin/', views.signin, name='signin'),
    path('signout/', views.signout, name='signout'),
    path('profile/', views.profile, name='profile'),
    
    # Dataset management URLs
    path('api/upload-dataset/', views.upload_dataset, name='upload_dataset'),
    path('api/datasets/', views.get_user_datasets, name='get_user_datasets'),
    path('api/summary-statistics/<int:dataset_id>/', views.get_summary_statistics, name='get_summary_statistics'),
]
