from django.urls import path, include
from . import views

urlpatterns = [
    path('', views.home, name='homepage'),
    path('account/', views.account_management, name='account_management'),
    path('dashboard/', views.analytics, name='analytics_dashboard'),
    path('getting-started/', views.getting_started, name='getting_started'),
    path('pricing/', views.pricing_plans, name='pricing'),
    
    # Authentication URLs
    path('signup/', views.signup, name='signup'),
    path('signin/', views.signin, name='signin'),
    path('signout/', views.signout, name='signout'),
    path('profile/', views.profile, name='profile'),
]
