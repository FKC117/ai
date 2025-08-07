from django.shortcuts import render, redirect
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from .forms import CustomUserCreationForm, CustomAuthenticationForm

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