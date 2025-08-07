from django.contrib import admin
from .models import UserProfile, UserDataset, DatasetVariable

# Register your models here.

@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = ('user', 'phone_number', 'country_code', 'created_at')
    list_filter = ('country_code', 'created_at')
    search_fields = ('user__email', 'user__first_name', 'user__last_name', 'phone_number')
    readonly_fields = ('created_at', 'updated_at')

@admin.register(UserDataset)
class UserDatasetAdmin(admin.ModelAdmin):
    list_display = ('user', 'name', 'file_size', 'rows', 'columns', 'uploaded_at')
    list_filter = ('uploaded_at',)
    search_fields = ('name', 'user__email')

@admin.register(DatasetVariable)
class DatasetVariableAdmin(admin.ModelAdmin):
    list_display = ('dataset', 'name', 'data_type', 'count', 'missing_count')
    list_filter = ('data_type',)
    search_fields = ('dataset__name', 'name')
