from django.db import models
from django.contrib.auth.models import User
from django.core.validators import RegexValidator

# Create your models here.

class UserProfile(models.Model):
    """Extended user profile with additional information"""
    
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    
    # Phone number field with validation
    phone_regex = RegexValidator(
        regex=r'^\+?1?\d{9,15}$',
        message="Phone number must be entered in the format: '+999999999'. Up to 15 digits allowed."
    )
    phone_number = models.CharField(
        validators=[phone_regex], 
        max_length=17, 
        blank=True,
        null=True,
        help_text="Phone number in format: +999999999"
    )
    
    # Country code for phone number
    country_code = models.CharField(
        max_length=10,
        blank=True,
        null=True,
        help_text="Country code (e.g., +1, +44)"
    )
    
    # Additional fields can be added here
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"{self.user.email} - {self.phone_number}"
    
    class Meta:
        verbose_name = 'User Profile'
        verbose_name_plural = 'User Profiles'

class UserDataset(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='datasets')
    name = models.CharField(max_length=255)
    file = models.FileField(upload_to='user_datasets/')
    file_size = models.BigIntegerField()  # Size in bytes
    rows = models.IntegerField()
    columns = models.IntegerField()
    numeric_columns = models.IntegerField()
    categorical_columns = models.IntegerField()
    missing_values_percentage = models.FloatField(default=0.0)
    duplicate_rows = models.IntegerField(default=0)
    outliers_count = models.IntegerField(default=0)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    is_active = models.BooleanField(default=True)
    
    class Meta:
        ordering = ['-uploaded_at']
    
    def __str__(self):
        return f"{self.user.username} - {self.name}"

class DatasetVariable(models.Model):
    dataset = models.ForeignKey(UserDataset, on_delete=models.CASCADE, related_name='variables')
    name = models.CharField(max_length=255)
    data_type = models.CharField(max_length=50)  # 'numeric', 'categorical', 'datetime'
    count = models.IntegerField()
    missing_count = models.IntegerField(default=0)
    
    # For numeric variables
    mean = models.FloatField(null=True, blank=True)
    std_dev = models.FloatField(null=True, blank=True)
    min_value = models.FloatField(null=True, blank=True)
    q25 = models.FloatField(null=True, blank=True)
    median = models.FloatField(null=True, blank=True)
    q75 = models.FloatField(null=True, blank=True)
    max_value = models.FloatField(null=True, blank=True)
    outliers_count = models.IntegerField(default=0)  # Add this field
    
    # For categorical variables
    unique_values = models.IntegerField(null=True, blank=True)
    most_common_value = models.CharField(max_length=255, null=True, blank=True)
    most_common_count = models.IntegerField(null=True, blank=True)
    
    class Meta:
        ordering = ['name']
    
    def __str__(self):
        return f"{self.dataset.name} - {self.name}"

class AnalysisSession(models.Model):
    """Store analysis sessions for LLM context"""
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='analysis_sessions')
    dataset = models.ForeignKey(UserDataset, on_delete=models.CASCADE, related_name='analysis_sessions')
    session_name = models.CharField(max_length=255, default='Analysis Session')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    is_active = models.BooleanField(default=True)
    
    class Meta:
        ordering = ['-updated_at']
    
    def __str__(self):
        return f"{self.user.username} - {self.dataset.name} - {self.session_name}"

class AnalysisInteraction(models.Model):
    """Store individual interactions for LLM context"""
    session = models.ForeignKey(AnalysisSession, on_delete=models.CASCADE, related_name='interactions')
    interaction_type = models.CharField(max_length=50)  # 'upload', 'summary_stats', 'chart_view', 'llm_query'
    description = models.TextField()
    metadata = models.JSONField(default=dict, blank=True)  # Store additional context
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['created_at']
    
    def __str__(self):
        return f"{self.session.session_name} - {self.interaction_type} - {self.created_at}"

class UserPreference(models.Model):
    """Store user preferences and settings"""
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='preferences')
    current_dataset = models.ForeignKey(UserDataset, on_delete=models.SET_NULL, null=True, blank=True, related_name='current_for_users')
    default_analysis_type = models.CharField(max_length=50, default='summary_stats')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"{self.user.username} - Preferences"

class AnalysisHistory(models.Model):
    """Track analysis history for each dataset"""
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='analysis_history')
    dataset = models.ForeignKey(UserDataset, on_delete=models.CASCADE, related_name='analysis_history')
    analysis_type = models.CharField(max_length=50)  # 'summary_stats', 'linear_regression', etc.
    created_at = models.DateTimeField(auto_now_add=True)
    is_complete = models.BooleanField(default=True)
    
    class Meta:
        ordering = ['-created_at']
        unique_together = ['user', 'dataset', 'analysis_type']
    
    def __str__(self):
        return f"{self.user.username} - {self.dataset.name} - {self.analysis_type}"

class UserWarningPreference(models.Model):
    """Store user preferences for warning dialogs"""
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='warning_preferences')
    show_delete_warning = models.BooleanField(default=True)
    show_multiselect_help = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"{self.user.username} - Warning Preferences"


class ReportDocument(models.Model):
    """Stores a user's report for a specific dataset and analysis session"""
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='reports')
    dataset = models.ForeignKey(UserDataset, on_delete=models.CASCADE, related_name='reports')
    session = models.ForeignKey(AnalysisSession, on_delete=models.CASCADE, related_name='reports')
    title = models.CharField(max_length=255, default='DataFlow Analytics Report')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-updated_at']
        unique_together = ['user', 'dataset', 'session']

    def __str__(self):
        return f"Report: {self.title} ({self.user.username} / {self.dataset.name})"


class ReportSection(models.Model):
    """A section within a report document (e.g., AI interpretation, summary stats)"""
    document = models.ForeignKey(ReportDocument, on_delete=models.CASCADE, related_name='sections')
    order = models.IntegerField(default=0)
    title = models.CharField(max_length=255)
    content = models.TextField()  # Store markdown/plain text from AI
    content_type = models.CharField(max_length=32, default='markdown')  # markdown | text | html
    section_type = models.CharField(max_length=64, default='ai_response')  # ai_response | summary_statistics | custom
    metadata = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['order', 'created_at']

    def __str__(self):
        return f"{self.document.title} - Section {self.order}: {self.title}"


class ChatMessage(models.Model):
    """Store chat messages for each analysis session"""
    session = models.ForeignKey(AnalysisSession, on_delete=models.CASCADE, related_name='chat_messages')
    message_type = models.CharField(max_length=20, choices=[
        ('user', 'User Message'),
        ('ai', 'AI Response')
    ])
    content = models.TextField()
    is_added_to_report = models.BooleanField(default=False)
    report_section = models.ForeignKey(ReportSection, on_delete=models.SET_NULL, null=True, blank=True, related_name='chat_message')
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['created_at']

    def __str__(self):
        return f"{self.session.session_name} - {self.message_type} - {self.created_at.strftime('%Y-%m-%d %H:%M')}"
