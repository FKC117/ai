from django.db import models
from django.contrib.auth.models import User
from django.core.validators import RegexValidator
from decimal import Decimal
from django.utils import timezone
from datetime import timedelta

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


class DatasetUIState(models.Model):
    """Store dataset-specific UI state and session information"""
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='dataset_ui_states')
    dataset = models.ForeignKey(UserDataset, on_delete=models.CASCADE, related_name='ui_states')
    current_session = models.ForeignKey(AnalysisSession, on_delete=models.SET_NULL, null=True, blank=True, related_name='ui_state_for_dataset')
    ui_state = models.JSONField(default=dict, blank=True)  # Store UI preferences like dataset_index, chat_scroll_position, etc.
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        unique_together = ['user', 'dataset']
        ordering = ['-updated_at']
    
    def __str__(self):
        return f"{self.user.username} - {self.dataset.name} - UI State"

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


class SubscriptionPlan(models.Model):
    """Subscription plan defining monthly token limits and pricing."""
    name = models.CharField(max_length=100, unique=True)
    monthly_token_limit = models.PositiveIntegerField(default=0)
    price_usd = models.DecimalField(max_digits=10, decimal_places=2, default=Decimal('0.00'))
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['price_usd']

    def __str__(self):
        return f"{self.name} ({self.monthly_token_limit} tok/mo)"

    @property
    def price_bdt(self):
        from .models import BillingSetting
        rate = BillingSetting.get_rate()
        return (self.price_usd * rate).quantize(Decimal('0.01'))


class UserSubscription(models.Model):
    """Active subscription of a user to a plan."""
    STATUS_CHOICES = (
        ('pending', 'Pending'),
        ('paid', 'Paid'),
        ('failed', 'Failed'),
        ('canceled', 'Canceled'),
    )

    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='subscriptions')
    plan = models.ForeignKey(SubscriptionPlan, on_delete=models.PROTECT, related_name='user_subscriptions')
    start_date = models.DateField(auto_now_add=True)
    is_active = models.BooleanField(default=True)
    canceled_at = models.DateTimeField(null=True, blank=True)

    payment_status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    payment_at = models.DateTimeField(null=True, blank=True)
    next_billing_date = models.DateField(null=True, blank=True)

    class Meta:
        ordering = ['-start_date']
        indexes = [models.Index(fields=['user', 'is_active'])]

    def __str__(self):
        return f"{self.user.username} -> {self.plan.name} ({'active' if self.is_active else 'inactive'})"

    def save(self, *args, **kwargs):
        # Auto-calc next_billing_date on a 30-day interval
        base_date = None
        if self.payment_at:
            base_date = self.payment_at.date()
        elif self.start_date:
            base_date = self.start_date
        else:
            base_date = timezone.now().date()

        # Only set/update if not set or if based on payment_at
        if self.payment_at or not self.next_billing_date:
            self.next_billing_date = base_date + timedelta(days=30)
        super().save(*args, **kwargs)


class TokenUsage(models.Model):
    """Tracks token consumption per event for billing and quota enforcement."""
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='token_usages')
    session = models.ForeignKey(AnalysisSession, on_delete=models.SET_NULL, null=True, blank=True, related_name='token_usages')
    dataset = models.ForeignKey(UserDataset, on_delete=models.SET_NULL, null=True, blank=True, related_name='token_usages')
    tokens_used = models.PositiveIntegerField()
    cost_usd = models.DecimalField(max_digits=12, decimal_places=6, default=Decimal('0.0'))
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']
        indexes = [models.Index(fields=['user', 'created_at'])]

    def __str__(self):
        return f"{self.user.username} used {self.tokens_used} tokens (${self.cost_usd}) on {self.created_at}"


class BillingSetting(models.Model):
    """Global billing settings editable via admin (singleton)."""
    usd_to_bdt_rate = models.DecimalField(max_digits=10, decimal_places=4, default=Decimal('122.5'))
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Billing Settings (USD->BDT: {self.usd_to_bdt_rate})"

    @classmethod
    def get_rate(cls) -> Decimal:
        obj = cls.objects.first()
        return obj.usd_to_bdt_rate if obj else Decimal('122.5')
