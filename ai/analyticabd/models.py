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
