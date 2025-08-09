from django.contrib import admin
from .models import (
    UserProfile,
    UserDataset,
    DatasetVariable,
    AnalysisSession,
    AnalysisInteraction,
    UserPreference,
    DatasetUIState,
    AnalysisHistory,
    UserWarningPreference,
    ReportDocument,
    ReportSection,
    ChatMessage,
    SubscriptionPlan,
    UserSubscription,
    TokenUsage,
    BillingSetting,
)


@admin.register(SubscriptionPlan)
class SubscriptionPlanAdmin(admin.ModelAdmin):
    list_display = ("name", "monthly_token_limit", "price_usd", "price_bdt_display", "is_active", "updated_at")
    list_filter = ("is_active",)
    search_fields = ("name",)

    def price_bdt_display(self, obj):
        return obj.price_bdt
    price_bdt_display.short_description = "Price (BDT)"


@admin.register(UserSubscription)
class UserSubscriptionAdmin(admin.ModelAdmin):
    list_display = ("user", "plan", "start_date", "is_active", "payment_status", "payment_at", "next_billing_date")
    list_filter = ("is_active", "plan", "payment_status")
    search_fields = ("user__username",)


@admin.register(TokenUsage)
class TokenUsageAdmin(admin.ModelAdmin):
    list_display = ("user", "tokens_used", "cost_usd", "created_at", "dataset", "session")
    list_filter = ("created_at",)
    search_fields = ("user__username",)


@admin.register(BillingSetting)
class BillingSettingAdmin(admin.ModelAdmin):
    list_display = ("usd_to_bdt_rate", "updated_at")

    def has_add_permission(self, request):
        # Limit to a single instance
        return not BillingSetting.objects.exists()


@admin.register(AnalysisSession)
class AnalysisSessionAdmin(admin.ModelAdmin):
    list_display = ("user", "dataset", "session_name", "is_active", "updated_at")
    list_filter = ("is_active", "dataset")
    search_fields = ("user__username", "session_name")


@admin.register(AnalysisInteraction)
class AnalysisInteractionAdmin(admin.ModelAdmin):
    list_display = ("session", "interaction_type", "created_at")
    list_filter = ("interaction_type",)
    search_fields = ("session__session_name",)


@admin.register(UserPreference)
class UserPreferenceAdmin(admin.ModelAdmin):
    list_display = ("user", "current_dataset", "default_analysis_type", "updated_at")
    search_fields = ("user__username",)


@admin.register(DatasetUIState)
class DatasetUIStateAdmin(admin.ModelAdmin):
    list_display = ("user", "dataset", "current_session", "updated_at")
    search_fields = ("user__username", "dataset__name")


@admin.register(AnalysisHistory)
class AnalysisHistoryAdmin(admin.ModelAdmin):
    list_display = ("user", "dataset", "analysis_type", "is_complete", "created_at")
    list_filter = ("analysis_type", "is_complete")
    search_fields = ("user__username", "dataset__name")


@admin.register(UserWarningPreference)
class UserWarningPreferenceAdmin(admin.ModelAdmin):
    list_display = ("user", "show_delete_warning", "show_multiselect_help", "updated_at")
    search_fields = ("user__username",)


@admin.register(ReportDocument)
class ReportDocumentAdmin(admin.ModelAdmin):
    list_display = ("user", "dataset", "session", "title", "updated_at")
    search_fields = ("user__username", "dataset__name", "title")


@admin.register(ReportSection)
class ReportSectionAdmin(admin.ModelAdmin):
    list_display = ("document", "order", "title", "section_type", "created_at")
    list_filter = ("section_type",)
    search_fields = ("document__title", "title")


@admin.register(ChatMessage)
class ChatMessageAdmin(admin.ModelAdmin):
    list_display = ("session", "message_type", "created_at", "is_added_to_report")
    list_filter = ("message_type", "is_added_to_report")
    search_fields = ("session__session_name",)


# Keep basic registrations for core models
admin.site.register(UserProfile)
admin.site.register(UserDataset)
admin.site.register(DatasetVariable)
