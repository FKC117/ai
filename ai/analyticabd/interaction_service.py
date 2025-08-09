from .models import AnalysisInteraction, AnalysisHistory, UserPreference
from django.contrib.auth.models import User


def record_interaction_for_user(user: User, interaction_type: str, description: str, metadata: dict) -> None:
    """Record AnalysisInteraction and maintain AnalysisHistory for the user's current dataset."""
    user_pref, _ = UserPreference.objects.get_or_create(user=user)
    current_dataset = user_pref.current_dataset
    if not current_dataset:
        raise ValueError("No current dataset set for user")

    # Create interaction
    AnalysisInteraction.objects.create(
        session=None,  # will be set by view using session_manager if needed
        interaction_type=interaction_type,
        description=description,
        metadata=metadata,
    )

    # Maintain history for known analysis types
    if interaction_type in ["summary_stats", "linear_regression", "t_test", "anova", "correlation"]:
        AnalysisHistory.objects.get_or_create(
            user=user,
            dataset=current_dataset,
            analysis_type=interaction_type,
            defaults={"is_complete": True},
        )
