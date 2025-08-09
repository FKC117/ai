from typing import Optional, Tuple, Dict, List
from django.contrib.auth.models import User
from django.utils import timezone

from .models import (
    UserDataset,
    AnalysisSession,
    DatasetUIState,
    ChatMessage,
    UserPreference,
)


def get_dataset_for_user(user: User, dataset_id: int) -> UserDataset:
    """Return a dataset owned by user or raise DoesNotExist."""
    return UserDataset.objects.get(id=dataset_id, user=user)


def get_or_create_active_session(user: User, dataset: UserDataset) -> AnalysisSession:
    """Get or create the active AnalysisSession for a user and dataset."""
    session, created = AnalysisSession.objects.get_or_create(
        user=user,
        dataset=dataset,
        is_active=True,
        defaults={"session_name": f"Analysis of {dataset.name}"},
    )
    if not created:
        session.updated_at = timezone.now()
        session.save(update_fields=["updated_at"])
    return session


def set_current_dataset_and_session(user: User, dataset_id: int) -> Tuple[UserDataset, AnalysisSession]:
    """Set user's current dataset and ensure an active session is linked in DatasetUIState."""
    dataset = get_dataset_for_user(user, dataset_id)

    # Persist current dataset on user preference
    user_pref, _ = UserPreference.objects.get_or_create(user=user)
    user_pref.current_dataset = dataset
    user_pref.save(update_fields=["current_dataset"])

    # Ensure active session
    session = get_or_create_active_session(user, dataset)

    # Link in dataset-specific UI state
    dataset_ui_state, created = DatasetUIState.objects.get_or_create(
        user=user,
        dataset=dataset,
        defaults={"current_session": session},
    )
    if not created:
        dataset_ui_state.current_session = session
        dataset_ui_state.save(update_fields=["current_session"])

    return dataset, session


def get_current_session_id_and_ui_state(user: User) -> Tuple[Optional[int], Dict]:
    """Return (session_id, ui_state) for the user's current dataset, if any."""
    user_pref, _ = UserPreference.objects.get_or_create(user=user)
    if not user_pref.current_dataset:
        return None, {}

    dataset_ui_state = DatasetUIState.objects.filter(
        user=user, dataset=user_pref.current_dataset
    ).first()
    if not dataset_ui_state:
        return None, {}

    session_id = dataset_ui_state.current_session.id if dataset_ui_state.current_session else None
    return session_id, dataset_ui_state.ui_state or {}


def save_dataset_ui_state(user: User, dataset_id: int, ui_state: Dict) -> None:
    """Save UI state for a specific user+dataset in DatasetUIState."""
    dataset = get_dataset_for_user(user, dataset_id)
    dataset_ui_state, created = DatasetUIState.objects.get_or_create(
        user=user,
        dataset=dataset,
        defaults={"ui_state": ui_state},
    )
    if not created:
        dataset_ui_state.ui_state = ui_state
        dataset_ui_state.save(update_fields=["ui_state"])


def resolve_session_for(user: User, dataset_id: int, session_id: Optional[int]) -> Optional[AnalysisSession]:
    """Resolve a valid AnalysisSession for a given user, dataset and optional session_id."""
    dataset = get_dataset_for_user(user, dataset_id)

    # Explicit session id
    if session_id:
        try:
            return AnalysisSession.objects.get(id=int(session_id), user=user, dataset=dataset)
        except (ValueError, AnalysisSession.DoesNotExist):
            pass

    # Fallback to active session for dataset
    session = AnalysisSession.objects.filter(user=user, dataset=dataset, is_active=True).first()
    if session:
        return session

    return None


def get_chat_history_for_session(session: AnalysisSession) -> List[Dict]:
    """Return serialized chat messages for the session ordered by time."""
    messages = ChatMessage.objects.filter(session=session).order_by("created_at")
    chat_history: List[Dict] = []
    for message in messages:
        chat_history.append(
            {
                "id": message.id,
                "type": message.message_type,
                "content": message.content,
                "created_at": message.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                "is_added_to_report": message.is_added_to_report,
                "report_section_id": message.report_section.id if message.report_section else None,
            }
        )
    return chat_history
