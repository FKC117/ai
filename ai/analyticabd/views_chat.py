from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import json

from .models import UserDataset, AnalysisSession, ChatMessage
from .session_manager import resolve_session_for
from .analytics_service import get_summary_statistics_data


@login_required
@csrf_exempt
@require_http_methods(["POST"])
def send_chat_message(request):
    """Handle AI chat messages and store conversation history"""
    print("=== BACKEND AI CHAT DEBUG START ===")
    try:
        data = json.loads(request.body)
        message = data.get('message', '').strip()
        dataset_id = data.get('dataset_id')
        session_id = data.get('session_id')

        print(f"Received request - Message: '{message[:50]}...' | Dataset ID: {dataset_id} | Session ID: {session_id}")

        if not message or not dataset_id:
            print("❌ Validation failed - missing message or dataset_id")
            return JsonResponse({'error': 'Message and dataset_id are required'}, status=400)

        print(f"✅ Validation passed - processing dataset {dataset_id}")
        dataset = UserDataset.objects.get(id=dataset_id, user=request.user)
        print(f"✅ Dataset found: {dataset.name} ({dataset.rows} rows, {dataset.columns} columns)")

        # Get or create session
        session = resolve_session_for(request.user, int(dataset_id), int(session_id) if session_id else None)
        if not session:
            session = AnalysisSession.objects.create(
                user=request.user,
                dataset=dataset,
                session_name=f"Chat Session for {dataset.name}"
            )
            print(f"✅ Created new session: {session.id}")

        # Store user message
        user_message = ChatMessage.objects.create(
            session=session,
            message_type='user',
            content=message
        )
        print(f"✅ Stored user message: {user_message.id}")

        # Get summary statistics for context
        print("📊 Getting summary statistics...")
        summary_data = get_summary_statistics_data(dataset_id)
        print(f"✅ Summary data retrieved - {len(summary_data.get('variable_summary', {}))} variables")

        # Prepare context for AI
        context = {
            'dataset_name': dataset.name,
            'dataset_rows': dataset.rows,
            'dataset_columns': dataset.columns,
            'summary_statistics': summary_data,
            'user_message': message,
            'dataset_info': {
                'rows': dataset.rows,
                'columns': dataset.columns,
                'name': dataset.name
            }
        }

        # Get previous chat messages for context
        previous_messages = ChatMessage.objects.filter(session=session).order_by('created_at')[:10]
        chat_history = []
        for msg in previous_messages:
            if msg.message_type == 'user':
                chat_history.append(f"User: {msg.content}")
            else:
                chat_history.append(f"AI: {msg.content}")

        context['chat_history'] = '\n'.join(chat_history[-6:])  # Last 6 messages for context
        print(f"✅ Chat history prepared - {len(chat_history)} previous messages")

        # Use the actual AI client
        print("🤖 Initializing AI client...")
        try:
            from .ai.llm_client import LLMClient
            llm_client = LLMClient()
            print("✅ AI client initialized successfully")

            # Build the full message with context
            full_message = f"""
User Query: {message}

Dataset Information:
- Name: {dataset.name}
- Rows: {dataset.rows:,}
- Columns: {dataset.columns}

Previous Chat History:
{context['chat_history']}

Please provide a comprehensive analysis based on the user's query and the dataset information provided.
"""
            print(f"📝 Sending message to AI (length: {len(full_message)} chars)")

            ai_response = llm_client.chat(full_message, context)
            print(f"✅ AI response received (length: {len(ai_response)} chars)")
            print(f"AI response preview: {ai_response[:200]}...")

        except Exception as ai_error:
            print(f"❌ AI Error: {ai_error}")
            print(f"AI Error type: {type(ai_error).__name__}")
            print(f"AI Error details: {str(ai_error)}")

            # Fallback to basic response if AI fails
            ai_response = f"I'm analyzing your dataset '{dataset.name}' with {dataset.rows:,} rows and {dataset.columns} columns. Here's what I found:\n\n"

            # Add some intelligent response based on the message
            if 'summary' in message.lower() or 'statistics' in message.lower():
                ai_response += "## Summary Statistics Overview\n\n"
                ai_response += f"- **Total Variables**: {len(summary_data.get('variable_summary', {}))}\n"
                ai_response += f"- **Numeric Variables**: {summary_data.get('dataset_overview', {}).get('numeric_columns', 0)}\n"
                ai_response += f"- **Categorical Variables**: {summary_data.get('dataset_overview', {}).get('categorical_columns', 0)}\n\n"

                var_summary = summary_data.get('variable_summary', {})
                if var_summary:
                    ai_response += "## Key Variable Insights\n\n"
                    for i, (var_name, var_data) in enumerate(list(var_summary.items())[:5]):
                        if var_data.get('type') == 'numeric':
                            ai_response += f"- **{var_name}**: Mean = {var_data.get('mean', 'N/A'):.2f}, Std = {var_data.get('std', 'N/A'):.2f}\n"
                        else:
                            ai_response += f"- **{var_name}**: {var_data.get('unique_count', 'N/A')} unique values\n"

            elif 'quality' in message.lower() or 'missing' in message.lower():
                ai_response += "## Data Quality Analysis\n\n"
                dq = summary_data.get('data_quality', {})
                if dq:
                    for var_name, var_stats in list(dq.items())[:5]:
                        missing_pct = var_stats.get('missing_percentage', 0)
                        ai_response += f"- **{var_name}**: {missing_pct:.1%} missing values\n"

            elif 'correlation' in message.lower():
                ai_response += "## Correlation Analysis\n\n"
                corr = summary_data.get('correlation_matrix', {})
                if corr and corr.get('strong_correlations'):
                    ai_response += "Strong correlations found:\n"
                    for corr_item in corr['strong_correlations'][:3]:
                        ai_response += f"- **{corr_item['variable1']}** ↔ **{corr_item['variable2']}**: {corr_item['correlation']:.3f}\n"
                else:
                    ai_response += "No strong correlations detected in the dataset.\n"

            else:
                ai_response += "I can help you analyze your dataset! Here are some things you can ask me about:\n\n"
                ai_response += "- **Summary statistics** and key insights\n"
                ai_response += "- **Data quality** and missing value analysis\n"
                ai_response += "- **Correlation analysis** between variables\n"
                ai_response += "- **Distribution patterns** and outliers\n"
                ai_response += "- **Recommendations** for further analysis\n\n"
                ai_response += "What specific aspect would you like me to focus on?"

            print(f"✅ Fallback response generated (length: {len(ai_response)} chars)")

        # Store AI response
        ai_message = ChatMessage.objects.create(
            session=session,
            message_type='ai',
            content=ai_response
        )
        print(f"✅ Stored AI message: {ai_message.id}")

        response_data = {
            'success': True,
            'response': ai_response,
            'session_id': session.id,
            'message_id': ai_message.id
        }
        print(f"✅ Sending response - Session ID: {session.id}, Message ID: {ai_message.id}")
        print("=== BACKEND AI CHAT DEBUG SUCCESS ===")

        return JsonResponse(response_data)

    except UserDataset.DoesNotExist:
        print("❌ Dataset not found")
        print("=== BACKEND AI CHAT DEBUG ERROR ===")
        return JsonResponse({'error': 'Dataset not found'}, status=404)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        print(f"Error type: {type(e).__name__}")
        print(f"Error details: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        print("=== BACKEND AI CHAT DEBUG ERROR ===")
        return JsonResponse({'error': f'Error processing message: {str(e)}'}, status=500)


@login_required
def get_chat_history(request):
    """Retrieve chat history for a specific session"""
    try:
        dataset_id = request.GET.get('dataset_id')
        session_id = request.GET.get('session_id')

        if not dataset_id:
            return JsonResponse({'error': 'Dataset ID is required'}, status=400)

        session = resolve_session_for(request.user, int(dataset_id), int(session_id) if session_id else None)

        if not session:
            return JsonResponse({'messages': [], 'session_id': None})

        messages = ChatMessage.objects.filter(session=session).order_by('created_at')
        chat_history = []
        for message in messages:
            chat_history.append({
                'id': message.id,
                'type': message.message_type,
                'content': message.content,
                'created_at': message.created_at.strftime('%Y-%m-%d %H:%M:%S'),
                'is_added_to_report': message.is_added_to_report,
                'report_section_id': message.report_section.id if message.report_section else None,
            })

        return JsonResponse({'messages': chat_history, 'session_id': session.id, 'session_name': session.session_name})

    except UserDataset.DoesNotExist:
        return JsonResponse({'error': 'Dataset not found'}, status=404)
    except Exception as e:
        return JsonResponse({'error': f'Error retrieving chat history: {str(e)}'}, status=500)
