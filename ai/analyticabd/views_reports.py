from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
import json

from .models import UserDataset, AnalysisSession, ReportDocument, ReportSection, ChatMessage
from .report_generator import generate_report_document, extract_html_tables_from_content
from .analytics_service import get_summary_statistics_data


@login_required
@csrf_exempt
@require_http_methods(["POST"])
def add_to_report(request):
    """Add content to the user's report document"""
    print("=== ADD TO REPORT DEBUG START ===")
    try:
        data = json.loads(request.body)
        dataset_id = data.get('dataset_id')
        session_id = data.get('session_id')
        title = data.get('title', 'Analysis Section')
        content = data.get('content', '')
        section_type = data.get('section_type', 'general')
        metadata = data.get('metadata', {})
        message_id = data.get('message_id')
        report_data = data.get('report_data', {})  # tables and images

        if not dataset_id:
            return JsonResponse({'error': 'Dataset ID is required'}, status=400)

        dataset = UserDataset.objects.get(id=dataset_id, user=request.user)

        # Resolve session
        try:
            resolved_session = AnalysisSession.objects.get(id=int(session_id), user=request.user, dataset=dataset)
        except Exception:
            resolved_session = AnalysisSession.objects.filter(user=request.user, dataset=dataset, is_active=True).order_by('-updated_at').first()
            if not resolved_session:
                resolved_session = AnalysisSession.objects.create(user=request.user, dataset=dataset, session_name=f"Analysis of {dataset.name}")

        document, _ = ReportDocument.objects.get_or_create(user=request.user, dataset=dataset, session=resolved_session)

        # Determine order
        last_section = document.sections.order_by('-order').first()
        next_order = (last_section.order + 1) if last_section else 1

        # If summary_statistics and no content provided, auto-generate concise block
        if (section_type == 'summary_statistics') and (not content):
            try:
                stats = get_summary_statistics_data(dataset.id)
                ov = stats.get('dataset_overview', {})
                content_parts = []
                content_parts.append(f"## Dataset Overview\n")
                content_parts.append(f"Total Rows: {ov.get('total_rows', 'N/A')} | Total Columns: {ov.get('total_columns', 'N/A')}\n")

                dq = stats.get('data_quality', {})
                if dq:
                    content_parts.append(f"## Data Quality Report\n")
                    for var_name, var_stats in dq.items():
                        missing_pct = var_stats.get('missing_percentage', 0)
                        completeness = var_stats.get('completeness', 0)
                        quality_score = var_stats.get('quality_score', 0)
                        content_parts.append(f"- **{var_name}**: Missing {missing_pct:.1%}, Completeness {completeness:.1%}, Quality Score {quality_score:.1f}\n")

                di = stats.get('distribution_insights', {})
                if di:
                    content_parts.append(f"## Distribution Insights\n")
                    content_parts.append(f"- Skewed Variables: {di.get('skewed_variables', 0)}\n")
                    content_parts.append(f"- Normal Distribution: {di.get('normal_distribution', 0)}\n")
                    content_parts.append(f"- High Variance: {di.get('high_variance', 0)}\n")
                    content_parts.append(f"- Zero Variance: {di.get('zero_variance', 0)}\n")

                content_parts.append(f"## Complete Dataset Summary Table\n")
                var_items = list(stats.get('variable_summary', {}).items())
                if var_items:
                    table_rows = ['| Variable | Type | Count | Missing % | Mean | Std Dev | Min | 25th % | Median | 75th % | Max | Skewness | Kurtosis | Quality Score |']
                    table_rows.append('|----------|------|-------|-----------|------|---------|-----|---------|--------|---------|-----|----------|----------|--------------|')

                    for name, meta in var_items:
                        dq_info = stats.get('data_quality', {}).get(name, {})
                        missing_pct = dq_info.get('missing_percentage', 0)
                        quality_score = dq_info.get('quality_score', 1.0)

                        if meta.get('type') == 'numeric':
                            mean_val = meta.get('mean', 'N/A')
                            std_val = meta.get('std', 'N/A')
                            min_val = meta.get('min', 'N/A')
                            q25_val = meta.get('q25', 'N/A')
                            median_val = meta.get('median', 'N/A')
                            q75_val = meta.get('q75', 'N/A')
                            max_val = meta.get('max', 'N/A')
                            skewness = meta.get('skewness', 'N/A')
                            kurtosis = meta.get('kurtosis', 'N/A')

                            if isinstance(mean_val, (int, float)): mean_val = f"{mean_val:.2f}"
                            if isinstance(std_val, (int, float)): std_val = f"{std_val:.2f}"
                            if isinstance(min_val, (int, float)): min_val = f"{min_val:.2f}"
                            if isinstance(q25_val, (int, float)): q25_val = f"{q25_val:.2f}"
                            if isinstance(median_val, (int, float)): median_val = f"{median_val:.2f}"
                            if isinstance(q75_val, (int, float)): q75_val = f"{q75_val:.2f}"
                            if isinstance(max_val, (int, float)): max_val = f"{max_val:.2f}"
                            if isinstance(skewness, (int, float)): skewness = f"{skewness:.2f}"
                            if isinstance(kurtosis, (int, float)): kurtosis = f"{kurtosis:.2f}"

                            row = f"| {name} | numeric | {meta.get('count', 'N/A')} | {missing_pct:.1%} | {mean_val} | {std_val} | {min_val} | {q25_val} | {median_val} | {q75_val} | {max_val} | {skewness} | {kurtosis} | {quality_score:.2f} |"
                        else:
                            unique_count = meta.get('unique_count', 'N/A')
                            most_common = meta.get('most_common', 'N/A')
                            most_common_count = meta.get('most_common_count', 'N/A')
                            row = f"| {name} | categorical | {meta.get('count', 'N/A')} | {missing_pct:.1%} | {unique_count} unique | {most_common} ({most_common_count}) |  |  |  |  |  |  |  | {quality_score:.2f} |"
                        table_rows.append(row)

                    content_parts.append('\n'.join(table_rows))

                    corr = stats.get('correlation_matrix', {})
                    if corr and corr.get('strong_correlations'):
                        content_parts.append(f"\n## Correlation Summary\n")
                        content_parts.append('| Variable 1 | Variable 2 | Correlation | Strength |')
                        content_parts.append('|------------|------------|-------------|----------|')
                        for corr_item in corr['strong_correlations']:
                            corr_val = corr_item['correlation']
                            strength = 'Strong' if abs(corr_val) >= 0.8 else 'Moderate' if abs(corr_val) >= 0.6 else 'Weak'
                            content_parts.append(f"| {corr_item['variable1']} | {corr_item['variable2']} | {corr_val:.3f} | {strength} |")

                content = ('\n\n'.join(content_parts))
            except Exception:
                content = 'Summary statistics added.'

        # Persist report_data metadata
        if report_data:
            metadata['report_data'] = report_data

        # Auto-extract HTML tables from AI responses if no report_data provided
        if section_type == 'ai_response' and content and not report_data:
            extracted_tables = extract_html_tables_from_content(content)
            if extracted_tables:
                metadata.setdefault('report_data', {}).setdefault('tables', []).extend(extracted_tables)

        # Add comprehensive summary table to metadata for summary_statistics
        if section_type == 'summary_statistics':
            try:
                stats = get_summary_statistics_data(dataset.id)
                if stats.get('comprehensive_table'):
                    metadata.setdefault('report_data', {}).setdefault('tables', []).append({
                        'type': 'html',
                        'html': stats['comprehensive_table'],
                        'title': 'Complete Dataset Summary Table',
                    })
            except Exception:
                pass

        report_section = ReportSection.objects.create(
            document=document,
            order=next_order,
            title=title,
            content=content,
            section_type=section_type,
            metadata=metadata,
        )

        if section_type == 'ai_response' and message_id:
            try:
                chat_message = ChatMessage.objects.get(id=message_id, session=resolved_session)
                chat_message.is_added_to_report = True
                chat_message.report_section = report_section
                chat_message.save()
            except ChatMessage.DoesNotExist:
                pass

        return JsonResponse({'success': True, 'message': 'Added to report', 'section_order': next_order})

    except UserDataset.DoesNotExist:
        return JsonResponse({'error': 'Dataset not found'}, status=404)
    except Exception as e:
        return JsonResponse({'error': f'Error adding to report: {str(e)}'}, status=500)


@login_required
@require_http_methods(["GET"])
def download_report(request):
    dataset_id = request.GET.get('dataset_id')
    session_id = request.GET.get('session_id')
    if not dataset_id or not session_id:
        return JsonResponse({'error': 'dataset_id and session_id are required'}, status=400)
    return generate_report_document(request, dataset_id, session_id)
