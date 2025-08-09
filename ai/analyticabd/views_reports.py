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

        # If this is an EDA bundle trigger, create clustered sections (skip duplicates)
        is_eda_bundle = (title or '').strip().lower() in ['eda', 'eda summary', 'eda (exploratory data analysis)']
        if is_eda_bundle:
            try:
                stats = get_summary_statistics_data(dataset.id)
                filename = dataset.name
                created_sections = []

                # Helper to check duplicate by eda_subsection
                def has_section(sub):
                    try:
                        return document.sections.filter(metadata__eda_subsection=sub).exists()
                    except Exception:
                        return False

                # SUMMARY subsection
                if not has_section('summary'):
                    ov = stats.get('dataset_overview', {})
                    parts = [
                        '## Dataset Overview',
                        f"Total Rows: {ov.get('total_rows', 'N/A')} | Total Columns: {ov.get('total_columns', 'N/A')}\n",
                        '## Data Quality Report'
                    ]
                    for var_name, var_stats in (stats.get('data_quality', {}) or {}).items():
                        missing_pct = var_stats.get('missing_percentage', 0)
                        completeness = var_stats.get('completeness', 0)
                        quality_score = var_stats.get('quality_score', 0)
                        parts.append(f"- **{var_name}**: Missing {missing_pct:.1%}, Completeness {completeness:.1%}, Quality Score {quality_score:.1f}")
                    content_summary = '\n'.join(parts)
                    meta_summary = {'eda_subsection': 'summary'}
                    if stats.get('comprehensive_table'):
                        meta_summary['report_data'] = {'tables': [{
                            'type': 'html',
                            'html': stats['comprehensive_table'],
                            'title': 'Complete Dataset Summary Table',
                        }]}
                    ReportSection.objects.create(
                        document=document,
                        order=next_order,
                        title='EDA: Summary Statistics',
                        content=content_summary,
                        section_type='summary_statistics',
                        metadata=meta_summary,
                    )
                    created_sections.append('summary')
                    next_order += 1

                # DISTRIBUTIONS subsection
                if not has_section('distributions'):
                    dist_img = f"/api/analysis/{dataset.id}/distributions.png?top=12&bins=20"
                    # Expanded table with key distribution stats
                    var_html_rows = []
                    var_summary = (stats.get('variable_summary', {}) or {})
                    data_quality = (stats.get('data_quality', {}) or {})
                    # Numeric variables first
                    for name, meta in var_summary.items():
                        vtype = (meta.get('type') or '').lower()
                        if vtype == 'numeric':
                            count = meta.get('count', 'N/A')
                            dq = data_quality.get(name, {})
                            missing_pct = dq.get('missing_percentage', 0.0)
                            mean = meta.get('mean'); std = meta.get('std') if meta.get('std') is not None else meta.get('std_dev')
                            min_v = meta.get('min'); median = meta.get('median'); max_v = meta.get('max')
                            var_html_rows.append(
                                "<tr>"
                                f"<td style='border:1px solid #ddd;padding:6px;'>{name}</td>"
                                f"<td style='border:1px solid #ddd;padding:6px;'>numeric</td>"
                                f"<td style='border:1px solid #ddd;padding:6px;text-align:right;'>{count}</td>"
                                f"<td style='border:1px solid #ddd;padding:6px;text-align:right;'>{missing_pct:.1%}</td>"
                                f"<td style='border:1px solid #ddd;padding:6px;text-align:right;'>{f'{mean:.2f}' if isinstance(mean,(int,float)) else 'N/A'}</td>"
                                f"<td style='border:1px solid #ddd;padding:6px;text-align:right;'>{f'{std:.2f}' if isinstance(std,(int,float)) else 'N/A'}</td>"
                                f"<td style='border:1px solid #ddd;padding:6px;text-align:right;'>{f'{min_v:.2f}' if isinstance(min_v,(int,float)) else 'N/A'}</td>"
                                f"<td style='border:1px solid #ddd;padding:6px;text-align:right;'>{f'{median:.2f}' if isinstance(median,(int,float)) else 'N/A'}</td>"
                                f"<td style='border:1px solid #ddd;padding:6px;text-align:right;'>{f'{max_v:.2f}' if isinstance(max_v,(int,float)) else 'N/A'}</td>"
                                "</tr>"
                            )
                    # Then categorical variables summary
                    for name, meta in var_summary.items():
                        vtype = (meta.get('type') or '').lower()
                        if vtype != 'numeric':
                            count = meta.get('count', 'N/A')
                            dq = data_quality.get(name, {})
                            missing_pct = dq.get('missing_percentage', 0.0)
                            unique = meta.get('unique_count', 'N/A')
                            most_common = meta.get('most_common', 'N/A')
                            most_common_count = meta.get('most_common_count', 'N/A')
                            var_html_rows.append(
                                "<tr>"
                                f"<td style='border:1px solid #ddd;padding:6px;'>{name}</td>"
                                f"<td style='border:1px solid #ddd;padding:6px;'>categorical</td>"
                                f"<td style='border:1px solid #ddd;padding:6px;text-align:right;'>{count}</td>"
                                f"<td style='border:1px solid #ddd;padding:6px;text-align:right;'>{missing_pct:.1%}</td>"
                                f"<td style='border:1px solid #ddd;padding:6px;text-align:right;'>—</td>"
                                f"<td style='border:1px solid #ddd;padding:6px;text-align:right;'>—</td>"
                                f"<td style='border:1px solid #ddd;padding:6px;text-align:right;'>—</td>"
                                f"<td style='border:1px solid #ddd;padding:6px;text-align:right;'>—</td>"
                                f"<td style='border:1px solid #ddd;padding:6px;text-align:left;'>{unique} unique; top: {most_common} ({most_common_count})</td>"
                                "</tr>"
                            )
                    dist_table_html = (
                        "<table style=\"border-collapse:collapse;width:100%;margin:8px 0;\">"
                        "<thead>"
                        "<tr style=\"background:#f8fafc;\">"
                        "<th style=\"border:1px solid #ddd;padding:6px;text-align:left;\">Variable</th>"
                        "<th style=\"border:1px solid #ddd;padding:6px;text-align:left;\">Type</th>"
                        "<th style=\"border:1px solid #ddd;padding:6px;text-align:right;\">Count</th>"
                        "<th style=\"border:1px solid #ddd;padding:6px;text-align:right;\">Missing %</th>"
                        "<th style=\"border:1px solid #ddd;padding:6px;text-align:right;\">Mean</th>"
                        "<th style=\"border:1px solid #ddd;padding:6px;text-align:right;\">Std</th>"
                        "<th style=\"border:1px solid #ddd;padding:6px;text-align:right;\">Min</th>"
                        "<th style=\"border:1px solid #ddd;padding:6px;text-align:right;\">Median</th>"
                        "<th style=\"border:1px solid #ddd;padding:6px;text-align:left;\">Max / Category Summary</th>"
                        "</tr>"
                        "</thead>"
                        "<tbody>" + ''.join(var_html_rows[:24]) + "</tbody></table>"
                    )
                    ReportSection.objects.create(
                        document=document,
                        order=next_order,
                        title=f'EDA: Distribution of Data for {filename}',
                        content='',
                        section_type='distributions',
                        metadata={
                            'eda_subsection': 'distributions',
                            'report_data': {
                                'images': [{'type': 'url', 'src': dist_img, 'title': 'Distributions'}],
                                'tables': [{'type': 'html', 'html': dist_table_html, 'title': 'Distribution Summary'}],
                            }
                        },
                    )
                    created_sections.append('distributions')
                    next_order += 1

                # CORRELATION subsection
                if not has_section('correlation'):
                    corr_img = f"/api/analysis/{dataset.id}/heatmap.png"
                    strong = (stats.get('correlation_matrix', {}) or {}).get('strong_correlations', []) or []
                    corr_rows = []
                    for item in strong:
                        v1 = item.get('variable1'); v2 = item.get('variable2'); r = item.get('correlation')
                        strength = 'Strong' if isinstance(r, (int, float)) and abs(r) >= 0.8 else ('Moderate' if isinstance(r, (int, float)) and abs(r) >= 0.6 else 'Weak')
                        corr_rows.append(
                            "<tr>"
                            f"<td style='border:1px solid #ddd;padding:6px;'>{v1}</td>"
                            f"<td style='border:1px solid #ddd;padding:6px;'>{v2}</td>"
                            f"<td style='border:1px solid #ddd;padding:6px;text-align:right;'>{f'{r:.3f}' if isinstance(r,(int,float)) else 'N/A'}</td>"
                            f"<td style='border:1px solid #ddd;padding:6px;text-align:left;'>{strength}</td>"
                            "</tr>"
                        )
                    corr_table_html = (
                        "<table style=\"border-collapse:collapse;width:100%;margin:8px 0;\">"
                        "<thead>"
                        "<tr style=\"background:#f8fafc;\">"
                        "<th style=\"border:1px solid #ddd;padding:6px;text-align:left;\">Variable 1</th>"
                        "<th style=\"border:1px solid #ddd;padding:6px;text-align:left;\">Variable 2</th>"
                        "<th style=\"border:1px solid #ddd;padding:6px;text-align:right;\">Correlation</th>"
                        "<th style=\"border:1px solid #ddd;padding:6px;text-align:left;\">Strength</th>"
                        "</tr>"
                        "</thead>"
                        "<tbody>" + ''.join(corr_rows[:48]) + "</tbody></table>"
                    )
                    ReportSection.objects.create(
                        document=document,
                        order=next_order,
                        title=f'EDA: Correlation Matrix for {filename}',
                        content='',
                        section_type='correlation',
                        metadata={
                            'eda_subsection': 'correlation',
                            'report_data': {
                                'images': [{'type': 'url', 'src': corr_img, 'title': 'Correlation Heatmap'}],
                                'tables': ([{'type': 'html', 'html': corr_table_html, 'title': 'Strong Correlations'}] if corr_rows else []),
                            }
                        },
                    )
                    created_sections.append('correlation')
                    next_order += 1

                # OUTLIERS subsection
                if not has_section('outliers'):
                    out_img = f"/api/analysis/{dataset.id}/outliers.png?top=12&iqr=1.5"
                    out_info = stats.get('outlier_analysis', {}) or {}
                    out_rows = []
                    for name, meta_o in out_info.items():
                        cnt = meta_o.get('count', 0)
                        pct = meta_o.get('percentage', 0.0)
                        lb = meta_o.get('lower_bound'); ub = meta_o.get('upper_bound')
                        out_rows.append(
                            f"<tr><td style='border:1px solid #ddd;padding:6px;'>{name}</td><td style='border:1px solid #ddd;padding:6px;text-align:right;'>{cnt}</td><td style='border:1px solid #ddd;padding:6px;text-align:right;'>{pct:.1%}</td><td style='border:1px solid #ddd;padding:6px;text-align:right;'>{f'{lb:.2f}' if isinstance(lb,(int,float)) else 'N/A'}</td><td style='border:1px solid #ddd;padding:6px;text-align:right;'>{f'{ub:.2f}' if isinstance(ub,(int,float)) else 'N/A'}</td></tr>"
                        )
                    out_table_html = (
                        "<table style=\"border-collapse:collapse;width:100%;margin:8px 0;\"><thead><tr style=\"background:#f8fafc;\"><th style=\"border:1px solid #ddd;padding:6px;text-align:left;\">Variable</th><th style=\"border:1px solid #ddd;padding:6px;text-align:right;\">Outliers</th><th style=\"border:1px solid #ddd;padding:6px;text-align:right;\">%</th><th style=\"border:1px solid #ddd;padding:6px;text-align:right;\">Lower bound</th><th style=\"border:1px solid #ddd;padding:6px;text-align:right;\">Upper bound</th></tr></thead><tbody>"
                        + ''.join(sorted(out_rows, reverse=True)[:24]) + "</tbody></table>"
                    )
                    ReportSection.objects.create(
                        document=document,
                        order=next_order,
                        title=f'EDA: Outlier Analysis for {filename}',
                        content='',
                        section_type='outliers',
                        metadata={
                            'eda_subsection': 'outliers',
                            'report_data': {
                                'images': [{'type': 'url', 'src': out_img, 'title': 'Outliers (Boxplots)'}],
                                'tables': ([{'type': 'html', 'html': out_table_html, 'title': 'Outlier Summary'}] if out_rows else []),
                            }
                        },
                    )
                    created_sections.append('outliers')
                    next_order += 1

                # Mark UI state (eda_added=true) per dataset
                try:
                    from .session_manager import save_dataset_ui_state
                    save_dataset_ui_state(request.user, dataset.id, {'eda_added': True})
                except Exception:
                    pass

                return JsonResponse({'success': True, 'message': 'EDA added to report', 'sections_created': created_sections})
            except Exception:
                return JsonResponse({'error': 'Error creating EDA sections'}, status=500)

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
