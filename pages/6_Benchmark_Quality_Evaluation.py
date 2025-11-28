"""Benchmark dataset quality evaluation page."""

import re
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app_utils import ensure_page_config, get_benchmark_info, render_sidebar

# Color palette for quality evaluation
QUALITY_COLORS = {
    'high': '#2ecc71',
    'medium': '#f39c12',
    'low': '#e74c3c',
    'poor': '#e74c3c',
    'moderate': '#f39c12',
    'substantial': '#3498db',
    'almost_perfect': '#2ecc71',
    'gpt4o': '#10a37f',
    'gpt5.1': '#74aa9c',
    'claude_sonnet': '#d97757',
    'claude_haiku': '#f4b088',
    'confidence_low': '#e74c3c',
    'confidence_medium': '#f39c12',
    'confidence_high': '#2ecc71',
}


@st.cache_data
def parse_judge_results_quality(filepath: str) -> dict:
    """Parse judge_llm_result.txt into structured dictionary"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    data = {}
    lines = content.split('\n')

    for line in lines:
        if 'Total samples:' in line:
            data['total_samples'] = int(re.search(r'(\d+)', line).group(1))
            break

    kappa_section = {}
    for i, line in enumerate(lines):
        if 'Mean κ:' in line:
            kappa_section['mean'] = float(re.search(r'([\d.]+)', line).group(1))
        elif 'Range:' in line and 'κ' in lines[i-2]:
            match = re.search(r'\[([-\d.]+), ([\d.]+)\]', line)
            kappa_section['range'] = [float(match.group(1)), float(match.group(2))]

    agreement_dist = {}
    agreement_pct = {}
    for line in lines:
        match = re.match(r'\s+(poor|moderate|substantial|almost_perfect):\s+(\d+)\s+\(([\d.]+)%\)', line)
        if match:
            level = match.group(1)
            count = int(match.group(2))
            pct = float(match.group(3))
            agreement_dist[level] = count
            agreement_pct[level] = pct

    kappa_section['distribution'] = agreement_dist
    kappa_section['percentages'] = agreement_pct
    data['kappa'] = kappa_section

    conf_section = {}
    for i, line in enumerate(lines):
        if 'Mean confidence:' in line:
            conf_section['mean'] = float(re.search(r'([\d.]+)', line).group(1))
        elif 'Range:' in line and 'confidence' in lines[i-2].lower():
            match = re.search(r'\[([\d.]+), ([\d.]+)\]', line)
            conf_section['range'] = [float(match.group(1)), float(match.group(2))]

    conf_dist = {}
    conf_pct = {}
    for line in lines:
        match = re.match(r'\s+(low|medium|high):\s+(\d+)\s+\(([\d.]+)%\)', line)
        if match:
            level = match.group(1)
            count = int(match.group(2))
            pct = float(match.group(3))
            conf_dist[level] = count
            conf_pct[level] = pct

    conf_section['distribution'] = conf_dist
    conf_section['percentages'] = conf_pct
    data['confidence'] = conf_section

    quality_tiers = {}
    for line in lines:
        if 'High quality:' in line:
            match = re.search(r'(\d+)\s+\(([\d.]+)%\)', line)
            if match:
                quality_tiers['high'] = (int(match.group(1)), float(match.group(2)))
        elif 'Medium quality:' in line:
            match = re.search(r'(\d+)\s+\(([\d.]+)%\)', line)
            if match:
                quality_tiers['medium'] = (int(match.group(1)), float(match.group(2)))
        elif 'Low quality:' in line:
            match = re.search(r'(\d+)\s+\(([\d.]+)%\)', line)
            if match:
                quality_tiers['low'] = (int(match.group(1)), float(match.group(2)))
        elif 'Needs review:' in line:
            match = re.search(r'(\d+)\s+\(([\d.]+)%\)', line)
            if match:
                quality_tiers['needs_review'] = (int(match.group(1)), float(match.group(2)))

    data['quality_tiers'] = quality_tiers

    consensus = {}
    for line in lines:
        if 'Overall average:' in line:
            match = re.search(r'([\d.]+)\s+\[([\d.]+),\s+([\d.]+)\]', line)
            if match:
                consensus['overall'] = {
                    'avg': float(match.group(1)),
                    'range': [float(match.group(2)), float(match.group(3))]
                }
        elif 'Factual accuracy:' in line:
            match = re.search(r'([\d.]+)\s+\[([\d.]+),\s+([\d.]+)\]', line)
            if match:
                consensus['factual_accuracy'] = {
                    'avg': float(match.group(1)),
                    'range': [float(match.group(2)), float(match.group(3))]
                }
        elif 'Groundedness:' in line:
            match = re.search(r'([\d.]+)\s+\[([\d.]+),\s+([\d.]+)\]', line)
            if match:
                consensus['groundedness'] = {
                    'avg': float(match.group(1)),
                    'range': [float(match.group(2)), float(match.group(3))]
                }
        elif 'Consistency:' in line:
            match = re.search(r'([\d.]+)\s+\[([\d.]+),\s+([\d.]+)\]', line)
            if match:
                consensus['consistency'] = {
                    'avg': float(match.group(1)),
                    'range': [float(match.group(2)), float(match.group(3))]
                }
        elif 'Completeness:' in line:
            match = re.search(r'([\d.]+)\s+\[([\d.]+),\s+([\d.]+)\]', line)
            if match:
                consensus['completeness'] = {
                    'avg': float(match.group(1)),
                    'range': [float(match.group(2)), float(match.group(3))]
                }

    data['consensus_scores'] = consensus

    model_bias = {}
    current_model = None
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('gpt4o:'):
            current_model = 'gpt4o'
            model_bias[current_model] = {}
        elif stripped.startswith('gpt5.1:'):
            current_model = 'gpt5.1'
            model_bias[current_model] = {}
        elif stripped.startswith('claude_sonnet:'):
            current_model = 'claude_sonnet'
            model_bias[current_model] = {}
        elif stripped.startswith('claude_haiku:'):
            current_model = 'claude_haiku'
            model_bias[current_model] = {}
        elif current_model and 'Average:' in line:
            match = re.search(r'([\d.]+)\s+\[([\d.]+),\s+([\d.]+)\]', line)
            if match:
                model_bias[current_model]['average'] = {
                    'mean': float(match.group(1)),
                    'range': [float(match.group(2)), float(match.group(3))]
                }
        elif current_model and 'Factual accuracy:' in line:
            match = re.search(r'([\d.]+)', line)
            if match:
                model_bias[current_model]['factual_accuracy'] = float(match.group(1))
        elif current_model and 'Groundedness:' in line:
            match = re.search(r'([\d.]+)', line)
            if match:
                model_bias[current_model]['groundedness'] = float(match.group(1))
        elif current_model and 'Consistency:' in line:
            match = re.search(r'([\d.]+)', line)
            if match:
                model_bias[current_model]['consistency'] = float(match.group(1))
        elif current_model and 'Completeness:' in line:
            match = re.search(r'([\d.]+)', line)
            if match:
                model_bias[current_model]['completeness'] = float(match.group(1))

    data['model_bias'] = model_bias

    task_quality = {}
    current_task = None
    for line in lines:
        stripped = line.strip()
        if stripped and stripped.endswith(':') and stripped[:-1] in {'coding', 'dialogue', 'general', 'math', 'qa', 'summarization'}:
            current_task = stripped[:-1]
            task_quality[current_task] = {}
        elif current_task:
            if 'Count:' in line:
                match = re.search(r'(\d+)', line)
                if match:
                    task_quality[current_task]['count'] = int(match.group(1))
            elif 'Mean consensus:' in line:
                match = re.search(r'([\d.]+)', line)
                if match:
                    task_quality[current_task]['consensus'] = float(match.group(1))
            elif 'Mean κ:' in line:
                match = re.search(r'([\d.]+)', line)
                if match:
                    task_quality[current_task]['kappa'] = float(match.group(1))
            elif 'Mean confidence:' in line:
                match = re.search(r'([\d.]+)', line)
                if match:
                    task_quality[current_task]['confidence'] = float(match.group(1))
            elif 'High quality:' in line:
                match = re.search(r'(\d+)\s+\(([\d.]+)%\)', line)
                if match:
                    task_quality[current_task]['high_quality_pct'] = float(match.group(2))
            elif 'Needs review:' in line:
                match = re.search(r'(\d+)\s+\(([\d.]+)%\)', line)
                if match:
                    task_quality[current_task]['needs_review_pct'] = float(match.group(2))

    data['task_quality'] = task_quality

    return data


def create_agreement_chart_quality(data: dict) -> go.Figure:
    """Create inter-rater agreement distribution chart"""
    kappa_data = data['kappa']
    levels = ['poor', 'moderate', 'substantial', 'almost_perfect']
    colors_map = {
        'poor': QUALITY_COLORS['poor'],
        'moderate': QUALITY_COLORS['moderate'],
        'substantial': QUALITY_COLORS['substantial'],
        'almost_perfect': QUALITY_COLORS['almost_perfect']
    }

    fig = go.Figure()
    for level in levels:
        pct = kappa_data['percentages'].get(level, 0)
        count = kappa_data['distribution'].get(level, 0)
        fig.add_trace(go.Bar(
            name=level.replace('_', ' ').title(),
            x=[pct],
            y=['Distribution'],
            orientation='h',
            marker_color=colors_map[level],
            text=f'{pct:.1f}% ({count})',
            textposition='inside',
            hovertemplate=f'{level.replace("_", " ").title()}: {pct:.1f}% ({count} samples)<extra></extra>'
        ))

    fig.update_layout(
        barmode='stack',
        title='Inter-Rater Agreement',
        xaxis_title='Percentage',
        yaxis_title='',
        height=300,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=100, b=20)
    )
    return fig


def create_confidence_chart_quality(data: dict) -> go.Figure:
    """Create collective confidence distribution chart"""
    conf_data = data['confidence']
    levels = ['low', 'medium', 'high']
    colors_map = {
        'low': QUALITY_COLORS['confidence_low'],
        'medium': QUALITY_COLORS['confidence_medium'],
        'high': QUALITY_COLORS['confidence_high']
    }

    fig = go.Figure()
    for level in levels:
        pct = conf_data['percentages'].get(level, 0)
        count = conf_data['distribution'].get(level, 0)
        fig.add_trace(go.Bar(
            name=level.title(),
            x=[pct],
            y=['Distribution'],
            orientation='h',
            marker_color=colors_map[level],
            text=f'{pct:.1f}% ({count})',
            textposition='inside',
            hovertemplate=f'{level.title()}: {pct:.1f}% ({count} samples)<extra></extra>'
        ))

    fig.update_layout(
        barmode='stack',
        title='Collective Confidence Distribution',
        xaxis_title='Percentage',
        yaxis_title='',
        height=300,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=100, b=20)
    )
    return fig


def create_quality_pie_quality(data: dict) -> go.Figure:
    """Create quality tier distribution pie chart"""
    quality_data = data['quality_tiers']
    labels = ['High Quality', 'Medium Quality', 'Low Quality']
    values = [quality_data['high'][1], quality_data['medium'][1], quality_data['low'][1]]
    colors = [QUALITY_COLORS['high'], QUALITY_COLORS['medium'], QUALITY_COLORS['low']]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        marker_colors=colors,
        hole=0.4,
        textinfo='label+percent',
        hovertemplate='%{label}: %{value:.1f}%<br>(%{text} samples)<extra></extra>',
        text=[quality_data['high'][0], quality_data['medium'][0], quality_data['low'][0]]
    )])

    fig.update_layout(
        title='Quality Tier Distribution',
        annotations=[dict(text=f'{data["total_samples"]}<br>samples', x=0.5, y=0.5, font_size=16, showarrow=False)],
        height=400,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    return fig


def create_consensus_chart_quality(data: dict) -> go.Figure:
    """Create consensus scores bar chart with ranges"""
    consensus = data['consensus_scores']
    dimensions = ['Overall', 'Factual Accuracy', 'Groundedness', 'Consistency', 'Completeness']
    scores = []
    error_minus = []
    error_plus = []

    dim_keys = ['overall', 'factual_accuracy', 'groundedness', 'consistency', 'completeness']
    for key in dim_keys:
        if key in consensus:
            avg = consensus[key]['avg']
            min_val = consensus[key]['range'][0]
            max_val = consensus[key]['range'][1]
            scores.append(avg)
            error_minus.append(avg - min_val)
            error_plus.append(max_val - avg)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=dimensions,
        x=scores,
        orientation='h',
        marker_color='#3498db',
        text=[f'{s:.3f}' for s in scores],
        textposition='outside',
        error_x=dict(
            type='data',
            symmetric=False,
            array=error_plus,
            arrayminus=error_minus,
            color='gray'
        ),
        hovertemplate='%{y}: %{x:.3f}<extra></extra>'
    ))

    fig.add_vline(x=4.0, line_dash="dash", line_color="green", annotation_text="4.0 threshold")
    fig.update_layout(
        title='Consensus Scores (with Min-Max Range)',
        xaxis_title='Score (1-5)',
        xaxis=dict(range=[0, 5.5]),
        height=400,
        margin=dict(l=150, r=20, t=60, b=20)
    )
    return fig


def create_model_bias_chart_quality(data: dict) -> go.Figure:
    """Create model bias comparison chart"""
    model_bias = data['model_bias']
    models = ['gpt4o', 'gpt5.1', 'claude_sonnet', 'claude_haiku']
    dimensions = ['average', 'factual_accuracy', 'groundedness', 'consistency', 'completeness']
    dim_labels = ['Average', 'Factual', 'Groundedness', 'Consistency', 'Completeness']

    fig = go.Figure()
    for model in models:
        if model in model_bias:
            values = []
            for dim in dimensions:
                if dim == 'average':
                    values.append(model_bias[model]['average']['mean'])
                else:
                    values.append(model_bias[model].get(dim, 0))

            fig.add_trace(go.Bar(
                name=model.replace('_', ' ').title(),
                x=dim_labels,
                y=values,
                marker_color=QUALITY_COLORS.get(model, '#95a5a6'),
                text=[f'{v:.2f}' for v in values],
                textposition='outside'
            ))

    fig.update_layout(
        title='Model Bias Comparison',
        xaxis_title='Dimension',
        yaxis_title='Score',
        yaxis=dict(range=[0, 5.5]),
        barmode='group',
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=80, b=20)
    )
    return fig


def create_task_table_quality(data: dict) -> pd.DataFrame:
    """Create styled task quality comparison table"""
    task_quality = data['task_quality']
    rows = []
    for task, metrics in task_quality.items():
        rows.append({
            'Task': task.capitalize(),
            'Count': metrics.get('count', 0),
            'Consensus': f"{metrics.get('consensus', 0):.3f}",
            'Kappa (κ)': f"{metrics.get('kappa', 0):.3f}",
            'Confidence': f"{metrics.get('confidence', 0):.3f}",
            'High Quality %': f"{metrics.get('high_quality_pct', 0):.1f}%",
            'Needs Review %': f"{metrics.get('needs_review_pct', 0):.1f}%"
        })
    return pd.DataFrame(rows)


def render_benchmark_quality_page() -> None:
    st.title("📊 Benchmark Dataset Quality Evaluation")
    st.markdown("### Cross-Model Validation by 4 Judge LLMs")
    st.markdown("---")

    project_root = Path(__file__).resolve().parents[1]
    data_file = project_root / "benchmark" / "judge_llm_result.txt"
    if not data_file.exists():
        st.error(f"❌ Data file not found: {data_file}")
        st.info("Please ensure benchmark/judge_llm_result.txt exists")
        return

    data = parse_judge_results_quality(str(data_file))

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(label="📈 Total Samples", value=data['total_samples'])

    with col2:
        st.metric(
            label="🎯 Mean Kappa (κ)",
            value=f"{data['kappa']['mean']:.3f}",
            delta="Moderate Agreement"
        )

    with col3:
        st.metric(
            label="✅ Mean Confidence",
            value=f"{data['confidence']['mean']:.3f}",
            delta="High"
        )

    with col4:
        high_pct = data['quality_tiers']['high'][1]
        st.metric(
            label="🏆 High Quality",
            value=f"{high_pct:.1f}%",
            delta=f"{data['quality_tiers']['high'][0]} samples"
        )

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        fig1 = create_agreement_chart_quality(data)
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        fig2 = create_confidence_chart_quality(data)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        with st.expander("ℹ️ Quality Tier 계산 방법", expanded=False):
            st.markdown("""
**4가지 지표를 종합 평가:**
- Overall Average ≥ 3.5
- Inter-Rater Aggrement  ≥ 0.5
- Collective Confidence ≥ 0.75
- Factual Accuracy ≥ 3.5

→ **High** / **Medium** / **Low** 3단계 분류
            """)
        fig3 = create_quality_pie_quality(data)
        st.plotly_chart(fig3, use_container_width=True)

    with col2:
        with st.expander("ℹ️ Consensus Scores 계산 방법", expanded=False):
            st.markdown("""
**4개 Judge LLM 점수의 단순 평균:**

각 차원별 평균 계산:
- Factual Accuracy
- Groundedness
- Consistency
- Completeness

→ 4개 차원 평균의 평균 = **Overall**
            """)
        fig4 = create_consensus_chart_quality(data)
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("---")

    st.markdown("### Model Bias Comparison")
    with st.expander("ℹ️ Model Bias 분석 방법", expanded=False):
        st.markdown("""
**각 모델의 평균 점수 비교:**

4개 Judge LLM이 244개 샘플에 대해 매긴 점수의 평균을 비교하여:
- 어느 모델이 더 엄격하게 평가하는지 (낮은 점수)
- 어느 모델이 더 관대하게 평가하는지 (높은 점수)
- 모델 간 점수 차이가 큰 차원은 무엇인지

를 파악합니다.

**의미**: 각 Judge LLM의 평가 성향 차이
        """)
    fig5 = create_model_bias_chart_quality(data)
    st.plotly_chart(fig5, use_container_width=True)

    st.markdown("---")

    st.markdown("### Quality by Task Type")
    df = create_task_table_quality(data)
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("---")

    with st.expander("🔍 주요 발견 사항", expanded=False):
        st.markdown("**평가자 간 일치도:**")
        st.markdown(f"- 평균 Inter-Rater Aggrement: **{data['kappa']['mean']:.3f}** (중간 수준 일치도)")
        st.markdown(f"- {data['kappa']['percentages'].get('almost_perfect', 0):.1f}%의 샘플이 거의 완벽한 일치도를 보임 (κ ≥ 0.8)")
        st.markdown(f"- {data['kappa']['percentages'].get('poor', 0):.1f}%의 샘플이 낮은 일치도를 보임 (κ < 0.4)")

        st.markdown("\n**신뢰도:**")
        st.markdown(f"- 평균 집단 신뢰도: **{data['confidence']['mean']:.3f}**")
        st.markdown(f"- {data['confidence']['percentages'].get('high', 0):.1f}%의 샘플이 높은 신뢰도를 가짐 (≥ 0.85)")
        st.markdown(f"- {data['confidence']['percentages'].get('low', 0):.1f}%의 샘플이 낮은 신뢰도를 가짐 (< 0.7)")

        st.markdown("\n**품질 등급:**")
        st.markdown(f"- {data['quality_tiers']['high'][1]:.1f}%의 샘플이 고품질 ({data['quality_tiers']['high'][0]}개 샘플)")
        st.markdown(f"- {data['quality_tiers']['needs_review'][1]:.1f}%의 샘플이 재검토 필요 ({data['quality_tiers']['needs_review'][0]}개 샘플)")

        st.markdown("\n**태스크별 인사이트:**")
        task_quality = data['task_quality']
        best_task = max(task_quality.items(), key=lambda x: x[1].get('kappa', 0))
        worst_task = min(task_quality.items(), key=lambda x: x[1].get('kappa', 0))
        st.markdown(f"- 가장 높은 일치도: **{best_task[0].capitalize()}** (κ = {best_task[1]['kappa']:.3f})")
        st.markdown(f"- 가장 어려운 태스크: **{worst_task[0].capitalize()}** (κ = {worst_task[1]['kappa']:.3f}, 재검토 필요 {worst_task[1].get('needs_review_pct', 0):.1f}%)")

    with st.expander("📐 지표 계산 방법", expanded=False):
        st.markdown("### Consensus Scores 계산")
        st.markdown("""
**계산 방식**: 4개 Judge LLM(GPT-4o, GPT-5.1, Claude Sonnet, Claude Haiku)이 매긴 점수의 단순 평균

**단계**:
1. 각 차원(Factual Accuracy, Groundedness, Consistency, Completeness)별로 4개 모델의 점수를 평균
2. 4개 차원 평균의 평균을 Overall Average로 계산

**예시**:
```
Factual Accuracy: (4.8 + 4.5 + 4.3 + 4.6) / 4 = 4.55
Groundedness: (4.8 + 4.5 + 4.5 + 4.5) / 4 = 4.56
Consistency: (5.0 + 4.9 + 4.6 + 4.9) / 4 = 4.83
Completeness: (4.8 + 4.4 + 4.2 + 4.5) / 4 = 4.44
→ Overall: (4.55 + 4.56 + 4.83 + 4.44) / 4 = 4.596
```

**의미**: 모델들이 이 샘플을 평균적으로 몇 점이라고 평가했는가?
        """)

        st.markdown("---")

        st.markdown("### Quality Tier 분류")
        st.markdown("""
**계산 방식**: 4가지 지표를 종합적으로 고려하여 High/Medium/Low로 분류

**분류 기준**:

**🟢 High Quality**:
- Overall Average ≥ 3.5 (전반적으로 준수한 품질)
- Inter-Rater Aggrement ≥ 0.5 (moderate~substantial 수준 합의)
- Collective Confidence ≥ 0.75 (모델 간 의견이 꽤 일관됨)
- Factual Accuracy ≥ 3.5 (사실성도 충분히 높음)

**🟡 Medium Quality**:
- Overall Average ≥ 3.0
- Inter-Rater Aggrement ≥ 0.3
- Collective Confidence ≥ 0.6

**🔴 Low Quality**:
- 위 조건을 만족하지 못함

**예시**:
```
샘플 A: Overall 4.5, κ=0.7, Conf=0.85, Factual=4.6 → High ✅
샘플 B: Overall 3.2, κ=0.35, Conf=0.65, Factual=3.1 → Medium ⚠️
샘플 C: Overall 2.8, κ=0.2, Conf=0.5, Factual=2.5 → Low ❌
```

**의미**: 이 샘플이 벤치마크로 사용하기에 얼마나 신뢰할 만한가?
        """)

        st.markdown("---")

        st.markdown("### Fleiss' Kappa (Inter-Rater Agreement)")
        st.markdown("""
**계산 방식**: Fleiss' Kappa를 사용하여 4개 모델의 일치도 측정

**공식**:
```
κ = (P_o - P_e) / (1 - P_e)

P_o: 관찰된 일치도
P_e: 우연에 의한 기대 일치도 (1/5, 1-5 점수 범위)
```

**예시**:
```
점수 [4, 4, 4, 3]
→ 4점에 3명 일치, 3점에 1명
→ P_o = (3×2 + 1×0) / (4×3) = 0.5
→ κ = (0.5 - 0.2) / (1 - 0.2) = 0.375
```

**해석 기준** (Landis & Koch):
- ≥ 0.8: Almost Perfect
- 0.6-0.8: Substantial
- 0.4-0.6: Moderate
- < 0.4: Poor

**의미**: 모델들이 같은 점수를 얼마나 자주 매기는가?
        """)

        st.markdown("---")

        st.markdown("### Collective Confidence")
        st.markdown("""
**계산 방식**: 모델 점수의 표준편차를 기반으로 합의 강도 측정

**공식**:
```
Confidence = 1 - (std_dev / max_std_dev)

std_dev: 모델 점수들의 표준편차
max_std_dev: 최대 표준편차 (2.0, 1-5 점수 범위)
```

**예시**:
```
점수 [4, 4, 4, 4] → std=0.0 → confidence=1.0 (완전 합의) ✅
점수 [4, 4, 4, 3] → std=0.43 → confidence=0.78 (높은 합의)
점수 [5, 4, 3, 2] → std=1.12 → confidence=0.44 (낮은 합의) ⚠️
```

**해석 기준**:
- ≥ 0.85: High
- 0.7-0.85: Medium
- < 0.7: Low

**의미**: 모델들의 점수가 비슷한 범위에 있는가?
        """)


def main() -> None:
    ensure_page_config()
    benchmark_info = get_benchmark_info()
    render_sidebar(benchmark_info)
    render_benchmark_quality_page()


if __name__ == "__main__":
    main()
