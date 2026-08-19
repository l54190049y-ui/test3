# -*- coding: utf-8 -*-
"""招聘数据清洗、持久化存储与指标计算。"""
import json
import os
import re
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.join(BASE_DIR, 'data', '岗位招聘详情.xlsx')
STORE_FILE = os.path.join(BASE_DIR, 'data', 'recruitment_store.json')

# 中文字段 -> 内部英文字段
COL_MAP = {
    '岗位': 'position',
    '部门': 'department',
    '招聘类目': 'category',
    '当前状态': 'status',
    '推送方简历数': 'resumes',
    '用人部门通知邀约': 'invited',
    '面试人数': 'interviewed',
    '通过人数': 'passed',
    'offer人数': 'offer',
    '入职人数': 'onboarded',
    '离职人数': 'left',
    '现存人数': 'current_headcount',
    '招聘需求': 'demand',
    '备注说明': 'remark',
    '招聘起': 'start_date',
    '招聘止': 'end_date',
    '持续时长(天)': 'duration_days',
    '日均简历': 'daily_resumes',
}

DISPLAY_MAP = {v: k for k, v in COL_MAP.items()}  # 英文 -> 中文（展示用）

CORE_COLS = [
    'position', 'department', 'category', 'status',
    'resumes', 'invited', 'interviewed', 'passed', 'offer', 'onboarded', 'left', 'current_headcount',
    'demand', 'remark', 'start_date', 'end_date', 'duration_days', 'daily_resumes',
]

NUMERIC_COLS = ['resumes', 'invited', 'interviewed', 'passed', 'offer', 'onboarded',
                'left', 'current_headcount', 'duration_days', 'daily_resumes']
DATE_COLS = ['start_date', 'end_date']
DERIVED_COLS = ['demand_min', 'demand_max', 'resume_to_interview_rate',
                'interview_to_pass_rate', 'pass_to_offer_rate', 'offer_to_onboard_rate']

STATUS_ORDER = ['招聘中', '完成招聘', '暂停']


def _to_num(value):
    """把 '无推送直约'/'—'/'3个复面ing' 等转成数字，无法识别返回 NaN。"""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return float('nan')
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    if s in ('', '—', '-', '无', '暂无'):
        return float('nan')
    m = re.search(r'-?\d+(\.\d+)?', s)
    if m:
        try:
            return float(m.group(0))
        except ValueError:
            return float('nan')
    return float('nan')


def _parse_demand(value):
    """解析 '2-3' 这类需求人数，返回 (min, max)。"""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return (float('nan'), float('nan'))
    s = str(value).strip()
    nums = [float(x) for x in re.findall(r'\d+(?:\.\d+)?', s)]
    if not nums:
        return (float('nan'), float('nan'))
    if len(nums) == 1:
        return (nums[0], nums[0])
    return (min(nums), max(nums))


def clean_raw(raw_df):
    """原始 Excel/存储 DataFrame -> 清洗后的标准 DataFrame（含派生列）。"""
    df = raw_df.rename(columns=COL_MAP)
    # 删除 Excel 常见的全空 Unnamed 列
    df = df[[c for c in df.columns if not (str(c).startswith('Unnamed') and df[c].isna().all())]]
    for col in CORE_COLS:
        if col not in df.columns:
            df[col] = pd.NA

    df['position'] = df['position'].fillna('未命名岗位').astype(str).str.strip()
    df['department'] = df['department'].fillna('未填写').astype(str)
    df['category'] = df['category'].fillna('未分类').astype(str)
    df['status'] = df['status'].fillna('未知').astype(str)

    for col in NUMERIC_COLS:
        df[col] = df[col].apply(_to_num)

    dm = df['demand'].apply(_parse_demand)
    df['demand_min'] = dm.apply(lambda x: x[0])
    df['demand_max'] = dm.apply(lambda x: x[1])

    for col in DATE_COLS:
        df[col] = pd.to_datetime(df[col], errors='coerce')

    def _rate(num, den):
        r = num / den * 100
        return r.where(den.notna() & (den != 0))

    df['resume_to_interview_rate'] = _rate(df['interviewed'], df['resumes'])
    df['interview_to_pass_rate'] = _rate(df['passed'], df['interviewed'])
    df['pass_to_offer_rate'] = _rate(df['offer'], df['passed'])
    df['offer_to_onboard_rate'] = _rate(df['onboarded'], df['offer'])
    return df.reset_index(drop=True)


def load_current_data():
    """读取当前数据：优先用用户存储的数据，否则用内置示例 Excel。"""
    if os.path.exists(STORE_FILE):
        try:
            with open(STORE_FILE, 'r', encoding='utf-8') as fp:
                records = json.load(fp)
            if records:
                return clean_raw(pd.DataFrame(records))
        except Exception:
            pass
    return clean_raw(pd.read_excel(DEFAULT_DATA, sheet_name=0))


def load_editor_data():
    """编辑器用数据：核心列 + 用户自定义列（不含派生列）。"""
    df = load_current_data()
    keep = CORE_COLS + [c for c in df.columns if c not in CORE_COLS and c not in DERIVED_COLS]
    return df[[c for c in keep if c in df.columns]].copy()


def save_store(df):
    """保存当前数据到本地存储文件（JSON）。"""
    os.makedirs(os.path.dirname(STORE_FILE), exist_ok=True)
    records = df.reset_index(drop=True).to_dict('records')
    with open(STORE_FILE, 'w', encoding='utf-8') as fp:
        json.dump(records, fp, ensure_ascii=False, default=str)


def reset_store():
    """清空存储，回到内置示例数据。"""
    os.makedirs(os.path.dirname(STORE_FILE), exist_ok=True)
    with open(STORE_FILE, 'w', encoding='utf-8') as fp:
        json.dump([], fp)


def merge_dataframes(base_raw, new_raw):
    """合并两批数据（默认保留全部旧数据）。

    以 岗位+部门+招聘起 为匹配键：上传中同键的行覆盖旧值，新键行追加。
    返回 (合并后 DataFrame, 新增行数, 更新行数)。
    """
    b = clean_raw(base_raw)
    n = clean_raw(new_raw)

    def row_key(df):
        k = df[['position', 'department', 'start_date']].copy()
        k['start_date'] = pd.to_datetime(k['start_date'], errors='coerce').dt.strftime('%Y-%m-%d').fillna('')
        return (k['position'] + '|' + k['department'] + '|' + k['start_date']).tolist()

    bk, nk = row_key(b), row_key(n)
    merged = {}
    for key, row in zip(bk, b.to_dict('records')):
        merged.setdefault(key, row)
    added = updated = 0
    for key, row in zip(nk, n.to_dict('records')):
        if key in merged:
            updated += 1
        else:
            added += 1
        merged[key] = row
    return pd.DataFrame(list(merged.values())), added, updated


def to_display(df):
    """英文列名 -> 中文列名（用于编辑器/导出）。"""
    return df.rename(columns=DISPLAY_MAP)


def from_display(df):
    """中文列名 -> 英文列名。"""
    return df.rename(columns={v: k for k, v in DISPLAY_MAP.items()})


def extra_columns(df):
    """返回当前数据里的自定义列（非核心、非派生）。"""
    return [c for c in df.columns if c not in CORE_COLS and c not in DERIVED_COLS]


def status_label(status):
    colors = {'招聘中': '#e07b39', '完成招聘': '#2e9e6b', '暂停': '#8a8f98', '未知': '#c0c4cc'}
    return colors.get(status, '#c0c4cc')


def compute_kpis(df):
    """汇总 KPI。"""
    hiring = int((df['status'] == '招聘中').sum())
    done = int((df['status'] == '完成招聘').sum())
    paused = int((df['status'] == '暂停').sum())
    total_positions = int(len(df))
    total_resumes = _safe_sum(df['resumes'])
    total_interviewed = _safe_sum(df['interviewed'])
    total_passed = _safe_sum(df['passed'])
    total_offer = _safe_sum(df['offer'])
    total_onboarded = _safe_sum(df['onboarded'])
    total_left = _safe_sum(df['left'])
    current = _safe_sum(df['current_headcount'])

    done_df = df[df['status'] == '完成招聘']
    avg_duration = done_df['duration_days'].dropna().mean()
    median_duration = done_df['duration_days'].dropna().median()

    return {
        'total_positions': total_positions,
        'hiring': hiring,
        'done': done,
        'paused': paused,
        'total_resumes': total_resumes,
        'total_invited': _safe_sum(df['invited']),
        'total_interviewed': total_interviewed,
        'total_passed': total_passed,
        'total_offer': total_offer,
        'total_onboarded': total_onboarded,
        'total_left': total_left,
        'current': current,
        'avg_duration': _fmt_num(avg_duration),
        'median_duration': _fmt_num(median_duration),
    }


def _safe_sum(s):
    v = s.dropna().sum()
    return float(v) if not pd.isna(v) else 0.0


def _fmt_num(v):
    if v is None or pd.isna(v):
        return '—'
    return round(float(v), 1)


def funnel_data(df):
    """招聘漏斗各环节总量。邀约环节缺失较多时自动省略。"""
    stages = [
        {'stage': '推送简历', 'value': _safe_sum(df['resumes']), 'color': '#4c8bf5'},
        {'stage': '邀约面试', 'value': _safe_sum(df['invited']), 'color': '#5aa9e6'},
        {'stage': '进入面试', 'value': _safe_sum(df['interviewed']), 'color': '#36b37e'},
        {'stage': '面试通过', 'value': _safe_sum(df['passed']), 'color': '#ffb020'},
        {'stage': '发放 Offer', 'value': _safe_sum(df['offer']), 'color': '#ff7452'},
        {'stage': '入职', 'value': _safe_sum(df['onboarded']), 'color': '#8f6ef0'},
    ]
    if df['invited'].notna().sum() < 5:
        stages = [s for s in stages if s['stage'] != '邀约面试']
    return stages


def department_summary(df):
    """按部门汇总在招/完成岗位数与人数。"""
    out = []
    for dept, g in df.groupby('department'):
        out.append({
            '部门': dept,
            '岗位数': int(len(g)),
            '招聘中': int((g['status'] == '招聘中').sum()),
            '完成招聘': int((g['status'] == '完成招聘').sum()),
            '暂停': int((g['status'] == '暂停').sum()),
            '简历总数': int(_safe_sum(g['resumes'])),
            '面试总数': int(_safe_sum(g['interviewed'])),
            '入职总数': int(_safe_sum(g['onboarded'])),
        })
    return pd.DataFrame(out).sort_values('岗位数', ascending=False).reset_index(drop=True)


def position_duration(df):
    """完成招聘岗位的招聘周期（天）。"""
    d = df[(df['status'] == '完成招聘') & df['duration_days'].notna()].copy()
    d['周期(天)'] = d['duration_days'].round(1)
    return d[['position', 'department', '周期(天)']].sort_values('周期(天)', ascending=True).reset_index(drop=True)


def build_funnel_fig(df):
    """Plotly 漏斗图。"""
    import plotly.graph_objects as go
    data = funnel_data(df)
    fig = go.Figure(go.Funnel(
        y=[d['stage'] for d in data],
        x=[d['value'] for d in data],
        textinfo='value+percent initial',
        marker={'color': [d['color'] for d in data], 'line': {'width': 1, 'color': '#fff'}},
    ))
    fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), height=340,
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      font=dict(family='Microsoft YaHei, PingFang SC, sans-serif'))
    return fig