# -*- coding: utf-8 -*-
"""招聘 HR 工作台：数据看板（含数据管理）/ 岗位匹配（岗位信息→批量上传→DeepSeek 智能打分）。"""
import datetime as _dt
import hashlib
import io
import json
import os
import urllib.parse
import urllib.request

import pandas as pd
import streamlit as st
import plotly.express as px

import data_utils as du
import matching as mt

st.set_page_config(page_title='招聘 HR 工作台', page_icon='📊', layout='wide')

st.markdown("""
<style>
.block-container {padding-top: 1.2rem;}
[data-testid="stMetric"] {
    background: #f7f9fc; border: 1px solid #e5e9f0; border-radius: 10px;
    padding: 12px 16px;
}
div[data-testid="stMetricLabel"] {font-size: 0.9rem; color: #5b6472;}
div[data-testid="stMetricValue"] {font-size: 1.6rem;}
</style>
""", unsafe_allow_html=True)

EDU_ORDER = ['不限', '大专', '本科', '硕士', '博士']
EDU_LEVEL = {'不限': 0, '大专': 4, '本科': 6, '硕士': 7, '博士': 8}
EDU_NAME = {v: k for k, v in EDU_LEVEL.items()}

POS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'user_positions.json')


def kpi_row(kpis, keys, labels, suffix=''):
    cols = st.columns(len(keys))
    for col, k in zip(cols, keys):
        v = kpis[k]
        col.metric(labels[k], f'{v:g}{suffix}' if isinstance(v, (int, float)) else v)


# ---------------- 持久化存储层（云端数据库优先，本地文件兜底） ----------------
# 为什么需要这一层：Streamlit 云端服务器的磁盘是临时的，应用休眠/重启后写在磁盘上的文件
# 会被清空，而刷新页面又会开启全新会话（session_state 归零）。所以岗位库（含 JD）、被删除
# 的内置岗位、打分记录统一存到外部数据库；没配置时自动退化为本地文件（本地运行完全够用）。
APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(APP_DIR, 'data')
KEY_FILE = os.path.join(DATA_DIR, 'deepseek_key.txt')
LOCAL_KV_FILE = os.path.join(DATA_DIR, 'kv_store.json')

KV_TABLE = 'hr_workbench_kv'
POS_KEY = 'positions_v1'
DEL_KEY = 'deleted_builtins_v1'
HIST_KEY = 'score_history_v1'
HIST_MAX = 20
KV_WARN = []

SUPABASE_SQL = """create table if not exists hr_workbench_kv (
  key text primary key,
  value jsonb,
  updated_at timestamptz default now()
);"""


def _secret(*names):
    """读取配置：环境变量优先，其次 st.secrets（支持扁平键与 [supabase] 分区）。"""
    for n in names:
        v = os.environ.get(n)
        if v:
            return str(v).strip()
    try:
        sec = st.secrets
    except Exception:
        return ''
    for n in names:
        try:
            v = sec.get(n)
        except Exception:
            v = None
        if v:
            return str(v).strip()
    try:
        for section in ('supabase', 'SUPABASE', 'Supabase'):
            if section in sec:
                sub = sec[section]
                for n in names:
                    try:
                        v = sub.get(n)
                    except Exception:
                        v = None
                    if v:
                        return str(v).strip()
    except Exception:
        pass
    return ''


def supabase_conf():
    """返回（项目地址, 密钥）；未配置时返回空字符串。"""
    url = _secret('SUPABASE_URL')
    key = _secret('SUPABASE_KEY', 'SUPABASE_ANON_KEY', 'SUPABASE_SERVICE_KEY')
    return url.rstrip('/'), key


def cloud_enabled():
    url, key = supabase_conf()
    return bool(url and key)


def _now_str():
    return _dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def _jsonable(obj):
    """把 numpy / pandas 等类型转成可 JSON 序列化的普通类型。"""
    if obj is None or isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, int):
        return int(obj)
    if isinstance(obj, float):
        return None if obj != obj or obj in (float('inf'), float('-inf')) else float(obj)
    if hasattr(obj, 'item'):
        try:
            return _jsonable(obj.item())
        except Exception:
            pass
    return str(obj)


def _local_read_all():
    try:
        with open(LOCAL_KV_FILE, 'r', encoding='utf-8') as fp:
            data = json.load(fp)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _local_write_all(data):
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(LOCAL_KV_FILE, 'w', encoding='utf-8') as fp:
            json.dump(data, fp, ensure_ascii=False, indent=1)
    except Exception:
        pass


def _http_json(method, url, headers, payload=None, timeout=20):
    body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode('utf-8')
    req = urllib.request.Request(url, data=body, method=method, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode('utf-8')
    return json.loads(raw) if raw.strip() else None


def _sb_headers(key):
    return {'apikey': key, 'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'}


def kv_get(key, default=None):
    """读持久层：优先云端数据库，取不到再退回本地文件。"""
    if cloud_enabled():
        url, skey = supabase_conf()
        try:
            query = urllib.parse.urlencode({'key': f'eq.{key}', 'select': 'value'})
            rows = _http_json('GET', f'{url}/rest/v1/{KV_TABLE}?{query}', _sb_headers(skey)) or []
            if rows:
                return rows[0].get('value', default)
        except Exception as e:
            _kv_warn(f'云端读取失败（{e}），已改用本地备份。')
    return _local_read_all().get(key, default)


def kv_set(key, value):
    """写持久层：云端可用时写云端，同时本地留一份兜底。"""
    value = _jsonable(value)
    wrote_cloud = False
    if cloud_enabled():
        url, skey = supabase_conf()
        headers = _sb_headers(skey)
        headers['Prefer'] = 'resolution=merge-duplicates,return=minimal'
        try:
            _http_json('POST', f'{url}/rest/v1/{KV_TABLE}', headers,
                       [{'key': key, 'value': value, 'updated_at': _now_str()}])
            wrote_cloud = True
        except Exception as e:
            _kv_warn(f'云端保存失败（{e}），已暂存到应用本地文件。')
    store = _local_read_all()
    store[key] = value
    _local_write_all(store)
    return wrote_cloud


def _kv_warn(msg):
    """记录持久化告警，跨 rerun 也能提示到用户。"""
    KV_WARN.append(msg)
    try:
        errs = list(st.session_state.get('kv_errors', []))
        if msg not in errs:
            errs.append(msg)
        st.session_state['kv_errors'] = errs[-3:]
    except Exception:
        pass


def storage_status():
    if cloud_enabled():
        return True, '🟢 云端数据库已连接：岗位库、JD 与打分记录会永久保存。'
    return False, ('🟡 当前为临时存储：刷新页面不会丢，但应用休眠/重启后可能清空；'
                   '配置好 SUPABASE_URL / SUPABASE_KEY 即可永久保存。')


def _storage_notice():
    ok, msg = storage_status()
    if ok:
        st.caption(msg)
    else:
        st.info(msg + '（展开页面底部的「如何开启永久保存」可查看配置步骤）')
    for w in st.session_state.pop('kv_errors', [])[:2]:
        st.warning(w)


# ---------------- 岗位库（内置 + 用户自定义，均可删除） ----------------

def _norm_pos(p, source=None):
    q = dict(p)
    q['name'] = str(q.get('name') or '未命名岗位').strip()
    q['department'] = str(q.get('department') or '')
    try:
        q['education'] = int(q.get('education') or 0)
    except Exception:
        q['education'] = 0
    try:
        q['years'] = float(q.get('years') or 0)
    except Exception:
        q['years'] = 0.0
    q['keywords'] = list(q.get('keywords') or [])
    q['hard_conditions'] = list(q.get('hard_conditions') or [])
    q['description'] = q.get('description') or ''
    q['source'] = source or q.get('source') or '自定义'
    return q


def _builtin_positions():
    return [_norm_pos(p, '内置') for p in mt.JD_POSITIONS]


def _legacy_custom_positions():
    """兼容旧版本：把 data/user_positions.json 里的自定义岗位迁移进持久层。"""
    try:
        if os.path.exists(POS_FILE):
            with open(POS_FILE, 'r', encoding='utf-8') as fp:
                return [_norm_pos(p, '自定义') for p in json.load(fp)]
    except Exception:
        pass
    return []


def load_positions():
    """岗位库 = 内置岗位（可删、可恢复）+ 自定义岗位，全部来自持久层。"""
    deleted_list = sorted(set(kv_get(DEL_KEY, []) or []))
    st.session_state['deleted_builtins'] = deleted_list
    deleted = set(deleted_list)
    stored = kv_get(POS_KEY, None)
    need_save = False
    if isinstance(stored, list) and stored:
        positions = [_norm_pos(p) for p in stored]
    else:
        positions = _builtin_positions()
        need_save = True
    have = {p['name'] for p in positions}
    for p in _legacy_custom_positions():
        if p['name'] not in have:
            positions.append(p)
            have.add(p['name'])
            need_save = True
    for p in _builtin_positions():          # 代码里新增的内置岗位自动补齐
        if p['name'] not in have and p['name'] not in deleted:
            positions.append(p)
            have.add(p['name'])
            need_save = True
    out = [p for p in positions if not (p['source'] == '内置' and p['name'] in deleted)]
    if len(out) != len(positions):
        need_save = True
    if need_save:
        kv_set(POS_KEY, out)
    return out


def get_positions():
    if 'positions' not in st.session_state:
        st.session_state['positions'] = load_positions()
    return st.session_state['positions']


def deleted_builtins():
    return sorted(set(st.session_state.get('deleted_builtins', [])))


def save_positions(positions, deleted=None):
    kv_set(POS_KEY, [_norm_pos(p) for p in positions])
    if deleted is not None:
        st.session_state['deleted_builtins'] = sorted(set(deleted))
        kv_set(DEL_KEY, sorted(set(deleted)))


def load_history(force=False):
    if force or 'score_history' not in st.session_state:
        hist = kv_get(HIST_KEY, [])
        st.session_state['score_history'] = hist if isinstance(hist, list) else []
    return st.session_state['score_history']


def save_history(hist):
    hist = list(hist)[:HIST_MAX]
    st.session_state['score_history'] = hist
    kv_set(HIST_KEY, hist)


def backup_payload():
    return _jsonable({
        'version': 1,
        'exported_at': _now_str(),
        'positions': get_positions(),
        'deleted_builtins': deleted_builtins(),
        'score_history': load_history(),
    })


def restore_payload(data):
    """从备份 JSON 恢复岗位库与打分记录，返回（岗位数, 记录数）。"""
    positions = [_norm_pos(p) for p in (data.get('positions') or [])]
    deleted = [str(x) for x in (data.get('deleted_builtins') or [])]
    hist = data.get('score_history') or []
    save_positions(positions, deleted)
    save_history(hist)
    st.session_state.pop('positions', None)
    st.session_state['positions'] = load_positions()
    return len(st.session_state['positions']), len(hist)


def position_options(positions):
    return [mt.position_label(p) for p in positions]


def find_position(positions, label):
    for p in positions:
        if mt.position_label(p) == label:
            return p
    return positions[0]


def default_api_key():
    try:
        return st.secrets.get('DEEPSEEK_API_KEY', '')
    except Exception:
        return ''


def _saved_api_key():
    try:
        if os.path.exists(KEY_FILE):
            with open(KEY_FILE, 'r', encoding='utf-8') as fp:
                return fp.read().strip()
    except Exception:
        pass
    return ''


def _save_api_key(key):
    try:
        os.makedirs(os.path.dirname(KEY_FILE), exist_ok=True)
        with open(KEY_FILE, 'w', encoding='utf-8') as fp:
            fp.write(key.strip())
    except Exception:
        pass


# ==================== 数据看板 ====================

def _editor_column_config(editor_df):
    """按列类型生成 data_editor 的列配置。"""
    cfg = {}
    for en in editor_df.columns:
        cn = du.DISPLAY_MAP.get(en, en)
        if en in du.NUMERIC_COLS:
            cfg[cn] = st.column_config.NumberColumn(cn, format='%.0f')
        elif en in du.DATE_COLS:
            cfg[cn] = st.column_config.DateColumn(cn)
        elif en == 'status':
            cfg[cn] = st.column_config.SelectboxColumn(cn, options=du.STATUS_ORDER)
        elif en == 'demand':
            cfg[cn] = st.column_config.TextColumn(cn, help='如：2-3')
    return cfg


def _fix_editor_types(df):
    """把编辑器返回的数据修正类型。"""
    df = df.copy()
    if 'position' in df.columns:
        df['position'] = df['position'].fillna('未命名岗位').astype(str).str.strip()
    if 'department' in df.columns:
        df['department'] = df['department'].fillna('未填写').astype(str)
    if 'category' in df.columns:
        df['category'] = df['category'].fillna('未分类').astype(str)
    if 'status' in df.columns:
        df['status'] = df['status'].fillna('未知').astype(str)
    for col in du.NUMERIC_COLS:
        if col in df.columns:
            df[col] = df[col].apply(du._to_num)
    for col in du.DATE_COLS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    return df


def _data_manage():
    """数据管理：上传合并 / 在线增删改 / 字段管理 / 导出。"""
    st.markdown('**① 上传 Excel（默认“合并”，不会覆盖已有数据）**')
    c1, c2, c3 = st.columns([2.2, 1, 1])
    up = c1.file_uploader('选择 .xlsx 文件', type=['xlsx', 'xls'], key='up_file')
    mode = c2.radio('上传方式', ['合并到现有', '覆盖全部'], index=0, key='up_mode')
    if c3.button('📤 应用上传', key='up_apply'):
        if up is None:
            st.warning('请先选择 Excel 文件。')
        else:
            try:
                new_raw = pd.read_excel(up, sheet_name=0)
            except Exception as e:
                st.error(f'读取失败：{e}')
            else:
                if mode == '覆盖全部':
                    du.save_store(du.clean_raw(new_raw))
                    st.success(f'已覆盖：共 {len(new_raw)} 行数据。')
                else:
                    merged, added, updated = du.merge_dataframes(du.load_current_data(), new_raw)
                    du.save_store(merged)
                    st.success(f'合并完成：新增 {added} 行，更新 {updated} 行，现有共 {len(merged)} 行。'
                               f'（按 岗位+部门+招聘起 匹配，同一条目以新文件为准）')
                st.rerun()

    st.markdown('**② 在线增删改数据（加行/删行/改单元格后点保存）**')
    editor_df = du.load_editor_data()
    edited = st.data_editor(
        du.to_display(editor_df),
        num_rows='dynamic',
        key='data_editor',
        column_config=_editor_column_config(editor_df),
        width='stretch',
        height=380,
        hide_index=True,
    )
    if st.button('💾 保存编辑', type='primary', key='save_edit'):
        back = du.from_display(edited)
        du.save_store(_fix_editor_types(back))
        st.success('已保存，图表与明细已同步刷新。')
        st.rerun()

    st.markdown('**③ 字段管理（新增 / 删除自定义列）**')
    extras = du.extra_columns(editor_df)
    c1, c2 = st.columns(2)
    new_name = c1.text_input('新字段名称（中文）', key='new_col_name', placeholder='如：面试官、岗位城市')
    new_type = c2.selectbox('新字段类型', ['文本', '数字'], key='new_col_type')
    if st.button('➕ 新增字段', key='add_col'):
        name = new_name.strip()
        if not name:
            st.warning('请输入字段名称。')
        elif name in du.DISPLAY_MAP.values() or name in extras:
            st.warning('该字段已存在。')
        else:
            df = du.load_current_data()
            df[name] = 0.0 if new_type == '数字' else ''
            du.save_store(df)
            st.success(f'已新增字段「{name}」。')
            st.rerun()
    if extras:
        del_sel = st.multiselect('选择要删除的自定义字段', extras, key='del_cols')
        if st.button('🗑️ 删除选中字段', key='del_cols_btn'):
            if del_sel:
                du.save_store(du.load_current_data().drop(columns=del_sel))
                st.success(f'已删除字段：{"、".join(del_sel)}')
                st.rerun()
    else:
        st.caption('当前没有自定义字段；核心字段（岗位/部门/状态/人数等）不可删除。')

    st.markdown('**④ 导出 / 重置**')
    cur = du.load_current_data()
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as w:
        du.to_display(cur).to_excel(w, index=False, sheet_name='岗位统计')
    c1, c2, c3 = st.columns(3)
    c1.download_button('📥 导出 Excel', buf.getvalue(), file_name='招聘数据.xlsx',
                       mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', key='export_xlsx')
    c2.download_button('📄 导出 CSV', du.to_display(cur).to_csv(index=False).encode('utf-8-sig'),
                       file_name='招聘数据.csv', mime='text/csv', key='export_csv')
    if c3.button('↩️ 重置为内置示例数据', key='reset_data'):
        du.reset_store()
        st.rerun()


def page_dashboard():
    st.title('📊 数据看板')
    st.caption('岗位招聘漏斗、周期与转化效率总览。上传新数据默认合并保留历史，也可在线增删改。')

    with st.expander('🗂️ 数据管理：上传合并 / 在线增删改 / 字段管理 / 导出', expanded=False):
        _data_manage()

    df = du.load_current_data()

    with st.sidebar:
        st.subheader('筛选')
        sel_status = st.multiselect('当前状态', du.STATUS_ORDER, default=du.STATUS_ORDER)
        depts = sorted(df['department'].dropna().unique().tolist())
        sel_dept = st.multiselect('部门', depts, default=depts)
        cats = sorted(df['category'].dropna().unique().tolist())
        sel_cat = st.multiselect('招聘类目', cats, default=cats)

    mask = (df['status'].isin(sel_status)) & (df['department'].isin(sel_dept)) & (df['category'].isin(sel_cat))
    d = df[mask].copy()
    if d.empty:
        st.warning('当前筛选条件下没有数据，请调整筛选。')
        return

    kpis = du.compute_kpis(d)
    labels = {
        'total_positions': '岗位总数', 'hiring': '招聘中', 'done': '完成招聘', 'paused': '暂停',
        'total_resumes': '推送简历', 'total_interviewed': '面试人数', 'total_passed': '面试通过',
        'total_offer': 'Offer 人数', 'total_onboarded': '入职人数', 'avg_duration': '平均招聘周期(天)',
        'median_duration': '中位周期(天)', 'current': '现存人数', 'total_left': '离职人数',
    }

    st.subheader('核心指标')
    kpi_row(kpis, ['total_positions', 'hiring', 'done', 'paused'], labels)
    kpi_row(kpis, ['total_resumes', 'total_interviewed', 'total_offer', 'total_onboarded'], labels)
    kpi_row(kpis, ['avg_duration', 'median_duration', 'current', 'total_left'], labels)

    c1, c2 = st.columns([3, 2])
    with c1:
        st.subheader('招聘漏斗')
        fig = du.build_funnel_fig(d)
        invited_ok = d['invited'].notna().sum() >= 5
        if not invited_ok:
            st.caption('注：“邀约面试”环节多数岗位未填写，漏斗已自动省略该环节。')
        st.plotly_chart(fig, width='stretch')
    with c2:
        st.subheader('招聘状态分布')
        status_cnt = d['status'].value_counts().reindex(du.STATUS_ORDER).dropna()
        pie = px.pie(values=status_cnt.values, names=status_cnt.index,
                     color=status_cnt.index,
                     color_discrete_map={s: du.status_label(s) for s in status_cnt.index})
        pie.update_traces(textinfo='label+value')
        pie.update_layout(showlegend=False, height=340, margin=dict(t=10, b=10, l=10, r=10),
                          paper_bgcolor='rgba(0,0,0,0)', font=dict(family='Microsoft YaHei, sans-serif'))
        st.plotly_chart(pie, width='stretch')

    c3, c4 = st.columns(2)
    with c3:
        st.subheader('各部门岗位与入职情况')
        dep = du.department_summary(d)
        bar = px.bar(dep.sort_values('岗位数', ascending=True), x='岗位数', y='部门', orientation='h',
                     color='入职总数', color_continuous_scale='Blues', text='岗位数',
                     labels={'岗位数': '岗位数'})
        bar.update_layout(height=380, margin=dict(t=10, b=10, l=10, r=10), showlegend=False,
                          paper_bgcolor='rgba(0,0,0,0)', font=dict(family='Microsoft YaHei, sans-serif'))
        st.plotly_chart(bar, width='stretch')
    with c4:
        st.subheader('完成招聘岗位 · 招聘周期(天)')
        dur = du.position_duration(d)
        if dur.empty:
            st.info('当前筛选下暂无完成招聘的岗位数据。')
        else:
            bar2 = px.bar(dur, x='周期(天)', y='position', orientation='h', text='周期(天)',
                          color='周期(天)', color_continuous_scale='YlOrRd',
                          labels={'position': '岗位'})
            bar2.update_layout(height=380, margin=dict(t=10, b=10, l=10, r=10), showlegend=False,
                               yaxis=dict(autorange='reversed'),
                               paper_bgcolor='rgba(0,0,0,0)', font=dict(family='Microsoft YaHei, sans-serif'))
            st.plotly_chart(bar2, width='stretch')

    st.subheader('岗位明细')
    show = d.copy()
    show['招聘起'] = show['start_date'].dt.date.astype(str).replace('NaT', '—')
    show['招聘止'] = show['end_date'].dt.date.astype(str).replace('NaT', '—')
    for col in du.NUMERIC_COLS:
        if col in show.columns:
            show[col] = show[col].round(1)
    rename = {
        'position': '岗位', 'department': '部门', 'category': '招聘类目', 'status': '当前状态',
        'resumes': '推送简历', 'invited': '邀约', 'interviewed': '面试', 'passed': '通过',
        'offer': 'Offer', 'onboarded': '入职', 'left': '离职', 'current_headcount': '现存',
        'demand': '招聘需求', 'duration_days': '周期(天)', 'daily_resumes': '日均简历',
    }
    for ec in du.extra_columns(d):
        rename[ec] = ec
    show = show[[c for c in rename if c in show.columns]].rename(columns=rename)
    st.dataframe(show, width='stretch', height=420, hide_index=True)


# ==================== 岗位匹配：①岗位信息 → ②批量上传 → ③智能打分 ====================

def _position_manage(positions):
    """岗位库维护：删除任意岗位（内置也能删）、恢复内置岗位、备份与恢复。"""
    labels = [f"{mt.position_label(p)}　[{p.get('source', '')}]" for p in positions]
    c1, c2 = st.columns([3, 1])
    sel_del = c1.selectbox('选择要删除的岗位（内置岗位同样可以删除）', ['（不删除）'] + labels, key='jd_del')
    if c2.button('🗑️ 删除选中岗位', key='jd_del_btn'):
        if sel_del != '（不删除）' and sel_del in labels:
            idx = labels.index(sel_del)
            target = positions[idx]
            is_builtin = target.get('source') == '内置'
            positions.pop(idx)
            deleted = deleted_builtins()
            if is_builtin:
                deleted = sorted(set(deleted) | {target['name']})
            save_positions(positions, deleted)
            st.session_state['_reset_widgets'] = True
            st.session_state['flash'] = (
                f'已删除「{mt.position_label(target)}」'
                + ('（内置岗位，可用下面的按钮恢复）' if is_builtin else '（自定义岗位，已从持久库移除）'))
            st.rerun()

    deleted = deleted_builtins()
    if deleted:
        st.caption(f'已删除的内置岗位：{"、".join(deleted)}')
        if st.button('♻️ 恢复全部已删除的内置岗位', key='jd_restore'):
            save_positions(positions, [])
            st.session_state.pop('positions', None)
            st.session_state['_reset_widgets'] = True
            st.session_state['flash'] = f'已恢复 {len(deleted)} 个内置岗位。'
            st.rerun()

    with st.expander('💾 备份 / 恢复（换电脑、防误删）'):
        st.caption('备份文件包含全部岗位（含已保存的 JD）和打分记录，换浏览器或换电脑时导入即可。')
        b1, b2 = st.columns(2)
        b1.download_button('⬇️ 导出备份 JSON',
                           json.dumps(backup_payload(), ensure_ascii=False, indent=1).encode('utf-8'),
                           file_name=f'招聘工作台备份_{_dt.datetime.now().strftime("%Y%m%d_%H%M")}.json',
                           mime='application/json', key='bk_down')
        up = b2.file_uploader('⬆️ 导入备份 JSON', type=['json'], key='bk_up')
        if up is not None and b2.button('确认导入并覆盖', key='bk_apply'):
            try:
                data = json.loads(up.getvalue().decode('utf-8'))
            except Exception as e:
                st.error(f'备份文件读取失败：{e}')
            else:
                n_pos, n_hist = restore_payload(data)
                st.session_state['_reset_widgets'] = True
                st.session_state['flash'] = f'已恢复 {n_pos} 个岗位、{n_hist} 条打分记录。'
                st.rerun()


def _jd_manage(positions):
    """① 岗位信息：新增 / 查看岗位库。"""
    with st.expander('➕ 新增岗位（可粘贴 JD 自动解析）', expanded=False):
        name = st.text_input('岗位名称 *', key='jd_name', placeholder='如：电商运营管培生')
        dept = st.text_input('部门（可选）', key='jd_dept', placeholder='如：运营')
        desc = st.text_area('岗位描述（粘贴 JD）', key='jd_desc', height=130,
                            placeholder='粘贴岗位描述，便于自动解析关键词…')

        if st.button('🔍 自动解析描述', key='jd_parse'):
            kws = mt.parse_jd_keywords(desc)
            edu = mt._jd_edu_level(desc)
            yrs = mt._jd_years(desc)
            st.session_state['jd_edu'] = EDU_NAME.get(edu, '本科')
            st.session_state['jd_years'] = int(yrs)
            st.session_state['jd_kw'] = '、'.join(kws)
            st.success(f'解析完成：学历「{st.session_state["jd_edu"]}」、经验 {yrs:g} 年、'
                       f'技能关键词 {len(kws)} 个。请核对后保存。')

        edu_sel = st.selectbox('学历要求', EDU_ORDER,
                               index=EDU_ORDER.index(st.session_state.get('jd_edu', '本科')), key='jd_edu_sel')
        years = st.number_input('经验要求（年）', 0, 20, int(st.session_state.get('jd_years', 0)), step=1,
                                key='jd_years_in')
        kw_text = st.text_input('技能关键词（用顿号或逗号分隔）', key='jd_kw',
                                value=st.session_state.get('jd_kw', ''), placeholder='如：数据分析、Excel、活动策划')
        hard_text = st.text_input('硬性条件（用逗号分隔，可选）', key='jd_hard',
                                  placeholder='如：每周实习4天, 实习3个月以上')

        if st.button('💾 保存岗位', type='primary', key='jd_save'):
            if not name.strip():
                st.error('请填写岗位名称。')
            else:
                kws = [k.strip().strip('，,、') for k in kw_text.replace('、', ',').split(',') if k.strip()]
                if not kws:
                    kws = mt.parse_jd_keywords(desc)
                hard = [h.strip() for h in hard_text.split(',') if h.strip()]
                new_pos = {
                    'name': name.strip(), 'department': dept.strip() or '未填部门',
                    'education': EDU_LEVEL[edu_sel], 'years': float(years),
                    'keywords': kws, 'hard_conditions': hard, 'description': desc,
                    'source': '自定义',
                }
                positions.append(new_pos)
                save_positions(positions, deleted_builtins())
                st.success(f'已保存岗位「{name.strip()}」，共 {len(positions)} 个岗位。')

    n_builtin = sum(1 for p in positions if p.get('source') == '内置')
    st.markdown(f'**岗位库**（内置 {n_builtin} 个 · 自定义 {len(positions) - n_builtin} 个 · '
                f'共 {len(positions)} 个，均可删除）')
    rows = [{
        '岗位名称': p['name'], '部门': p.get('department', ''),
        '学历要求': EDU_NAME.get(p.get('education', 0), '不限'),
        '经验要求': f"{p.get('years', 0):g} 年",
        '技能关键词': '、'.join(p.get('keywords', [])[:8]),
        '硬性条件': '、'.join(p.get('hard_conditions', [])) or '—',
        '来源': p.get('source', ''),
    } for p in positions]
    st.dataframe(pd.DataFrame(rows), width='stretch', height=200, hide_index=True)
    _position_manage(positions)


def _gather_candidates():
    """② 批量上传简历（可多选文件 + 粘贴文本），按内容缓存解析结果。"""
    files = st.file_uploader('批量上传简历（PDF / Word / TXT，可多选）', type=['pdf', 'docx', 'txt'],
                             accept_multiple_files=True, key='shared_files')
    pasted = st.text_area('或粘贴简历文本（一份）', height=110, key='shared_paste',
                          placeholder='粘贴候选人简历内容…')
    raw = []
    for f in files or []:
        try:
            raw.append((f.name, mt.extract_text(f.name, f.getvalue())))
        except Exception as e:
            st.error(f'解析 {f.name} 失败：{e}')
    if pasted and pasted.strip():
        raw.append(('手动粘贴', pasted))
    sig = repr([(n, len(t)) for n, t in raw])
    if st.session_state.get('_raw_sig') != sig:
        cands = []
        for n, t in raw:
            info = mt.parse_resume(t)
            info['source'] = n
            cands.append(info)
        st.session_state['_raw_sig'] = sig
        st.session_state['_candidates'] = cands
    return st.session_state.get('_candidates', [])


def _do_scoring(pos, candidates, mode, api_key, base_url, model):
    """③ 执行打分，返回 (rows, details)。"""
    jd_text = mt.build_jd_text(pos)
    local_jd = {'education': pos.get('education', 0), 'years': pos.get('years', 0),
                'keywords': pos.get('keywords', []), 'hard_conditions': pos.get('hard_conditions', [])}

    rows, details = [], []
    if mode == '本地规则打分（无需 API）':
        for c in candidates:
            s = mt.score_resume(local_jd, c)
            rows.append({
                '姓名': c['name'] or '未识别', '电话': c['phone'] or '—',
                '学历': c['education_name'], '经验(年)': c['years'],
                '匹配分': s['total'], '建议': mt.score_tag(s['total']),
                '技能命中': '、'.join(s['skill_hits'][:8]) if s['skill_hits'] else '—',
            })
            details.append((c, None, s, ''))
        return rows, details

    if not api_key.strip():
        st.error('请先在下方填写 DeepSeek API Key（可到 platform.deepseek.com 免费申请），'
                 '或切换为“本地规则打分”。')
        return None, None

    cache = st.session_state.setdefault('ai_cache', {})
    prog = st.progress(0, text='DeepSeek 智能分析中…')
    total = len(candidates)
    for i, c in enumerate(candidates):
        local = mt.score_resume(local_jd, c)
        key = hashlib.md5((jd_text + '||' + c['text'][:800]).encode('utf-8')).hexdigest()
        ai, note = None, ''
        if key in cache:
            ai = cache[key]
        else:
            try:
                ai = mt.deepseek_analyze(api_key, jd_text, c['text'], base_url, model)
                cache[key] = ai
            except Exception as e:
                note = str(e)
        score = ai['score'] if ai else local['total']
        rows.append({
            '姓名': c['name'] or '未识别', '电话': c['phone'] or '—',
            '学历': c['education_name'], '经验(年)': c['years'],
            '匹配分': score,
            '建议': mt.score_tag(score),
            '智能摘要': (ai['summary'] if ai else 'AI 调用失败，已用本地规则分')[:60],
        })
        details.append((c, ai, local, note))
        prog.progress((i + 1) / total, text=f'已分析 {i + 1}/{total}：{c["name"] or "未识别"}')
    prog.empty()
    return rows, details


def _build_run(pos, mode, rows, details):
    """把一次打分整理成可持久化的记录。"""
    det = []
    for c, ai, local, note in details:
        score = ai['score'] if ai else local['total']
        det.append({
            'name': c.get('name') or '未识别',
            'phone': c.get('phone') or '—',
            'education': c.get('education_name', ''),
            'years': c.get('years', 0),
            'years_raw': c.get('years_raw', ''),
            'source': c.get('source', ''),
            'text': (c.get('text') or '')[:1500],
            'score': score,
            'tag': mt.score_tag(score),
            'ai': ai or None,
            'local': local,
            'note': note,
        })
    stamp = _dt.datetime.now().strftime('%Y%m%d%H%M%S')
    return _jsonable({
        'id': f'{stamp}-{hashlib.md5((mt.position_label(pos) + str(len(rows))).encode("utf-8")).hexdigest()[:6]}',
        'time': _dt.datetime.now().strftime('%Y-%m-%d %H:%M'),
        'position': mt.position_label(pos),
        'department': pos.get('department', ''),
        'mode': mode,
        'rows': rows,
        'details': det,
    })


def render_run(run):
    """渲染某一条已保存的打分记录。"""
    rows = run.get('rows') or []
    if not rows:
        st.info('这条记录里没有结果。')
        return
    res = pd.DataFrame(rows)
    if '匹配分' in res.columns:
        res = res.sort_values('匹配分', ascending=False).reset_index(drop=True)
    res.insert(0, '排名', res.index + 1)
    st.caption(f"岗位：{run.get('position', '')} ｜ 打分方式：{run.get('mode', '')} ｜ 时间：{run.get('time', '')}")
    st.dataframe(res, width='stretch', height=380, hide_index=True, column_config={
        '匹配分': st.column_config.ProgressColumn('匹配分', min_value=0, max_value=100, format='%.1f'),
        '排名': st.column_config.NumberColumn('排名'),
    })
    st.download_button('下载打分结果 CSV', res.to_csv(index=False).encode('utf-8-sig'),
                       file_name=f"匹配打分结果_{run.get('position', '')}_{run.get('time', '').replace(':', '')}.csv",
                       mime='text/csv', key=f"csv_{run.get('id')}")
    st.markdown('**筛选建议**：匹配分 ≥ 70 建议约面 ｜ 50–69 待定 ｜ < 50 暂缓')

    details = run.get('details') or []
    if not details:
        return
    st.markdown('**候选人详情**')
    for i, d in enumerate(sorted(details, key=lambda x: -(x.get('score') or 0))):
        ai = d.get('ai')
        local = d.get('local') or {}
        with st.expander(f"{d.get('name') or '未识别'} · 匹配分 {d.get('score')} · {d.get('tag', '')}"):
            if ai:
                if ai.get('summary'):
                    st.markdown(f"**💬 {ai['summary']}**")
                if ai.get('strengths'):
                    st.markdown('✅ **优势**：' + '；'.join(ai['strengths']))
                if ai.get('gaps'):
                    st.markdown('⚠️ **不足**：' + '；'.join(ai['gaps']))
                if ai.get('highlights'):
                    st.markdown('⭐ **亮点**：' + '；'.join(ai['highlights']))
                if ai.get('suggestion'):
                    st.markdown(f"📌 **建议**：{ai['suggestion']}")
            elif d.get('note'):
                st.markdown(f"⚠️ AI 分析失败（{d['note']}），已用本地规则分替代。")
            if local:
                st.markdown(f"**本地规则分**：学历 {local.get('edu_score', 0)}/20 ｜ "
                            f"经验 {local.get('exp_score', 0)}/30 ｜ 技能 {local.get('skill_score', 0)}/40 ｜ "
                            f"硬性 {local.get('hard_score', 0)}/10")
            st.markdown(f"**简历来源**：{d.get('source', '')} ｜ **电话** {d.get('phone') or '—'} ｜ "
                        f"**学历** {d.get('education', '')} ｜ **经验** {d.get('years_raw') or '未识别'}")
            if d.get('text'):
                st.text_area('简历原文（已保存 1500 字以内）', d['text'], height=160,
                             key=f"hist_txt_{run.get('id')}_{i}", disabled=True)


def _history_ui():
    """打分记录：存放在持久层里，刷新或重开页面都还在。"""
    st.markdown('#### 📌 打分结果（刷新页面、关闭浏览器后仍然保留）')
    hist = load_history()
    if not hist:
        st.info('还没有打分记录。上传简历后点「🚀 开始打分」，结果会自动保存到这里。')
        return

    ids = [r.get('id') for r in hist]
    labels = {r.get('id'): f"{r.get('time', '')} ｜ {r.get('position', '')} ｜ "
                         f"{len(r.get('rows') or [])} 份 ｜ {r.get('mode', '')}" for r in hist}
    want = st.session_state.get('view_run_id')
    idx = ids.index(want) if want in ids else 0
    sel_id = st.selectbox(f'查看哪一次打分（自动保留最近 {HIST_MAX} 次）', ids, index=idx,
                          format_func=lambda i: labels.get(i, i))
    st.session_state['view_run_id'] = sel_id
    run = next((r for r in hist if r.get('id') == sel_id), hist[0])
    render_run(run)

    with st.expander('🧹 记录管理：删除记录 / 清空'):
        c1, c2 = st.columns(2)
        if c1.button('🗑️ 删除这条记录', key='hist_del_one'):
            save_history([r for r in hist if r.get('id') != sel_id])
            st.session_state.pop('view_run_id', None)
            st.session_state['flash'] = '已删除该条打分记录。'
            st.rerun()
        if c2.button('🧹 清空全部打分记录', key='hist_clear'):
            save_history([])
            st.session_state.pop('view_run_id', None)
            st.session_state['flash'] = '已清空全部打分记录。'
            st.rerun()


def _setup_help():
    """未配置云端数据库时，给出一次性配置步骤。"""
    if cloud_enabled():
        return
    with st.expander('🔧 如何开启“永久保存”（一次性设置，约 3 分钟）'):
        st.markdown('现在的数据放在应用服务器上：刷新页面、换浏览器都不会丢，'
                    '但应用休眠或被重启后可能清空。按下面 5 步接到免费的云端数据库后，'
                    '岗位库、岗位 JD、打分记录就会永久保存。')
        st.markdown('**1.** 打开 [supabase.com](https://supabase.com)，用邮箱注册并新建一个免费项目'
                    '（Region 选 Singapore 更快）。')
        st.markdown('**2.** 左侧 **SQL Editor** → **New query**，粘贴下面这段 SQL，点 **Run**：')
        st.code(SUPABASE_SQL, language='sql')
        st.markdown('**3.** 左侧 **Project Settings → API**，复制 **Project URL** 与 **anon public** key。')
        st.markdown('**4.** 回到 share.streamlit.io 打开这个应用，进入 **Settings → Secrets**，'
                    '粘贴下面两行后点 **Save**：')
        st.code('SUPABASE_URL = "第 3 步复制的 Project URL"\n'
                'SUPABASE_KEY = "第 3 步复制的 anon key"', language='toml')
        st.markdown('**5.** 应用会自动重启。之后本页顶部提示会变成 🟢，数据即永久保存。')
        st.caption('如果保存时报权限不足（项目开了 RLS），在 SQL Editor 里再运行：'
                   'alter table hr_workbench_kv enable row level security;'
                   'create policy "hr_kv_all" on hr_workbench_kv for all using (true) with check (true);')


def page_matching():
    st.title('📄 岗位匹配：填写岗位 → 批量上传 → 智能打分')
    st.caption('一个流程走完：先维护岗位信息，再批量上传简历，最后 DeepSeek 智能分析打分排名（打分即筛选）。')
    _storage_notice()

    if st.session_state.get('_reset_widgets'):      # 删除/恢复岗位后，重置下拉框选中项
        st.session_state['_reset_widgets'] = False
        for k in ('match_pos', 'jd_del'):
            st.session_state.pop(k, None)

    if st.session_state.get('flash'):
        st.success(st.session_state.pop('flash'))

    positions = get_positions()
    if not positions:
        st.warning('岗位库现在是空的：请展开「➕ 新增岗位」新建岗位，'
                   '或点「♻️ 恢复全部已删除的内置岗位」把内置岗位找回来。')
        return

    # ---------- 第 1 步：岗位信息 ----------
    st.markdown('#### ① 填写 / 选择岗位信息')
    _jd_manage(positions)

    if not positions:
        st.warning('岗位库现在是空的，请先新增岗位。')
        return

    sel = st.selectbox('本次匹配的岗位', position_options(positions), key='match_pos')
    pos = find_position(positions, sel)
    with st.container(border=True):
        st.markdown(f"**{pos['name']}**（{pos.get('department', '')}）")
        st.caption(f"学历 {EDU_NAME.get(pos.get('education', 0), '不限')} 及以上 ｜ "
                   f"经验 {pos.get('years', 0):g} 年 ｜ 关键词：{'、'.join(pos.get('keywords', [])[:10])}")

    # ---------- 第 2 步：批量上传简历 ----------
    st.markdown('#### ② 批量上传简历')
    candidates = _gather_candidates()
    if candidates:
        st.success(f'已解析 {len(candidates)} 份简历：'
                   + '、'.join(f"{c['name'] or '未识别'}" for c in candidates[:8])
                   + ('…' if len(candidates) > 8 else ''))
    else:
        st.info('请先上传或粘贴简历，再进入第 ③ 步打分。')

    # ---------- 第 3 步：智能打分 ----------
    st.markdown('#### ③ 开始匹配打分')
    with st.expander('⚙️ 打分设置', expanded=True):
        mode = st.radio('打分方式', ['智能分析（DeepSeek）', '本地规则打分（无需 API）'],
                        horizontal=True, key='score_mode')
        if mode == '智能分析（DeepSeek）':
            bound_key = default_api_key() or _saved_api_key()
            if bound_key:
                src = 'Streamlit Secrets' if default_api_key() else '本地保存'
                st.success(f'✅ 已绑定 Key（来自 {src}），打开页面自动带出，无需手动输入。')
            api_key = st.text_input('DeepSeek API Key', type='password', value=bound_key,
                                    key='ds_key', placeholder='sk-...（platform.deepseek.com 申请）')
            remember = st.checkbox('💾 记住 Key 到本地（下次自动填入；云端建议用 Secrets 绑定）',
                                   value=bool(bound_key), key='ds_remember')
            c1, c2 = st.columns(2)
            base_url = c1.text_input('API 地址', value='https://api.deepseek.com', key='ds_base',
                                     help='默认官方接口；若用火山方舟等兼容服务可改为其地址')
            model = c2.text_input('模型', value='deepseek-chat', key='ds_model',
                                  help='官方为 deepseek-chat / deepseek-reasoner；火山方舟填对应模型或接入点 ID')
        else:
            api_key, base_url, model = '', '', ''

    score_btn = st.button('🚀 开始打分', type='primary', key='match_run')
    if score_btn:
        if not candidates:
            st.warning('请先在第 ② 步上传简历。')
        elif mode == '智能分析（DeepSeek）' and not api_key.strip():
            st.error('请填写 DeepSeek API Key，或切换为“本地规则打分”。')
        else:
            if mode == '智能分析（DeepSeek）' and remember and api_key.strip():
                _save_api_key(api_key.strip())
            rows, details = _do_scoring(pos, candidates, mode, api_key, base_url, model)
            if rows is not None:
                run = _build_run(pos, mode, rows, details)
                save_history([run] + [r for r in load_history() if r.get('id') != run['id']])
                st.session_state['view_run_id'] = run['id']
                st.session_state['flash'] = f'打分完成：{len(run["rows"])} 份简历已保存到下面的「打分记录」。'
                st.rerun()

    _history_ui()
    _setup_help()


# ==================== 入口 ====================

pages = {'📊 数据看板': page_dashboard,
         '📄 岗位匹配': page_matching}
with st.sidebar:
    st.title('🎯 招聘 HR 工作台')
    page = st.radio('功能模块', list(pages.keys()), label_visibility='collapsed')
st.sidebar.caption('智能打分调用 DeepSeek API（填 Key 后可用）；本地规则打分不依赖网络与 AI。')
pages[page]()
