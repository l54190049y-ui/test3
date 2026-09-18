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
DASH_KEY = 'dashboard_rows_v1'
SNAP_KEY = 'kpi_snapshots_v1'
SNAP_MAX = 30
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


# ---------------- 数据看板的数据也接进持久层（不用改 data_utils.py） ----------------

def _looks_fresh_container():
    """没有云端配置、本地持久层文件与 data_utils 临时文件都不存在 → 应用多半刚重启过。"""
    if cloud_enabled():
        return False
    if os.path.exists(LOCAL_KV_FILE):
        return False
    store_file = getattr(du, 'STORE_FILE', '')
    if store_file and os.path.exists(store_file):
        return False
    return True


def _legacy_dashboard_rows():
    """迁移旧版本写在 data_utils 临时文件里的看板数据。"""
    store_file = getattr(du, 'STORE_FILE', '')
    try:
        if store_file and os.path.exists(store_file):
            with open(store_file, 'r', encoding='utf-8') as fp:
                rows = json.load(fp)
            if isinstance(rows, list) and rows:
                return rows
    except Exception:
        pass
    return []


def _use_company_as_department(cols):
    """有的表里部门列叫「公司」：自动当作部门用；有「部门」列的表不受影响。"""
    try:
        cols = {str(c) for c in list(cols)}
    except Exception:
        cols = set()
    try:
        if '公司' in cols and '部门' not in cols:
            du.COL_MAP['公司'] = 'department'
        elif '公司' in du.COL_MAP and '公司' not in cols:
            du.COL_MAP.pop('公司', None)
    except Exception:
        pass


def _dashboard_rows():
    """看板数据在会话内缓存一份，避免每次 rerun 都去读云端。"""
    if '_dash_rows' not in st.session_state:
        rows = kv_get(DASH_KEY, None)
        if not (isinstance(rows, list) and rows):
            rows = _legacy_dashboard_rows()
            if rows:
                kv_set(DASH_KEY, rows)
        st.session_state['_dash_rows'] = rows if isinstance(rows, list) else []
    return st.session_state['_dash_rows']


def _durable_load_current_data():
    """看板读数据：优先持久层，取不到再走 data_utils 原本的逻辑（内置 Excel / 本地文件）。"""
    rows = _dashboard_rows()
    if rows:
        _use_company_as_department(rows[0].keys() if isinstance(rows[0], dict) else [])
        try:
            return du.clean_raw(pd.DataFrame(rows))
        except Exception:
            pass
    try:
        if os.path.exists(getattr(du, 'DEFAULT_DATA', '')):
            _use_company_as_department(pd.read_excel(du.DEFAULT_DATA, sheet_name=0, nrows=0).columns)
    except Exception:
        pass
    return _DU_LOAD()


def _durable_save_store(df):
    """看板写数据：原逻辑 + 持久层各写一份，并记一条快照供“跟上次比”。"""
    _DU_SAVE(df)
    rows = _jsonable(df.to_dict('records'))
    st.session_state['_dash_rows'] = rows
    kv_set(DASH_KEY, rows)
    try:
        _save_snapshot(du.clean_raw(df), force=True)
    except Exception:
        pass


def _durable_reset_store():
    """重置看板：回到内置示例数据，同时清掉持久层里的看板数据。"""
    _DU_RESET()
    st.session_state['_dash_rows'] = []
    kv_set(DASH_KEY, [])


_DU_LOAD = du.load_current_data
_DU_SAVE = du.save_store
_DU_RESET = du.reset_store
du.load_current_data = _durable_load_current_data
du.save_store = _durable_save_store
du.reset_store = _durable_reset_store


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
    try:
        dash_rows = du.load_editor_data().to_dict('records')
    except Exception:
        dash_rows = []
    return _jsonable({
        'version': 2,
        'exported_at': _now_str(),
        'positions': get_positions(),
        'deleted_builtins': deleted_builtins(),
        'score_history': load_history(),
        'dashboard_rows': dash_rows,
    })


def restore_payload(data):
    """从备份 JSON 恢复岗位库、打分记录与看板数据，返回（岗位数, 记录数, 看板行数）。"""
    positions = [_norm_pos(p) for p in (data.get('positions') or [])]
    deleted = [str(x) for x in (data.get('deleted_builtins') or [])]
    hist = data.get('score_history') or []
    dash = data.get('dashboard_rows') or []
    save_positions(positions, deleted)
    save_history(hist)
    if dash:
        try:
            du.save_store(du.clean_raw(pd.DataFrame(dash)))      # 走包装函数，会同时写持久层
        except Exception:
            pass
    st.session_state.pop('positions', None)
    st.session_state['positions'] = load_positions()
    return len(st.session_state['positions']), len(hist), len(dash)


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
                _use_company_as_department(new_raw.columns)
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


# ---------------- 老板视角：结论指标 / 风险岗位 / 卡点分析 ----------------

def _num(v, default=float('nan')):
    try:
        if v is None or pd.isna(v):
            return default
        return float(v)
    except Exception:
        return default


def _pct(num, den):
    n, m = _num(num), _num(den)
    if m != m or not m:
        return float('nan')
    return n / m * 100


def _fmt_num(v, digits=0):
    v = _num(v)
    return f'{v:.{digits}f}' if v == v else '—'


def _fmt_pct(v, digits=1):
    v = _num(v)
    return f'{v:.{digits}f}%' if v == v else '—'


def _worked_days(r, today):
    """岗位已用天数：招聘起 → 招聘止（在招的算到今天）。"""
    start, end = r.get('start_date'), r.get('end_date')
    if pd.isna(start):
        return float('nan')
    stop = end if pd.notna(end) else today
    return float((stop - start).days + 1)


def _drop_blank_rows(d):
    """去掉 Excel 里既没有岗位名、也没有部门的空行/合计行。返回（数据, 丢掉的行数）。"""
    if d.empty:
        return d, 0
    blank = d[d['position'].astype(str).str.strip().isin(['未命名岗位', '', 'nan', 'None'])
              & d['department'].astype(str).str.strip().isin(['未填写', '', 'nan', 'None'])]
    if blank.empty:
        return d, 0
    return d.drop(index=blank.index), int(len(blank))


def _position_analysis(d, target_days, ratio):
    """逐个岗位算周期、漏斗转化、卡点与问题清单（老板最关心的异常）。"""
    today = pd.Timestamp.today().normalize()
    done = d[d['status'] == '完成招聘']
    med_all = _num(done['duration_days'].dropna().median()) if not done.empty else float('nan')
    cat_med = {}
    if not done.empty:
        for cat, g in done.groupby('category'):
            m = _num(g['duration_days'].dropna().median())
            if m == m:
                cat_med[cat] = m

    records = []
    for _, r in d.iterrows():
        status = str(r.get('status', ''))
        hiring = status == '招聘中'
        cycle = _num(r.get('duration_days'))
        if cycle != cycle or cycle <= 0:
            if hiring:
                cycle = _worked_days(r, today)          # 在招：算到今天
            elif pd.notna(r.get('start_date')) and pd.notna(r.get('end_date')):
                cycle = _worked_days(r, today)          # 已完成/暂停：算到结束日
            else:
                cycle = float('nan')

        resumes = _num(r.get('resumes'))
        interviewed = _num(r.get('interviewed'))
        passed = _num(r.get('passed'))
        offer = _num(r.get('offer'))
        onboarded = _num(r.get('onboarded'))
        left = _num(r.get('left'))
        demand_max = _num(r.get('demand_max'))
        demand_min = _num(r.get('demand_min'))

        r2i = _pct(interviewed, resumes)
        i2p = _pct(passed, interviewed)
        p2o = _pct(offer, passed)
        o2o = _pct(onboarded, offer)

        base = cat_med.get(r.get('category'))
        if base is None:
            base = med_all if med_all == med_all else float('nan')

        issues, hints, score, flagged = [], [], 0, []
        if hiring and cycle == cycle and cycle > target_days:
            issues.append(f'在招 {cycle:.0f} 天，已超过目标周期 {target_days:.0f} 天')
            hints.append('先判断是“没人投”还是“投了筛不出来”：看简历量、面试转化和用人部门反馈')
            score += 3
        if (not hiring) and status in ('完成招聘', '暂停') and cycle == cycle:
            slow_abs = cycle > target_days
            slow_rel = base == base and cycle > base * ratio and cycle > base + 5
            if slow_abs or slow_rel:
                cmp = f'（目标 {target_days:.0f} 天' + (f'，同类中位 {base:.0f} 天）' if base == base else '）')
                issues.append(f'招聘周期 {cycle:.0f} 天偏长{cmp}')
                hints.append('复盘时间花在哪一环：简历不足、面试排期、还是决策慢')
                score += 2
        if r2i == r2i and r2i < 10 and resumes >= 10:
            issues.append(f'简历→面试只有 {r2i:.1f}%（{resumes:.0f} 份简历只面了 {interviewed:.0f} 人）')
            hints.append('简历量不缺但进不了面试：核对 JD 硬性条件、渠道来源与初筛口径')
            flagged.append(('简历→面试', r2i))
            score += 2
        if i2p == i2p and i2p < 30 and interviewed >= 5:
            issues.append(f'面试通过率只有 {i2p:.1f}%（{interviewed:.0f} 人过 {passed:.0f} 人）')
            hints.append('面试通过率低：统一评分标准，确认用人部门期望是否与 JD 一致')
            flagged.append(('面试→通过', i2p))
            score += 2
        if o2o == o2o and o2o < 80 and offer >= 1:
            issues.append(f'Offer 流失：{offer:.0f} 个 Offer 只入职 {onboarded:.0f} 人')
            hints.append('Offer 接受率低：检查薪资竞争力、谈薪节奏与候选人体验')
            flagged.append(('Offer→入职', o2o))
            score += 2
        if left == left and left > 0:
            issues.append(f'入职后离职 {left:.0f} 人')
            hints.append('新人流失：复盘入职引导与岗位预期管理')
            score += 1
        if demand_max == demand_max and demand_max > 0 and status == '完成招聘':
            if demand_min == demand_min and onboarded == onboarded and onboarded < demand_min:
                issues.append(f'未招满：需求 {demand_min:.0f} 人，只入职 {onboarded:.0f} 人')
                score += 1

        stages = [('简历→面试', r2i), ('面试→通过', i2p), ('Offer→入职', o2o)]
        valid = [(n, v) for n, v in stages if v == v]
        pool = flagged if flagged else valid
        bottleneck = min(pool, key=lambda x: x[1])[0] if pool else '—'

        records.append({
            '岗位': str(r.get('position', '')), '部门': str(r.get('department', '')),
            '类目': str(r.get('category', '')), '状态': status,
            '周期(天)': round(cycle, 1) if cycle == cycle else float('nan'),
            '目标(天)': float(target_days),
            '同类中位(天)': round(base, 1) if base == base else float('nan'),
            '需求': demand_max, '简历': resumes, '面试': interviewed, 'Offer': offer, '入职': onboarded,
            '通过': passed, '离职': left,
            '简历→面试%': round(r2i, 1) if r2i == r2i else float('nan'),
            '面试→通过%': round(i2p, 1) if i2p == i2p else float('nan'),
            'Offer→入职%': round(o2o, 1) if o2o == o2o else float('nan'),
            '主要卡点': bottleneck,
            '主要问题': '；'.join(issues),
            '建议': '；'.join(dict.fromkeys(hints)),
            '问题数': len(issues),
            '严重度': score,
        })
    return pd.DataFrame(records)


def _health(level):
    return {'green': '🟢', 'amber': '🟡', 'red': '🔴'}.get(level, '⚪')


def _headline_metrics(d):
    """核心经营数字（与目标周期无关，同时用于快照对比）。"""
    with_demand = d[d['demand_max'].notna() & (d['demand_max'] > 0)]
    demand = float(with_demand['demand_max'].sum())
    onboarded_dem = float(with_demand['onboarded'].fillna(0).sum())
    done = d[d['status'] == '完成招聘']
    resumes = float(d['resumes'].fillna(0).sum())
    onboarded = float(d['onboarded'].fillna(0).sum())
    with_offer = d[d['offer'].fillna(0) > 0]
    offer = float(with_offer['offer'].fillna(0).sum())
    onb_offer = float(with_offer['onboarded'].fillna(0).sum())
    return {
        'date': str(pd.Timestamp.today().date()),
        'saved_at': _now_str(),
        '岗位数': int(len(d)),
        '需求': round(demand, 1),
        '到位': round(onboarded_dem, 1),
        '到位率': round(onboarded_dem / demand * 100, 1) if demand else None,
        '在招': int((d['status'] == '招聘中').sum()),
        '平均周期': round(_num(done['duration_days'].dropna().mean()), 1) if not done.empty else None,
        '简历': round(resumes, 1),
        '入职': round(onboarded, 1),
        '转化率': round(onboarded / resumes * 100, 2) if resumes else None,
        'Offer接受率': min(round(onb_offer / offer * 100, 1), 100.0) if offer else None,
    }


def _snapshots():
    return [s for s in (kv_get(SNAP_KEY, []) or []) if isinstance(s, dict)]


def _save_snapshot(d, force=False):
    """记录当天快照（同一天只留一份），用于“比上次好了还是差了”。"""
    try:
        m = _headline_metrics(d)
        snaps = _snapshots()
        today = [s for s in snaps if s.get('date') == m['date']]
        if today and not force:
            old = {k: v for k, v in today[-1].items() if k != 'saved_at'}
            new = {k: v for k, v in m.items() if k != 'saved_at'}
            if old == new:
                return
        snaps = [s for s in snaps if s.get('date') != m['date']]
        snaps.append(m)
        snaps.sort(key=lambda s: str(s.get('date')))
        kv_set(SNAP_KEY, snaps[-SNAP_MAX:])
    except Exception:
        pass


def _delta_vs_prev(m):
    """与上一条快照比：{'到位': +6, '在招': -1, ...}，_date 是上期日期。"""
    prev = None
    for s in reversed(_snapshots()):
        if s.get('date') != m.get('date'):
            prev = s
            break
    if not prev:
        return None
    out = {'_date': prev.get('date')}
    for k in ('需求', '到位', '在招', '平均周期', '转化率', 'Offer接受率'):
        a, b = _num(m.get(k)), _num(prev.get(k))
        out[k] = (a - b) if (a == a and b == b) else None
    return out


def _delta_txt(dv, key, unit='', digits=0):
    if not dv or dv.get(key) is None:
        return None
    return f'{dv[key]:+.{digits}f}{unit}（较 {dv["_date"]}）'


def _dashboard_summary(d, ana, target_days):
    """第一屏：3 秒看懂现状 —— 状态灯、四个北极星指标、比上次的变化。"""
    m = _headline_metrics(d)
    dv = _delta_vs_prev(m)
    overdue = int(((ana['状态'] == '招聘中') & (ana['周期(天)'] > target_days)).sum())
    risk_n = int((ana['问题数'] > 0).sum())
    rate, cyc, conv, accept = _num(m['到位率']), _num(m['平均周期']), _num(m['转化率']), _num(m['Offer接受率'])

    lv_rate = 'green' if rate == rate and rate >= 100 else ('amber' if rate == rate and rate >= 80 else 'red')
    lv_risk = 'green' if (overdue == 0 and risk_n == 0) else ('amber' if (overdue <= 2 and risk_n <= 5) else 'red')
    lv_cyc = 'green' if (cyc != cyc or cyc <= target_days) else ('amber' if cyc <= target_days * 1.3 else 'red')
    lv_conv = 'green' if (conv == conv and conv >= 8) else ('amber' if conv == conv and conv >= 4 else 'red')
    order = ['green', 'amber', 'red']
    overall = max([lv_rate, lv_risk, lv_cyc, lv_conv], key=order.index)

    st.subheader('一、结论：3 秒看懂')
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f'{_health(lv_rate)} 需求到位率', _fmt_pct(rate, 0) if rate == rate else '—',
              _delta_txt(dv, '到位', ' 人') or f"{_fmt_num(m['到位'])}/{_fmt_num(m['需求'])} 人")
    c2.metric(f'{_health(lv_risk)} 在招与风险', f"{m['在招']} 个在招",
              (f'超期 {overdue} 个 · 问题岗位 {risk_n} 个' if (overdue or risk_n) else '无超期、无异常'),
              delta_color='inverse' if (overdue or risk_n) else 'off')
    c3.metric(f'{_health(lv_cyc)} 平均招聘周期', f'{_fmt_num(cyc, 1)} 天',
              _delta_txt(dv, '平均周期', ' 天', 1) or f'目标 {target_days:.0f} 天', delta_color='inverse')
    c4.metric(f'{_health(lv_conv)} 简历→入职转化', _fmt_pct(conv) if conv == conv else '—',
              _delta_txt(dv, '转化率', ' pt', 2) or f"累计 {_fmt_num(m['简历'])} 份简历")

    if overall == 'green':
        st.success('🟢 整体健康，暂时不需要你介入。')
    elif overall == 'amber':
        st.warning(f'🟡 基本正常，但有 {risk_n} 个岗位需要跟进 —— 下面第二节列的是需要你拍板或给资源的事。')
    else:
        st.error(f'🔴 有需要立刻处理的问题：{risk_n} 个岗位异常，其中 {overdue} 个在招岗位已超期。')

    st.caption(
        f"到位 {_fmt_num(m['到位'])}/{_fmt_num(m['需求'])} 人（{_fmt_pct(rate, 0)}）｜在招 {m['在招']} 个（超期 {overdue} 个）"
        f"｜平均周期 {_fmt_num(cyc, 1)} 天｜Offer 接受率 {_fmt_pct(accept, 0)}"
        f'　｜　状态灯口径：到位率 ≥100% 🟢、≥80% 🟡；在招无超期且无异常 🟢；周期 ≤{target_days:.0f} 天 🟢；'
        '简历→入职 ≥8% 🟢、≥4% 🟡。')


def _decision_items(d, ana, target_days):
    """把异常按“同一类问题”聚合成老板要拍板的事项（不是岗位清单）。"""
    risk = ana[ana['问题数'] > 0]
    items = []

    def join_names(rows, a='简历', b='面试'):
        return '、'.join(f"{r['岗位']}（{_fmt_num(r[a])}→{_fmt_num(r[b])}）" for _, r in rows.iterrows())

    sub = risk[(risk['简历→面试%'] < 10) & (risk['简历'] >= 10)]
    if not sub.empty:
        res, itv = float(sub['简历'].sum()), float(sub['面试'].sum())
        paused = int((sub['状态'] == '暂停').sum())
        items.append({
            '类型': '渠道 / 简历质量',
            '标题': f"{len(sub)} 个岗位收了 {res:.0f} 份简历，只进 {itv:.0f} 人面试（{itv / res * 100:.1f}%）",
            '涉及岗位': join_names(sub),
            '影响': f"约 {res - itv:.0f} 份简历没有产出" + (f"，其中 {paused} 个岗位已经暂停招聘" if paused else ''),
            '建议': '先停掉产出最差的渠道，改走校招/内推；同时复核这些岗位的学历、经验硬性条件是不是卡得过严',
            '需要老板': '确认是否放宽硬性条件、是否批准更换招聘渠道',
            '严重度': 3,
        })

    sub = risk[(risk['面试→通过%'] < 30) & (risk['面试'] >= 5)]
    if not sub.empty:
        itv, pss = float(sub['面试'].sum()), float(sub['通过'].sum())
        items.append({
            '类型': '用人标准 / 面试',
            '标题': f"{len(sub)} 个岗位面了 {itv:.0f} 人，只过 {pss:.0f} 人（{pss / itv * 100:.0f}%）",
            '涉及岗位': join_names(sub, '面试', '通过'),
            '影响': '简历筛选没问题，但面试大量淘汰：岗位画像与用人部门期望可能不一致，也在消耗面试官时间',
            '建议': '拉用人部门做一次标准对齐，复盘 3～5 份典型被淘汰的简历',
            '需要老板': '请用人部门负责人配合对齐面试标准（或确认标准是否过严）',
            '严重度': 2,
        })

    sub = risk[(risk['Offer→入职%'] < 80) & (risk['Offer'] >= 1)]
    if not sub.empty:
        off, onb = float(sub['Offer'].sum()), float(sub['入职'].sum())
        items.append({
            '类型': '薪酬 / Offer 竞争力',
            '标题': f"{len(sub)} 个岗位发出 {off:.0f} 个 Offer，只入职 {onb:.0f} 人（接受率 {onb / off * 100:.0f}%）",
            '涉及岗位': join_names(sub, 'Offer', '入职'),
            '影响': f"{off - onb:.0f} 人在 Offer 阶段流失，前面所有筛选投入作废，岗位空缺时间被拉长",
            '建议': '对比同岗位市场薪资、压缩发 Offer 到入职的间隔、增加入职前跟进',
            '需要老板': '是否需要调整薪资区间，或授权更快给出 Offer',
            '严重度': 3,
        })

    over = risk[(risk['状态'] == '招聘中') & (risk['周期(天)'] > target_days)]
    slow = risk[(risk['状态'] != '招聘中') & (risk['主要问题'].str.contains('偏长', na=False))]
    if not over.empty or not slow.empty:
        parts = [f"{r['岗位']}（{r['状态']}，{_fmt_num(r['周期(天)'])} 天）" for _, r in over.iterrows()]
        parts += [f"{r['岗位']}（{_fmt_num(r['周期(天)'])} 天，同类中位 {_fmt_num(r['同类中位(天)'])} 天）"
                  for _, r in slow.iterrows()]
        items.append({
            '类型': '周期 / 推进节奏',
            '标题': f"{len(over) + len(slow)} 个岗位周期偏长（目标 {target_days:.0f} 天）",
            '涉及岗位': '、'.join(parts),
            '影响': '岗位空缺时间拉长，直接影响业务用人；超期岗位越多，业务侧催办越多',
            '建议': '逐个拆卡点：是简历不足、面试排期慢，还是用人部门决策慢，再对症处理',
            '需要老板': '如果这些岗位业务紧急，请确认是否加急（加预算 / 指定专人跟进）',
            '严重度': 3 if not over.empty else 2,
        })

    sub = risk[risk['主要问题'].str.contains('未招满', na=False)]
    if not sub.empty:
        items.append({
            '类型': '编制未招满',
            '标题': f"{len(sub)} 个已结束的岗位没有招满",
            '涉及岗位': '、'.join(f"{r['岗位']}（需求 {_fmt_num(r['需求'])} 人，到位 {_fmt_num(r['入职'])} 人）"
                                for _, r in sub.iterrows()),
            '影响': '编制缺口仍然存在，业务可能需要继续分摊工作量',
            '建议': '确认是继续补招还是先关闭岗位、调整用工方式',
            '需要老板': '确认继续招（追加预算/时间）还是关闭岗位',
            '严重度': 1,
        })

    sub = risk[risk['离职'] > 0]
    if not sub.empty:
        items.append({
            '类型': '新人留存',
            '标题': f"{len(sub)} 个岗位出现入职后离职（共 {_fmt_num(sub['离职'].sum())} 人）",
            '涉及岗位': '、'.join(f"{r['岗位']}（离职 {_fmt_num(r['离职'])} 人）" for _, r in sub.iterrows()),
            '影响': '招到又走，等于重复付出招聘成本，还会占用新的编制',
            '建议': '复盘入职引导与岗位预期管理，必要时回访离职人员',
            '需要老板': '是否需要推动用人部门做新人留任复盘',
            '严重度': 1,
        })

    return sorted(items, key=lambda x: -x['严重度'])


def _dashboard_decisions(d, ana, target_days):
    """第二屏：需要老板拍板 / 给资源的事。"""
    st.subheader('二、需要你拍板的事')
    items = _decision_items(d, ana, target_days)
    if not items:
        st.success('✅ 没有需要你决策的事项，招聘节奏正常。')
        return
    st.caption('这里不是岗位清单，而是把同类问题合并后，需要你决定或给资源的事（按影响大小排序）。')
    for i, it in enumerate(items, 1):
        with st.container(border=True):
            st.markdown(f"**{i}. 【{it['类型']}】{it['标题']}**")
            st.markdown(f"- 🎯 涉及岗位：{it['涉及岗位']}")
            st.markdown(f"- 📉 影响：{it['影响']}")
            st.markdown(f"- 🛠 建议：{it['建议']}")
            st.markdown(f"- 🙋 **需要你**：{it['需要老板']}")

    risk = ana[ana['问题数'] > 0].sort_values(['严重度'], ascending=False)
    with st.expander('📋 这些岗位的完整数据（点开核对）', expanded=False):
        cols = ['岗位', '部门', '状态', '周期(天)', '目标(天)', '同类中位(天)', '需求', '简历', '面试', '通过',
                'Offer', '入职', '简历→面试%', '面试→通过%', 'Offer→入职%', '主要卡点', '主要问题']
        show = risk[[c for c in cols if c in risk.columns]]
        st.dataframe(show, width='stretch', hide_index=True, height=min(120 + 36 * len(show), 420),
                     column_config={'周期(天)': st.column_config.NumberColumn('周期(天)', format='%.0f'),
                                    '目标(天)': st.column_config.NumberColumn('目标(天)', format='%.0f'),
                                    '同类中位(天)': st.column_config.NumberColumn('同类中位(天)', format='%.0f'),
                                    '简历→面试%': st.column_config.NumberColumn('简历→面试%', format='%.1f'),
                                    '面试→通过%': st.column_config.NumberColumn('面试→通过%', format='%.1f'),
                                    'Offer→入职%': st.column_config.NumberColumn('Offer→入职%', format='%.1f')})
        st.caption('异常判定：在招超过目标周期；已完成/暂停岗位周期超过目标或明显慢于同类中位；'
                   '简历→面试 <10%（简历≥10 份）、面试通过率 <30%（面试≥5 人）、Offer→入职 <80%、有离职、编制未招满。')


def _dashboard_progress(d, ana, target_days):
    """第三屏：进度 —— 在招岗位的空缺风险 + 各公司/部门达成排行。"""
    st.subheader('三、进度：谁快谁慢')

    st.markdown('**在招岗位跟踪（空缺风险）**')
    open_pos = ana[ana['状态'] == '招聘中'].copy()
    if open_pos.empty:
        st.success('当前没有在招岗位。')
    else:
        open_pos['进度(%)'] = (open_pos['周期(天)'] / target_days * 100).clip(lower=0, upper=100)
        open_pos['判断'] = open_pos.apply(
            lambda r: '⚠️ 已超目标周期' if r['周期(天)'] > target_days
            else ('注意：已慢于同类中位' if (r['同类中位(天)'] == r['同类中位(天)'] and r['周期(天)'] > r['同类中位(天)'])
                  else '正常'), axis=1)
        show = open_pos[['岗位', '部门', '周期(天)', '目标(天)', '进度(%)', '同类中位(天)', '需求', '入职', '判断']]
        show = show.rename(columns={'周期(天)': '已招天数', '目标(天)': '目标天数', '同类中位(天)': '同类中位'})
        st.dataframe(show, width='stretch', hide_index=True,
                     column_config={
                         '进度(%)': st.column_config.ProgressColumn('进度', min_value=0, max_value=100, format='%.0f'),
                         '已招天数': st.column_config.NumberColumn('已招天数', format='%.0f'),
                         '目标天数': st.column_config.NumberColumn('目标天数', format='%.0f'),
                         '同类中位': st.column_config.NumberColumn('同类中位', format='%.0f')})
        st.caption('进度 = 已招天数 ÷ 目标周期。超过目标周期或慢于同类中位的岗位，会出现在上面第二节的决策清单里。')

    st.markdown('**各公司 / 部门达成排行**')
    agg = d.groupby('department').agg(岗位数=('position', 'count'),
                                      平均周期=('duration_days', 'mean')).reset_index()
    dem = d[d['demand_max'].fillna(0) > 0].groupby('department').agg(
        需求=('demand_max', 'sum'), 到位=('onboarded', 'sum')).reset_index()
    agg = agg.merge(dem, on='department', how='left')
    agg['达成率%'] = agg.apply(
        lambda r: round(r['到位'] / r['需求'] * 100) if r['需求'] else float('nan'), axis=1)
    agg['平均周期'] = agg['平均周期'].round(1)
    agg['状态'] = agg['达成率%'].apply(
        lambda v: '🟢 达成' if (v == v and v >= 100) else ('🟡 接近' if (v == v and v >= 80) else '🔴 有缺口'))
    agg = agg.rename(columns={'department': '公司/部门'})[
        ['公司/部门', '岗位数', '需求', '到位', '达成率%', '平均周期', '状态']]
    st.dataframe(agg.sort_values('达成率%', ascending=False), width='stretch', hide_index=True)
    chart = agg[agg['达成率%'] == agg['达成率%']].sort_values('达成率%')
    if not chart.empty:
        bar = px.bar(chart, x='达成率%', y='公司/部门', orientation='h', text='达成率%',
                     color='达成率%', color_continuous_scale='RdYlGn', range_color=[0, max(120, chart['达成率%'].max())])
        bar.add_vline(x=100, line_dash='dash', line_color='#888', annotation_text='100% 达成')
        bar.update_layout(height=max(260, 34 * len(chart)), margin=dict(t=10, b=10, l=10, r=10),
                          showlegend=False, coloraxis_showscale=False,
                          paper_bgcolor='rgba(0,0,0,0)', font=dict(family='Microsoft YaHei, sans-serif'))
        st.plotly_chart(bar, width='stretch')
    st.caption('达成率 = 该公司/部门「填了招聘需求」岗位的到位人数 ÷ 需求人数；没填需求的岗位不计入达成率。')


def _dashboard_efficiency(d, ana, target_days):
    """第四屏：效率 —— 时间和简历花在哪、每环流失多少。"""
    st.subheader('四、效率与瓶颈')
    c1, c2 = st.columns([3, 2])
    with c1:
        st.plotly_chart(du.build_funnel_fig(d), width='stretch')
        res = _num(d['resumes'].fillna(0).sum())
        itv = _num(d['interviewed'].fillna(0).sum())
        off = _num(d['offer'].fillna(0).sum())
        onb = _num(d['onboarded'].fillna(0).sum())
        st.caption(f'累计：收简历 {res:.0f} → 进面试 {itv:.0f}（{_fmt_pct(_pct(itv, res))}）'
                   f' → 发 Offer {off:.0f}（{_fmt_pct(_pct(off, itv))}）'
                   f' → 入职 {onb:.0f}（{_fmt_pct(_pct(onb, off))}）；整体转化 {_fmt_pct(_pct(onb, res))}。')
    with c2:
        stages = pd.DataFrame([
            {'环节': '简历→面试', '流失率%': round(100 - _pct(itv, res), 1)},
            {'环节': '面试→Offer', '流失率%': round(100 - _pct(off, itv), 1)},
            {'环节': 'Offer→入职', '流失率%': round(100 - _pct(onb, off), 1)},
        ])
        stages = stages[stages['流失率%'] == stages['流失率%']]
        if not stages.empty:
            fig = px.bar(stages, x='流失率%', y='环节', orientation='h', text='流失率%',
                         color='流失率%', color_continuous_scale='Reds', range_color=[0, 100])
            fig.update_layout(height=260, margin=dict(t=10, b=10, l=10, r=10), showlegend=False,
                              coloraxis_showscale=False,
                              paper_bgcolor='rgba(0,0,0,0)', font=dict(family='Microsoft YaHei, sans-serif'))
            st.plotly_chart(fig, width='stretch')
            st.caption('每一环的流失率就是可以压缩的空间。')

    st.markdown('**周期对标：红 = 慢于同类中位或超过目标**')
    cyc = ana[(ana['状态'] == '完成招聘') & (ana['周期(天)'] == ana['周期(天)'])].copy()
    if cyc.empty:
        st.info('当前筛选下暂无完成招聘的岗位数据。')
        return
    cyc['是否异常'] = cyc['问题数'] > 0
    cyc = cyc.sort_values('周期(天)')
    bar2 = px.bar(cyc, x='周期(天)', y='岗位', orientation='h', text='周期(天)',
                  color='是否异常', color_discrete_map={True: '#e05c5c', False: '#4c8bf5'},
                  labels={'是否异常': '需关注'})
    bar2.add_vline(x=target_days, line_dash='dash', line_color='#888',
                   annotation_text=f'目标 {target_days:.0f} 天', annotation_position='top')
    avg = _num(cyc['周期(天)'].mean())
    if avg == avg:
        bar2.add_vline(x=avg, line_dash='dot', line_color='#2e9e6b',
                       annotation_text=f'平均 {avg:.1f} 天', annotation_position='bottom')
    bar2.update_layout(height=380, margin=dict(t=10, b=10, l=10, r=10), legend_title_text='',
                       paper_bgcolor='rgba(0,0,0,0)', font=dict(family='Microsoft YaHei, sans-serif'))
    st.plotly_chart(bar2, width='stretch')


def _dashboard_trend(d):
    """第五屏：变化 —— 和上次相比，好了还是差了。"""
    st.subheader('五、变化：跟上次比')
    snaps = _snapshots()
    if len(snaps) < 2:
        st.info('这是第一条记录。以后每次上传/修改数据（或每天第一次打开）都会自动记一条，'
                '这里就会出现“比上次好了还是差了”的对比和趋势。')
        return
    prev, cur = snaps[-2], snaps[-1]

    def dlt(key, unit='', digits=0):
        a, b = _num(cur.get(key)), _num(prev.get(key))
        if a != a or b != b:
            return None
        return f'{a - b:+.{digits}f}{unit}'

    c1, c2, c3, c4 = st.columns(4)
    c1.metric('到位人数', _fmt_num(cur.get('到位')), dlt('到位', ' 人'), delta_color='normal')
    c2.metric('在招岗位', _fmt_num(cur.get('在招')), dlt('在招', ' 个'), delta_color='inverse')
    c3.metric('平均周期', f"{_fmt_num(cur.get('平均周期'), 1)} 天", dlt('平均周期', ' 天', 1), delta_color='inverse')
    c4.metric('简历→入职转化', _fmt_pct(cur.get('转化率'), 2), dlt('转化率', ' pt', 2), delta_color='normal')
    st.caption(f'对比区间：{prev.get("date")} → {cur.get("date")}（每次数据变化自动记录一条快照，最多保留 {SNAP_MAX} 条）')

    hist = pd.DataFrame(snaps)
    keep = [c for c in ['date', '到位', '在招', '平均周期', '转化率'] if c in hist.columns]
    hist = hist[keep].tail(12)
    h1, h2 = st.columns(2)
    with h1:
        if '到位' in hist.columns:
            line1 = px.line(hist, x='date', y='到位', markers=True, title='到位人数走势')
            line1.update_layout(height=260, margin=dict(t=30, b=10, l=10, r=10),
                                paper_bgcolor='rgba(0,0,0,0)', font=dict(family='Microsoft YaHei, sans-serif'))
            st.plotly_chart(line1, width='stretch')
    with h2:
        if '平均周期' in hist.columns:
            line2 = px.line(hist, x='date', y='平均周期', markers=True, title='平均招聘周期走势（天）')
            line2.update_layout(height=260, margin=dict(t=30, b=10, l=10, r=10),
                                paper_bgcolor='rgba(0,0,0,0)', font=dict(family='Microsoft YaHei, sans-serif'))
            st.plotly_chart(line2, width='stretch')
    with st.expander('历次快照明细', expanded=False):
        st.dataframe(hist.iloc[::-1], width='stretch', hide_index=True)


def _dashboard_detail(d, ana, only_risk):
    """第六屏：岗位明细（备查）。"""
    detail = d.reset_index(drop=True)
    extra = ana[['周期(天)', '主要卡点', '问题数']].copy()
    extra['异常'] = extra['问题数'].apply(lambda n: f'⚠️ {int(n)} 项' if n else '')
    detail = pd.concat([detail, extra[['周期(天)', '主要卡点', '异常', '问题数']]], axis=1)
    if only_risk:
        detail = detail[detail['问题数'] > 0]

    with st.expander('六、岗位明细（备查，含异常标记）', expanded=False):
        if detail.empty:
            st.info('当前筛选范围内没有需要关注的岗位。')
            return
        disp = detail.drop(columns=['duration_days'], errors='ignore').copy()
        disp['招聘起'] = disp['start_date'].dt.date.astype(str).replace('NaT', '—')
        disp['招聘止'] = disp['end_date'].dt.date.astype(str).replace('NaT', '—')
        for col in du.NUMERIC_COLS:
            if col in disp.columns and col != 'duration_days':
                disp[col] = disp[col].round(1)
        en2cn = {
            'position': '岗位', 'department': '部门', 'category': '招聘类目', 'status': '当前状态',
            'resumes': '推送简历', 'invited': '邀约', 'interviewed': '面试', 'passed': '通过',
            'offer': 'Offer', 'onboarded': '入职', 'left': '离职', 'current_headcount': '现存',
            'demand': '招聘需求', 'daily_resumes': '日均简历',
        }
        for ec in du.extra_columns(d):
            en2cn[ec] = ec
        disp = disp.rename(columns=en2cn)
        order = ['岗位', '部门', '招聘类目', '当前状态', '周期(天)', '异常', '主要卡点', '招聘需求',
                 '推送简历', '邀约', '面试', '通过', 'Offer', '入职', '离职', '现存', '日均简历',
                 '招聘起', '招聘止']
        order += [ec for ec in du.extra_columns(d)]
        st.dataframe(disp[[c for c in order if c in disp.columns]], width='stretch', height=420, hide_index=True)


def page_dashboard():
    st.title('📊 招聘经营看板')
    st.caption('给老板看的一页：结论 → 需要拍板的事 → 进度 → 效率 → 变化 → 明细。'
               '上传新数据默认合并保留历史，也可在线增删改。')

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
        st.subheader('基准设置')
        target_days = st.number_input('目标招聘周期（天）', min_value=5, max_value=180, value=30, step=5,
                                      key='target_days',
                                      help='在招岗位超过这个天数就算超期；已完成岗位超过它也会被标出来。')
        slow_ratio = st.slider('比同类中位数慢多少倍算异常', 1.0, 3.0, 1.5, 0.1, key='slow_ratio',
                               help='已完成岗位的招聘周期超过「同类岗位中位数 × 这个倍数」就标为异常。')
        only_risk = st.checkbox('明细只看需要关注的岗位', value=False, key='only_risk')

    mask = (df['status'].isin(sel_status)) & (df['department'].isin(sel_dept)) & (df['category'].isin(sel_cat))
    d = df[mask].copy()
    if d.empty:
        st.warning('当前筛选条件下没有数据，请调整筛选。')
        return

    d, dropped = _drop_blank_rows(d)
    if dropped:
        st.caption(f'ℹ️ 已自动忽略 {dropped} 行没有岗位名称、也没有部门的空行/合计行（Excel 汇总行常见），'
                   '不计入统计；如需彻底删除，可在「数据管理 → 在线增删改」里操作。')
    if d.empty:
        st.warning('去掉空行后没有数据了，请检查上传的表格。')
        return

    target_days = float(target_days)
    ana = _position_analysis(d, target_days, float(slow_ratio))
    _save_snapshot(d)                                  # 每天第一次打开自动记一条快照
    _dashboard_summary(d, ana, target_days)            # 一、结论
    _dashboard_decisions(d, ana, target_days)          # 二、需要你拍板的事
    _dashboard_progress(d, ana, target_days)           # 三、进度：谁快谁慢
    _dashboard_efficiency(d, ana, target_days)         # 四、效率与瓶颈
    _dashboard_trend(d)                                # 五、变化：跟上次比
    _dashboard_detail(d, ana, only_risk)               # 六、明细（折叠）


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
        st.caption('备份文件包含：全部岗位（含已保存的 JD）、打分记录、数据看板里的数据。'
                   '换浏览器、换电脑，或应用重启后导入，都能一次性恢复。')
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
                n_pos, n_hist, n_dash = restore_payload(data)
                st.session_state['_reset_widgets'] = True
                st.session_state['flash'] = (f'已恢复 {n_pos} 个岗位、{n_hist} 条打分记录、'
                                             f'{n_dash} 行看板数据。')
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

def _boot_restore():
    """应用刚重启（临时数据可能被清空）时，主动提示并支持一键恢复备份。"""
    if st.session_state.get('_boot_checked'):
        return
    if not _looks_fresh_container():
        return
    st.session_state['_boot_checked'] = True
    st.warning('⚠️ 检测到应用刚重启或首次运行：之前临时保存的内容（岗位 JD、打分记录、'
               '数据看板里上传的数据）可能已被清空。'
               '如果你之前导出过备份 JSON，选上它、点「恢复全部数据」，就能一次全部找回。')
    up = st.file_uploader('选择备份 JSON 文件', type=['json'], key='boot_up')
    if st.button('恢复全部数据', key='boot_apply'):
        if up is None:
            st.warning('请先选择备份文件。')
        else:
            try:
                data = json.loads(up.getvalue().decode('utf-8'))
            except Exception as e:
                st.error(f'备份文件读取失败：{e}')
            else:
                n_pos, n_hist, n_dash = restore_payload(data)
                st.session_state['_reset_widgets'] = True
                st.session_state['flash'] = (f'已恢复 {n_pos} 个岗位、{n_hist} 条打分记录、'
                                             f'{n_dash} 行看板数据。')
                st.rerun()


pages = {'📊 数据看板': page_dashboard,
         '📄 岗位匹配': page_matching}
with st.sidebar:
    st.title('🎯 招聘 HR 工作台')
    page = st.radio('功能模块', list(pages.keys()), label_visibility='collapsed')
st.sidebar.caption('智能打分调用 DeepSeek API（填 Key 后可用）；本地规则打分不依赖网络与 AI。')

if st.session_state.get('flash'):
    st.success(st.session_state.pop('flash'))
_boot_restore()
pages[page]()
