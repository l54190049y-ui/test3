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
import plotly.graph_objects as go

import data_utils as du
import matching as mt

st.set_page_config(page_title='招聘 HR 工作台', page_icon='📊', layout='wide')

st.markdown("""
<style>
.block-container {padding-top: 1.4rem; max-width: 1500px;}
#MainMenu, footer {visibility: hidden;}
h1 {font-size: 1.9rem !important; letter-spacing: .2px;}
h3 {font-size: 1.15rem !important; margin-top: .3rem;}
[data-testid="stMetric"] {
    background: rgba(127, 127, 127, 0.06);
    border: 1px solid rgba(127, 127, 127, 0.22);
    border-radius: 14px; padding: 14px 16px;
}
div[data-testid="stMetricLabel"] {font-size: 0.82rem; opacity: .75;}
div[data-testid="stMetricValue"] {font-size: 1.5rem; font-weight: 650;}
[data-testid="stMetricDelta"] {font-size: 0.78rem;}
.stTabs [data-baseweb="tab-list"] {gap: 6px; border-bottom: 1px solid rgba(127,127,127,.2);}
.stTabs [data-baseweb="tab"] {height: 40px; padding: 0 16px; border-radius: 10px 10px 0 0;}
.stTabs [aria-selected="true"] {background: rgba(76,139,245,.12); font-weight: 600;}
[data-testid="stExpander"] {border: 1px solid rgba(127,127,127,.2); border-radius: 12px;}
section[data-testid="stSidebar"] {border-right: 1px solid rgba(127,127,127,.2);}
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


def _storage_badge():
    """把存储状态收进侧边栏，主页面只放内容。"""
    ok, _ = storage_status()
    with st.sidebar:
        if ok:
            st.caption('🟢 数据已连接云端数据库，永久保存')
        else:
            st.caption('🟡 临时存储：应用重启后需导入备份（页面底部有开启永久保存的步骤）')
        for w in st.session_state.pop('kv_errors', [])[:2]:
            st.caption('⚠️ ' + w)


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


def _clean_any(df):
    """统一清洗入口：兼容「公司」列名、去重列名，任何上传文件都不会把页面搞崩。"""
    try:
        work = df.copy()
    except Exception:
        work = df
    try:
        if '公司' in list(work.columns) and '部门' not in list(work.columns):
            work = work.rename(columns={'公司': '部门'})
        clean = du.clean_raw(work)
    except Exception:
        clean = work
    try:
        if clean.columns.duplicated().any():
            clean = clean.loc[:, ~clean.columns.duplicated()]
    except Exception:
        pass
    return clean


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
        try:
            return _clean_any(pd.DataFrame(rows))
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
        _save_snapshot(_clean_any(df), force=True)
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
    """不再预置任何 JD：岗位库完全由使用者自己新增 / 上传。"""
    return []


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
    """岗位库完全来自持久层（使用者自己新增 / 上传的 JD），不再预置内置岗位。"""
    deleted_list = sorted(set(kv_get(DEL_KEY, []) or []))
    st.session_state['deleted_builtins'] = deleted_list
    deleted = set(deleted_list)
    stored = kv_get(POS_KEY, None)
    if isinstance(stored, list) and stored:
        positions = [_norm_pos(p) for p in stored]
    else:
        positions = []
    # 清掉旧版本内置的 JD，只保留使用者自己加的
    kept = [p for p in positions if p.get('source') != '内置']
    need_save = len(kept) != len(positions)
    positions = kept
    have = {p['name'] for p in positions}
    for p in _legacy_custom_positions():
        if p['name'] not in have:
            positions.append(p)
            have.add(p['name'])
            need_save = True
    out = [p for p in positions if p.get('source') != '内置']
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
                try:
                    if '公司' in list(new_raw.columns) and '部门' not in list(new_raw.columns):
                        new_raw = new_raw.rename(columns={'公司': '部门'})
                    if mode == '覆盖全部':
                        du.save_store(_clean_any(new_raw))
                        st.success(f'已覆盖：共 {len(new_raw)} 行数据。')
                    else:
                        merged, added, updated = du.merge_dataframes(du.load_current_data(), new_raw)
                        du.save_store(_clean_any(merged))
                        st.success(f'合并完成：新增 {added} 行，更新 {updated} 行，现有共 {len(merged)} 行。'
                                   f'（按 岗位+部门+招聘起 匹配，同一条目以新文件为准）')
                    st.rerun()
                except Exception as e:
                    st.error(f'这份文件没能读进来：{e}。请检查表头是否包含「岗位 / 公司(或部门) / 当前状态」等列，'
                             '或把文件另存为 .xlsx 后再试。')

    st.markdown('**② 在线增删改数据（加行/删行/改单元格后点保存）**')
    editor_df = _clean_any(du.load_editor_data())
    display_df = du.to_display(editor_df)
    if display_df.columns.duplicated().any():
        display_df = display_df.loc[:, ~display_df.columns.duplicated()]
    cfg = {k: v for k, v in _editor_column_config(editor_df).items() if k in list(display_df.columns)}
    try:
        edited = st.data_editor(display_df, num_rows='dynamic', key='data_editor', column_config=cfg,
                                width='stretch', height=380, hide_index=True)
    except Exception as e:
        st.caption(f'表格编辑暂不可用（{e}），已改为只读预览；可先下载修正后再上传。')
        st.dataframe(display_df, width='stretch', height=380, hide_index=True)
        edited = display_df
    if st.button('💾 保存编辑', type='primary', key='save_edit'):
        back = du.from_display(edited)
        du.save_store(_clean_any(_fix_editor_types(back)))
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
    # 同类岗位（同招聘类目）的周期基准：优先用该类目已完成岗位的中位数，样本太少才退回整体中位/目标值
    base_by_cat = {}
    for cat, g in (done.groupby('category') if not done.empty else []):
        gv = g['duration_days'].dropna()
        if len(gv) >= 1:
            mm = _num(gv.median())
            if mm == mm:
                base_by_cat[cat] = mm
    fallback_base = med_all if med_all == med_all else float(target_days)

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
        std = base_by_cat.get(r.get('category'), fallback_base)

        issues, hints, score, flagged = [], [], 0, []
        if hiring and cycle == cycle and cycle > std:
            issues.append(f'在招 {cycle:.0f} 天，已超过同类基准 {std:.0f} 天')
            hints.append('先判断是“没人投”还是“投了筛不出来”：看简历量、面试转化和用人部门反馈')
            score += 3
        if (not hiring) and status in ('完成招聘', '暂停') and cycle == cycle:
            slow_rel = cycle > std * ratio and cycle > std + 3
            if slow_rel:
                cmp = f'（同类基准 {std:.0f} 天）'
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
            '基准(天)': round(std, 1),
            '需求': demand_max, '简历': resumes, '面试': interviewed, 'Offer': offer, '入职': onboarded,
            '通过': passed, '离职': left, '现存': _num(r.get('current_headcount')),
            '招聘起': r.get('start_date'), '招聘止': r.get('end_date'),
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
    left = float(d['left'].fillna(0).sum())
    current = float(d['current_headcount'].fillna(0).sum())
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
        '离职': round(left, 1),
        '现存': round(current, 1),
        '留存率': round((onboarded - left) / onboarded * 100, 1) if onboarded else None,
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
    for k in ('需求', '到位', '在招', '平均周期', '转化率', 'Offer接受率', '留存率'):
        a, b = _num(m.get(k)), _num(prev.get(k))
        out[k] = (a - b) if (a == a and b == b) else None
    return out


def _delta_txt(dv, key, unit='', digits=0):
    if not dv or dv.get(key) is None:
        return None
    return f'{dv[key]:+.{digits}f}{unit}（较 {dv["_date"]}）'


def _open_positions_text(ana):
    """在招岗位名单（回答“在招的到底是哪几个岗位”）。"""
    op = ana[ana['状态'] == '招聘中']
    if op.empty:
        return '当前没有在招岗位'
    parts = []
    for _, r in op.sort_values('周期(天)', ascending=False).iterrows():
        days = f"{_fmt_num(r['周期(天)'])} 天" if r['周期(天)'] == r['周期(天)'] else '起始日未填'
        gap = f"缺口 {_fmt_num(max(_num(r['需求'], 0) - _num(r['入职'], 0), 0))} 人" if r['需求'] == r['需求'] else ''
        parts.append(f"{r['岗位']}（{r['部门']} · 已招 {days}{' · ' + gap if gap else ''}）")
    return '、'.join(parts)


def _open_issues(r, target_days):
    """只给“在招岗位”做标注（已完成的岗位是结果，不进问题区）。"""
    if str(r.get('状态', '')) != '招聘中':
        return []
    out = []
    std = _num(r.get('基准(天)', target_days), target_days)
    days = _num(r['周期(天)'])
    if days == days and days > std:
        out.append(f'已超同类基准 {days - std:.0f} 天')
    res, itv = _num(r['简历'], 0), _num(r['面试'], 0)
    if res <= 0:
        out.append('尚未收到简历')
    elif itv <= 0:
        out.append(f'已收 {res:.0f} 份简历，未安排面试')
    return out


def _dashboard_summary(d, ana, target_days):
    """一、总览：关键指标 + 在招岗位进度（图 + 表）。"""
    m = _headline_metrics(d)
    dv = _delta_vs_prev(m)
    rate, cyc = _num(m['到位率']), _num(m['平均周期'])
    conv, retain = _num(m['转化率']), _num(m['留存率'])

    op = ana[ana['状态'] == '招聘中'].copy()
    op['标注'] = op.apply(lambda r: '；'.join(_open_issues(r, target_days)), axis=1)
    overdue = int((op['标注'].str.contains('超目标', na=False)).sum())
    flagged = int((op['标注'].str.len() > 0).sum())

    lv_rate = 'green' if rate == rate and rate >= 100 else ('amber' if rate == rate and rate >= 80 else 'red')
    lv_open = 'green' if (overdue == 0 and flagged == 0) else ('amber' if overdue <= 2 and flagged <= 3 else 'red')
    lv_cyc = 'green' if (cyc != cyc or cyc <= target_days) else ('amber' if cyc <= target_days * 1.3 else 'red')
    lv_retain = 'green' if (retain != retain or retain >= 90) else ('amber' if retain >= 80 else 'red')

    st.subheader('关键指标')
    snaps_now = _snapshots()
    updated = snaps_now[-1].get('saved_at', '—') if snaps_now else '—'
    st.caption(f'数据截至 {m["date"]}　·　岗位 {len(d)} 个（在招 {m["在招"]} · 完成 '
               f"{int((d['status'] == '完成招聘').sum())} · 暂停 {int((d['status'] == '暂停').sum())}）"
               f'　·　最近更新时间 {updated}')

    gap = max(_num(m['需求']) - _num(m['到位']), 0)
    net = _num(m['入职']) - _num(m['离职'])
    c1 = st.columns(4)
    c1[0].metric('总岗位数', f'{len(d)} 个',
                 f"在招 {int((d['status'] == '招聘中').sum())} · 完成 {int((d['status'] == '完成招聘').sum())}"
                 + (f" · 暂停 {int((d['status'] == '暂停').sum())}" if int((d['status'] == '暂停').sum()) else ''),
                 delta_color='off')
    c1[1].metric(f'{_health(lv_open)} 招聘中', f"{m['在招']} 个",
                 (f'超期 {overdue} · 待跟进 {flagged}' if flagged else '进度正常'),
                 delta_color='inverse' if flagged else 'off')
    c1[2].metric('总需求人数', f"{_fmt_num(m['需求'])} 人", f'缺口 {_fmt_num(gap)} 人', delta_color='off')
    done_all = ana[ana['状态'] == '完成招聘']
    ok_mask = done_all['周期(天)'] == done_all['周期(天)']
    slow_mask = ok_mask & (done_all['周期(天)'] > target_days) & \
        (done_all['周期(天)'] > done_all['同类中位(天)'] * 1.2)
    slow_n = int(slow_mask.sum())
    c1[3].metric(f'{_health(lv_open)} 周期偏长岗位', f'{slow_n} 个',
                 f'按岗位对标（目标 {target_days:.0f} 天 / 同类中位）', delta_color='inverse' if slow_n else 'off')

    c2 = st.columns(4)
    c2[0].metric('总入职人数', f"{_fmt_num(m['入职'])} 人",
                 f'到位率 {_fmt_pct(rate, 0)}' if rate == rate else None, delta_color='off')
    c2[1].metric('总离职人数', f"{_fmt_num(m['离职'])} 人", delta_color='off')
    c2[2].metric(f'{_health(lv_retain)} 净留存', f'{_fmt_num(net)} 人',
                 _delta_txt(dv, '留存率', ' pt', 1) or f'留存率 {_fmt_pct(retain, 0)}',
                 delta_color='inverse')
    c2[3].metric('简历→入职转化', _fmt_pct(conv) if conv == conv else '—',
                 f"收到 {_fmt_num(m['简历'])} 份简历", delta_color='off')

    st.markdown('**岗位类型基准**（同招聘类目的周期参考，来自该类已完成岗位）')
    type_rows = []
    for cat, g in d.groupby('category'):
        gd = g[(g['status'] == '完成招聘') & g['duration_days'].notna() & (g['duration_days'] > 0)]
        std = _num(gd['duration_days'].median()) if len(gd) >= 1 else _num(ana['基准(天)'].median())
        type_rows.append({
            '招聘类目': cat, '岗位数': int(len(g)), '在招': int((g['status'] == '招聘中').sum()),
            '已完成': int(len(gd)), '基准周期(中位)': round(std, 1) if std == std else None,
            '已完成的区间': (f"{_fmt_num(gd['duration_days'].min())}–{_fmt_num(gd['duration_days'].max())} 天"
                             if len(gd) else '—')})
    st.dataframe(pd.DataFrame(type_rows), width='stretch', hide_index=True)

    st.markdown('**在招岗位进度**（灰段 = 距同类基准还剩多少天）')
    if op.empty:
        st.success('当前没有在招岗位。')
    else:
        op = op.sort_values('周期(天)', ascending=True).copy()
        op['距目标剩余'] = (target_days - op['周期(天)']).clip(lower=0)
        op['颜色'] = op['标注'].apply(lambda s: '#e05c5c' if s else '#4c8bf5')
        op['招聘起txt'] = pd.to_datetime(op['招聘起'], errors='coerce').dt.date.astype(str).replace('NaT', '—')
        card = pd.DataFrame({
            '公司': op['部门'].astype(str),
            '招聘起': op['招聘起txt'],
            '需求': op['需求'].apply(lambda v: _fmt_num(v)),
            '已入职': op['入职'].apply(lambda v: _fmt_num(v)),
            '简历': op['简历'].apply(lambda v: _fmt_num(v)),
            '面试': op['面试'].apply(lambda v: _fmt_num(v)),
            '状态说明': op['标注'].replace('', '正常'),
        }).values
        fig = go.Figure()
        fig.add_trace(go.Bar(y=op['岗位'], x=op['距目标剩余'], orientation='h',
                             name=f'距目标剩余（目标 {target_days:.0f} 天）', marker_color='#e8eef7',
                             hoverinfo='skip'))
        fig.add_trace(go.Bar(y=op['岗位'], x=op['周期(天)'], orientation='h', name='已招天数',
                             marker_color=op['颜色'], customdata=card,
                             hovertemplate=('<b>%{y}</b><br>公司：%{customdata[0]}　招聘起：%{customdata[1]}'
                                            '<br>已招：%{x} 天（目标 ' + f'{target_days:.0f}' + ' 天）'
                                            '<br>需求 %{customdata[2]} 人 · 已入职 %{customdata[3]} 人'
                                            '<br>简历 %{customdata[4]} 份 · 面试 %{customdata[5]} 人'
                                            '<br>%{customdata[6]}<extra></extra>'),
                             text=[f"{_fmt_num(v)} 天" for v in op['周期(天)']], textposition='inside'))
        fig.update_layout(barmode='stack', height=max(240, 46 * len(op)), margin=dict(t=10, b=10, l=10, r=10),
                          legend=dict(orientation='h', y=1.12), xaxis_title='天',
                          paper_bgcolor='rgba(0,0,0,0)', font=dict(family='Microsoft YaHei, sans-serif'))
        st.plotly_chart(fig, width='stretch')
        src_open = d[d['status'] == '招聘中'].copy()
        src_open['_k'] = src_open['position'].astype(str) + '||' + src_open['department'].astype(str)
        opx = op.copy()
        opx['_k'] = opx['岗位'].astype(str) + '||' + opx['部门'].astype(str)
        opx['缺口'] = opx.apply(
            lambda r: max(_num(r['需求'], 0) - _num(r['入职'], 0), 0) if r['需求'] == r['需求'] else float('nan'),
            axis=1)
        opx['距离(天)'] = opx['基准(天)'] - opx['周期(天)']
        merged = src_open.merge(opx[['_k', '周期(天)', '基准(天)', '距离(天)', '缺口', '标注']], on='_k', how='left')
        _editable_table(merged, key='edit_open',
                        extra_cols={'类目': merged['category'], '已招天数': merged['周期(天)'],
                                    '同类基准': merged['基准(天)'], '距离(天)': merged['距离(天)'],
                                    '缺口': merged['缺口'], '标注': merged['标注'].fillna('')})
        st.caption(f'在招 {len(op)} 个：超期 {overdue} 个、待跟进 {flagged} 个。'
                   '表格可以直接改（状态/招聘起/招聘止/需求/简历/面试/通过/Offer/入职/离职/现存），'
                   '改完点「💾 保存表格修改」即可，刷新也在。')


def _dashboard_monthly(d):
    """按月看新增岗位与完成岗位（周期不做跨岗位平均，放到「转化与周期」里按岗位看）。"""
    st.subheader('按月趋势')
    tmp = d.copy()
    tmp['开始月'] = pd.to_datetime(tmp['start_date'], errors='coerce').dt.to_period('M')
    tmp['结束月'] = pd.to_datetime(tmp['end_date'], errors='coerce').dt.to_period('M')
    new = tmp.dropna(subset=['开始月']).groupby('开始月').size()
    don = tmp.dropna(subset=['结束月'])
    don = don[don['status'] == '完成招聘']
    done_cnt = don.groupby('结束月').size()
    done_cyc = don.groupby('结束月')['duration_days'].mean()
    new_names = tmp[tmp['开始月'].notna()].groupby(tmp['开始月'].astype(str).where(tmp['开始月'].notna()))[
        'position'].apply(lambda s: '、'.join(str(x) for x in s))
    done_names = don.groupby(don['结束月'].astype(str))['position'].apply(lambda s: '、'.join(str(x) for x in s))
    months = sorted(str(m) for m in (set(new.index) | set(done_cnt.index)))
    if not months:
        st.info('数据里没有可用的「招聘起 / 招聘止」日期，暂时画不出月度趋势。')
        return
    hist = pd.DataFrame({'月份': months})
    hist['新增岗位'] = hist['月份'].map({str(k): v for k, v in new.items()}).fillna(0).astype(int)
    hist['完成岗位'] = hist['月份'].map({str(k): v for k, v in done_cnt.items()}).fillna(0).astype(int)
    hist['新增名单'] = hist['月份'].map({str(k): v for k, v in new_names.items()}).fillna('—')
    hist['完成名单'] = hist['月份'].map({str(k): v for k, v in done_names.items()}).fillna('—')

    fig = go.Figure()
    fig.add_trace(go.Bar(x=hist['月份'], y=hist['新增岗位'], name='新增岗位（按招聘起）', marker_color='#4c8bf5',
                         customdata=hist[['新增名单']].values,
                         hovertemplate='<b>%{x} 新增 %{y} 个岗位</b><br>%{customdata[0]}<extra></extra>'))
    fig.add_trace(go.Bar(x=hist['月份'], y=hist['完成岗位'], name='完成岗位（按招聘止）', marker_color='#2e9e6b',
                         customdata=hist[['完成名单']].values,
                         hovertemplate='<b>%{x} 完成 %{y} 个岗位</b><br>%{customdata[0]}<extra></extra>'))
    fig.update_layout(barmode='group', height=360, margin=dict(t=30, b=10, l=10, r=10),
                      yaxis=dict(title='岗位数'),
                      legend=dict(orientation='h', y=1.15),
                      paper_bgcolor='rgba(0,0,0,0)', font=dict(family='Microsoft YaHei, sans-serif'))
    st.plotly_chart(fig, width='stretch')
    st.dataframe(hist, width='stretch', hide_index=True)
    st.caption('「新增岗位」按招聘起算，「完成岗位」按招聘止算。周期不在这里做平均——各岗位周期不同，'
               '要看去「转化与周期」里按岗位看。')


def _dashboard_progress(d, ana, target_days):
    """三、组织结构：按公司/类目看分布，按公司看到位与缺口。"""
    st.subheader('公司 · 类目 · 达成')

    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown('**各公司岗位状态分布**')
        cnt = d.groupby(['department', 'status']).size().reset_index(name='岗位数')
        cnt = cnt.rename(columns={'department': '公司/部门', 'status': '状态'})
        names = d.groupby(['department', 'status'])['position'].apply(
            lambda s: '、'.join(str(x) for x in s)).reset_index(name='岗位名单')
        names = names.rename(columns={'department': '公司/部门', 'status': '状态'})
        cnt = cnt.merge(names, on=['公司/部门', '状态'], how='left')
        fig = px.bar(cnt, x='公司/部门', y='岗位数', color='状态', barmode='stack', text='岗位数',
                     color_discrete_map={s: du.status_label(s) for s in du.STATUS_ORDER},
                     custom_data=['岗位名单'])
        fig.update_traces(hovertemplate='<b>%{x} · %{fullData.name}</b><br>%{y} 个岗位<br>%{customdata[0]}'
                                        '<extra></extra>')
        fig.update_layout(height=340, margin=dict(t=10, b=10, l=10, r=10), legend=dict(orientation='h', y=1.15),
                          paper_bgcolor='rgba(0,0,0,0)', font=dict(family='Microsoft YaHei, sans-serif'))
        st.plotly_chart(fig, width='stretch')
    with c2:
        st.markdown('**招聘类目分布**')
        cat = d['category'].fillna('未分类').value_counts()
        cat_names = d.groupby(d['category'].fillna('未分类'))['position'].apply(
            lambda s: '、'.join(str(x) for x in s))
        catdf = pd.DataFrame({'类目': cat.index, '岗位数': cat.values,
                              '岗位名单': [cat_names.get(k, '—') for k in cat.index]})
        pie = px.pie(catdf, values='岗位数', names='类目', hole=0.45, custom_data=['岗位名单'])
        pie.update_traces(textinfo='label+value',
                          hovertemplate='<b>%{label}</b><br>%{value} 个岗位<br>%{customdata[0]}<extra></extra>')
        pie.update_layout(height=340, margin=dict(t=10, b=10, l=10, r=10), showlegend=False,
                          paper_bgcolor='rgba(0,0,0,0)', font=dict(family='Microsoft YaHei, sans-serif'))
        st.plotly_chart(pie, width='stretch')

    st.markdown('**各公司到位与缺口**')
    dd = d[d['demand_max'].fillna(0) > 0]
    if dd.empty:
        st.info('没有岗位填写「招聘需求」，无法计算到位与缺口。')
        return
    agg = dd.groupby('department').agg(需求=('demand_max', 'sum'), 到位=('onboarded', 'sum')).reset_index()
    agg['缺口'] = (agg['需求'] - agg['到位']).clip(lower=0)
    agg['达成率%'] = (agg['到位'] / agg['需求'] * 100).round(0)
    agg = agg.rename(columns={'department': '公司/部门'}).sort_values('达成率%', ascending=False)
    gap_detail = {}
    for dept, g in dd.groupby('department'):
        miss = g[g['onboarded'].fillna(0) < g['demand_max'].fillna(0)]
        gap_detail[dept] = '、'.join(
            f"{r['position']}（{_fmt_num(r['demand_max'])}→{_fmt_num(r['onboarded'])}）"
            for _, r in miss.iterrows()) or '—'
    agg['缺口岗位'] = agg['公司/部门'].map(gap_detail).fillna('—')
    st.dataframe(agg, width='stretch', hide_index=True)
    bar = go.Figure()
    bar.add_trace(go.Bar(y=agg['公司/部门'], x=agg['到位'], orientation='h', name='已到位', marker_color='#2e9e6b',
                         text=agg['到位'], textposition='inside',
                         hovertemplate='<b>%{y}</b><br>已到位 %{x} 人<extra></extra>'))
    bar.add_trace(go.Bar(y=agg['公司/部门'], x=agg['缺口'], orientation='h', name='缺口', marker_color='#e05c5c',
                         text=agg['缺口'], textposition='inside', customdata=agg[['缺口岗位']].values,
                         hovertemplate='<b>%{y}</b><br>缺口 %{x} 人<br>未到位：%{customdata[0]}<extra></extra>'))
    bar.update_layout(barmode='stack', height=max(260, 40 * len(agg)), margin=dict(t=10, b=10, l=10, r=10),
                      legend=dict(orientation='h', y=1.15), xaxis_title='人数',
                      paper_bgcolor='rgba(0,0,0,0)', font=dict(family='Microsoft YaHei, sans-serif'))
    st.plotly_chart(bar, width='stretch')
    st.caption('达成率 = 该公司「填了招聘需求」岗位的到位人数 ÷ 需求人数；缺口 = 需求 − 到位；没填需求的岗位不计入。')


def _dashboard_efficiency(d, ana, target_days):
    """四、转化漏斗（整体 + 各公司）；五、周期分析（只统计已完成岗位）。"""
    st.subheader('转化漏斗')
    base = d[d['resumes'].notna()].copy()
    if base.empty:
        st.info('没有岗位填写「推送方简历数」，无法计算漏斗。')
        return
    res = _num(base['resumes'].fillna(0).sum())
    itv = _num(base['interviewed'].fillna(0).sum())
    pss = _num(base['passed'].fillna(0).sum())
    off = _num(base['offer'].fillna(0).sum())
    onb = _num(base['onboarded'].fillna(0).sum())
    c1, c2 = st.columns([3, 2])
    with c1:
        def _top_by(col, n=6):
            part = base[['position', col]].copy()
            part[col] = pd.to_numeric(part[col], errors='coerce').fillna(0)
            part = part[part[col] > 0].sort_values(col, ascending=False)
            return '、'.join(str(x) for x in part['position'].head(n)) or '—'

        funnel = go.Figure(go.Funnel(
            y=['收简历', '进面试', '面试通过', '发 Offer', '入职'],
            x=[res, itv, pss, off, onb],
            textinfo='value+percent initial',
            marker={'color': ['#4c8bf5', '#36b37e', '#ffb020', '#ff7452', '#8f6ef0']},
            customdata=[[_top_by('resumes')], [_top_by('interviewed')], [_top_by('passed')],
                        [_top_by('offer')], [_top_by('onboarded')]],
            hovertemplate='<b>%{y}</b><br>%{x:.0f} 人（占初始 %{percentInitial}）'
                          '<br>主要岗位：%{customdata[0]}<extra></extra>'))
        funnel.update_layout(margin=dict(l=10, r=10, t=20, b=10), height=340,
                             paper_bgcolor='rgba(0,0,0,0)', font=dict(family='Microsoft YaHei, sans-serif'))
        st.plotly_chart(funnel, width='stretch')
        st.caption(f'累计：收简历 {res:.0f} → 进面试 {itv:.0f}（{_fmt_pct(_pct(itv, res))}）'
                   f' → 面试通过 {pss:.0f}（{_fmt_pct(_pct(pss, itv))}）'
                   f' → 发 Offer {off:.0f}（{_fmt_pct(_pct(off, pss))}）'
                   f' → 入职 {onb:.0f}（{_fmt_pct(_pct(onb, off))}）；整体转化 {_fmt_pct(_pct(onb, res))}。'
                   '口径：只统计填写了「推送方简历数」的岗位；只填了入职、没填简历/Offer 的批量入职岗位'
                   '（例如分拣员）不计入漏斗，但仍计入总览的入职与留存。')
    with c2:
        stages = pd.DataFrame([
            {'环节': '简历→面试', '转化率%': round(_pct(itv, res), 1)},
            {'环节': '面试→Offer', '转化率%': round(_pct(off, itv), 1)},
            {'环节': 'Offer→入职', '转化率%': round(_pct(onb, off), 1)},
        ])
        stages = stages[stages['转化率%'] == stages['转化率%']]
        if not stages.empty:
            fig = px.bar(stages, x='转化率%', y='环节', orientation='h', text='转化率%',
                         color='转化率%', color_continuous_scale='Blues', range_color=[0, 100])
            fig.update_layout(height=260, margin=dict(t=10, b=10, l=10, r=10), showlegend=False,
                              coloraxis_showscale=False, xaxis_range=[0, 100],
                              paper_bgcolor='rgba(0,0,0,0)', font=dict(family='Microsoft YaHei, sans-serif'))
            st.plotly_chart(fig, width='stretch')

    st.markdown('**各公司转化对比**')
    rows = []
    for dept, g in base.groupby('department'):
        r = _num(g['resumes'].fillna(0).sum())
        i = _num(g['interviewed'].fillna(0).sum())
        p = _num(g['passed'].fillna(0).sum())
        o = _num(g['offer'].fillna(0).sum())
        n = _num(g['onboarded'].fillna(0).sum())
        rows.append({'公司/部门': dept, '收简历': r, '进面试': i, '发Offer': o, '入职': n,
                     '简历→面试%': round(_pct(i, r), 1) if _pct(i, r) == _pct(i, r) else None,
                     '面试→Offer%': round(_pct(o, p), 1) if _pct(o, p) == _pct(o, p) else None,
                     'Offer→入职%': round(_pct(n, o), 1) if _pct(n, o) == _pct(n, o) else None})
    st.dataframe(pd.DataFrame(rows).sort_values('收简历', ascending=False),
                 width='stretch', hide_index=True)

    st.subheader('周期分析')
    done = ana[(ana['状态'] == '完成招聘') & (ana['周期(天)'] == ana['周期(天)'])].copy()
    if done.empty:
        st.info('暂无已完成岗位的周期数据。')
        return
    cmp = done[['岗位', '类目', '部门', '周期(天)', '同类中位(天)', '目标(天)', '需求', '入职']].copy()
    cmp['对标结果'] = cmp.apply(
        lambda r: ('偏长' if (r['周期(天)'] > r['目标(天)']) or
                   (r['同类中位(天)'] == r['同类中位(天)'] and r['周期(天)'] > r['同类中位(天)'] * 1.2)
                   else '正常'), axis=1)
    cmp = cmp.sort_values('周期(天)', ascending=False)
    st.markdown('**每个岗位的周期 vs 自己的目标 / 同类中位**')
    st.dataframe(cmp, width='stretch', hide_index=True, height=min(140 + 34 * len(cmp), 420))

    cyc = done.sort_values('周期(天)', ascending=True)
    bar2 = px.bar(cyc, x='周期(天)', y='岗位', orientation='h', text='周期(天)', color='类目',
                  custom_data=['类目', '同类中位(天)', '部门', '需求', '入职'])
    bar2.update_traces(hovertemplate='<b>%{y}</b>（%{customdata[2]}）<br>周期 %{x:.0f} 天'
                                     '<br>类目 %{customdata[0]} · 同类中位 %{customdata[1]} 天'
                                     '<br>需求 %{customdata[3]} · 入职 %{customdata[4]}<extra></extra>')
    bar2.add_vline(x=target_days, line_dash='dash', line_color='#888',
                   annotation_text=f'目标 {target_days:.0f} 天', annotation_position='top')
    bar2.update_layout(height=380, margin=dict(t=30, b=10, l=10, r=10), legend_title_text='',
                       paper_bgcolor='rgba(0,0,0,0)', font=dict(family='Microsoft YaHei, sans-serif'))
    st.plotly_chart(bar2, width='stretch')
    st.caption('每个岗位跟自己的目标周期、以及同类岗位（同招聘类目）的中位周期对比，不做跨岗位平均。'
               '颜色按招聘类目区分。')


def _dashboard_retention(d, ana):
    """六、人员留存：入职 / 离职 / 现存，以及跟上次比的变化。"""
    st.subheader('入职 · 离职 · 留存')
    ret = d.groupby('department').agg(入职=('onboarded', 'sum'), 离职=('left', 'sum'),
                                      现存=('current_headcount', 'sum')).reset_index()
    ret['留存%'] = ret.apply(
        lambda r: round((r['入职'] - r['离职']) / r['入职'] * 100, 1) if r['入职'] else float('nan'), axis=1)
    ret = ret.rename(columns={'department': '公司/部门'}).sort_values('入职', ascending=False)

    c1, c2 = st.columns([2, 3])
    with c1:
        st.dataframe(ret, width='stretch', hide_index=True)
        snap = _snapshots()
        if len(snap) >= 2:
            prev, cur = snap[-2], snap[-1]
            dv = _num(cur.get('离职')) - _num(prev.get('离职'))
            if dv == dv:
                st.caption(f'离职人数较上次（{prev.get("date")}）：{dv:+.0f} 人')
    with c2:
        chart = ret[ret['入职'].fillna(0) > 0]
        if chart.empty:
            st.info('暂无可统计的入职数据。')
        else:
            left_names = {}
            for dept, g in d.groupby('department'):
                lf = g[g['left'].fillna(0) > 0]
                left_names[dept] = '、'.join(
                    f"{r['position']}（离职 {_fmt_num(r['left'])}）" for _, r in lf.iterrows()) or '—'
            chart = chart.copy()
            chart['离职岗位'] = chart['公司/部门'].map(left_names).fillna('—')
            fig = go.Figure()
            fig.add_trace(go.Bar(x=chart['公司/部门'], y=chart['入职'], name='入职', marker_color='#2e9e6b',
                                 hovertemplate='<b>%{x}</b><br>入职 %{y} 人<extra></extra>'))
            fig.add_trace(go.Bar(x=chart['公司/部门'], y=chart['离职'], name='离职', marker_color='#e05c5c',
                                 customdata=chart[['离职岗位']].values,
                                 hovertemplate='<b>%{x}</b><br>离职 %{y} 人<br>%{customdata[0]}<extra></extra>'))
            fig.add_trace(go.Scatter(x=chart['公司/部门'], y=chart['现存'], name='现存', mode='lines+markers',
                                     line=dict(color='#4c8bf5', width=3),
                                     hovertemplate='<b>%{x}</b><br>现存 %{y} 人<extra></extra>'))
            fig.update_layout(barmode='group', height=340, margin=dict(t=30, b=10, l=10, r=10),
                              legend=dict(orientation='h', y=1.15), yaxis_title='人数',
                              paper_bgcolor='rgba(0,0,0,0)', font=dict(family='Microsoft YaHei, sans-serif'))
            st.plotly_chart(fig, width='stretch')

    left_pos = ana[ana['离职'] > 0]
    if left_pos.empty:
        st.caption('当前没有岗位出现入职后离职。留存% =（入职 − 离职）÷ 入职；现存人数为最后一次记录的在场人数。')
    else:
        st.markdown('**出现离职的岗位**：'
                    + '、'.join(f"{r['岗位']}（入职 {_fmt_num(r['入职'])} · 离职 {_fmt_num(r['离职'])} · "
                                f"现存 {_fmt_num(r['现存'])}）" for _, r in left_pos.iterrows()))


def _dashboard_export(d, ana, target_days):
    """把看板上的几张表一次导出成 Excel（多 sheet），方便直接发给老板。"""
    try:
        op = ana[ana['状态'] == '招聘中'].copy()
        op['标注'] = op.apply(lambda r: '；'.join(_open_issues(r, target_days)), axis=1)
        op['招聘起'] = pd.to_datetime(op['招聘起'], errors='coerce').dt.date.astype(str).replace('NaT', '')

        dd = d[d['demand_max'].fillna(0) > 0]
        dept = dd.groupby('department').agg(需求=('demand_max', 'sum'), 到位=('onboarded', 'sum')).reset_index()
        dept['缺口'] = (dept['需求'] - dept['到位']).clip(lower=0)
        dept['达成率%'] = (dept['到位'] / dept['需求'] * 100).round(0)
        dept = dept.rename(columns={'department': '公司/部门'}).sort_values('达成率%', ascending=False)

        tmp = d.copy()
        tmp['月份'] = pd.to_datetime(tmp['start_date'], errors='coerce').dt.to_period('M').astype(str)
        month = tmp[tmp['月份'] != 'NaT'].groupby('月份').size().rename('新增岗位').reset_index()

        ret = d.groupby('department').agg(入职=('onboarded', 'sum'), 离职=('left', 'sum'),
                                          现存=('current_headcount', 'sum')).reset_index()
        ret['留存%'] = ret.apply(
            lambda r: round((r['入职'] - r['离职']) / r['入职'] * 100, 1) if r['入职'] else None, axis=1)
        ret = ret.rename(columns={'department': '公司/部门'})

        detail = ana[[c for c in ['岗位', '部门', '类目', '状态', '需求', '入职', '周期(天)', '同类中位(天)',
                                  '简历', '面试', '通过', 'Offer', '离职', '现存', '简历→面试%', '面试→通过%',
                                  'Offer→入职%', '招聘起', '招聘止'] if c in ana.columns]].copy()
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine='openpyxl') as w:
            detail.to_excel(w, index=False, sheet_name='岗位明细')
            op[['岗位', '部门', '招聘起', '周期(天)', '目标(天)', '需求', '简历', '面试', '标注']].to_excel(
                w, index=False, sheet_name='在招跟踪')
            dept.to_excel(w, index=False, sheet_name='公司达成')
            month.to_excel(w, index=False, sheet_name='月度新增')
            ret.to_excel(w, index=False, sheet_name='人员留存')
        st.download_button('⬇️ 导出看板 Excel（5 个表）', buf.getvalue(),
                           file_name=f"招聘看板_{pd.Timestamp.today().date()}.xlsx",
                           mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                           key='dash_export')
    except Exception as e:
        st.caption(f'（导出暂时不可用：{e}）')


def _dashboard_detail(d, ana, only_risk, target_days):
    """七、岗位明细（备查与导出）。标注只针对在招岗位。"""
    st.markdown('**在线编辑**（直接改表格里的数据，改完点保存，刷新也在）')
    _editable_table(d.reset_index(drop=True), key='edit_all', height=400)
    st.divider()
    detail = d.reset_index(drop=True)
    extra = ana[['周期(天)', '主要卡点']].copy()
    extra['标注'] = ana.apply(
        lambda r: '；'.join(_open_issues(r, target_days)) if r['状态'] == '招聘中' else '', axis=1)
    extra['待跟进'] = (extra['标注'].fillna('').str.len() > 0).astype(int)
    detail = pd.concat([detail, extra[['周期(天)', '主要卡点', '标注', '待跟进']]], axis=1)
    if only_risk:
        detail = detail[detail['待跟进'] > 0]

    with st.expander('岗位明细（可导出）', expanded=False):
        if detail.empty:
            st.info('当前筛选范围内没有需要跟进的在招岗位。')
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
        order = ['岗位', '部门', '招聘类目', '当前状态', '周期(天)', '标注', '主要卡点', '招聘需求',
                 '推送简历', '邀约', '面试', '通过', 'Offer', '入职', '离职', '现存', '日均简历',
                 '招聘起', '招聘止']
        order += [ec for ec in du.extra_columns(d)]
        st.dataframe(disp[[c for c in order if c in disp.columns]], width='stretch', height=420, hide_index=True)


EDIT_COLS = [('状态', 'status'), ('招聘起', 'start_date'), ('招聘止', 'end_date'), ('招聘需求', 'demand'),
             ('推送简历', 'resumes'), ('面试', 'interviewed'), ('通过', 'passed'), ('Offer', 'offer'),
             ('入职', 'onboarded'), ('离职', 'left'), ('现存', 'current_headcount')]


def _editable_table(src, key, extra_cols=None, height=360):
    """可直接在网页里改并保存的表格：保存后写进持久层，刷新也不会丢。"""
    if src is None or len(src) == 0:
        st.info('当前没有可编辑的数据。')
        return
    view = pd.DataFrame({'岗位': src['position'].astype(str), '部门': src['department'].astype(str)})
    for cn, en in EDIT_COLS:
        view[cn] = src[en] if en in src.columns else None
    editable = ['岗位', '部门'] + [cn for cn, _ in EDIT_COLS]
    for cn, vals in (extra_cols or {}).items():
        view[cn] = list(vals)
    cfg = {'岗位': st.column_config.TextColumn('岗位'), '部门': st.column_config.TextColumn('部门')}
    for cn, en in EDIT_COLS:
        if en == 'status':
            cfg[cn] = st.column_config.SelectboxColumn(cn, options=du.STATUS_ORDER)
        elif en in du.DATE_COLS:
            cfg[cn] = st.column_config.DateColumn(cn)
        elif en in du.NUMERIC_COLS:
            cfg[cn] = st.column_config.NumberColumn(cn, format='%.0f')
        else:
            cfg[cn] = st.column_config.TextColumn(cn, help='可填 2-3 表示区间')
    try:
        edited = st.data_editor(view, key=key, num_rows='fixed', hide_index=True, height=height,
                                column_config=cfg,
                                disabled=[c for c in view.columns if c not in editable])
    except Exception as e:
        st.caption(f'表格暂时不可编辑（{e}），已改为只读显示。')
        st.dataframe(view, width='stretch', hide_index=True, height=height)
        return
    if edited is None:
        edited = view
    if st.button('💾 保存表格修改', key=key + '_save'):
        full = _clean_any(du.load_current_data()).reset_index(drop=True)
        idx = {f"{r['position']}||{r['department']}": i for i, r in full.iterrows()}
        changed = 0
        for _, r in edited.reset_index(drop=True).iterrows():
            k = f"{r['岗位']}||{r['部门']}"
            if k not in idx:
                continue
            i = idx[k]
            full.at[i, 'position'] = r['岗位']
            full.at[i, 'department'] = r['部门']
            for cn, en in EDIT_COLS:
                v = r.get(cn)
                if en in du.DATE_COLS:
                    v = pd.to_datetime(v, errors='coerce')
                elif en in du.NUMERIC_COLS:
                    s = '' if v is None else str(v).strip()
                    v = float('nan') if s in ('', 'None', 'nan', 'NaT') else _num(v)
                full.at[i, en] = v
            changed += 1
        du.save_store(full)
        st.session_state['flash'] = f'已保存 {changed} 个岗位的修改，图表已同步刷新。'
        st.rerun()


def _dashboard_projects(d):
    """公司卡片墙：每家公司一张卡，关键数字一眼看完。"""
    st.subheader('各公司一览')
    rows = []
    for dept, g in d.groupby('department'):
        demand = _num(g['demand_max'].fillna(0).sum())
        onb = _num(g['onboarded'].fillna(0).sum())
        done = g[g['status'] == '完成招聘']
        rows.append({
            '公司': dept, '岗位': int(len(g)),
            '在招': int((g['status'] == '招聘中').sum()),
            '完成': int((g['status'] == '完成招聘').sum()),
            '暂停': int((g['status'] == '暂停').sum()),
            '需求': demand, '入职': onb, '离职': _num(g['left'].fillna(0).sum()),
            '缺口': max(demand - onb, 0),
            '平均时长': _num(done['duration_days'].dropna().mean()) if not done.empty else float('nan'),
            '转化率': _pct(onb, _num(g['resumes'].fillna(0).sum())),
        })
    if not rows:
        st.info('暂无公司数据。')
        return
    cards = pd.DataFrame(rows).sort_values(['缺口', '岗位'], ascending=[False, False])
    for i in range(0, len(cards), 3):
        cols = st.columns(3)
        for col, (_, r) in zip(cols, cards.iloc[i:i + 3].iterrows()):
            with col:
                with st.container(border=True):
                    st.markdown(f"**{r['公司']}**")
                    extra = f" · 暂停 {r['暂停']}" if r['暂停'] else ''
                    st.caption(f"岗位 {r['岗位']} 个（在招 {r['在招']} · 完成 {r['完成']}{extra}）")
                    m1, m2 = st.columns(2)
                    m1.metric('需求 / 入职', f"{_fmt_num(r['需求'])} / {_fmt_num(r['入职'])}")
                    m2.metric('缺口', _fmt_num(r['缺口']))
                    tail = f"简历→入职 {_fmt_pct(r['转化率'])}"
                    if r['离职']:
                        tail += f" ｜ 离职 {_fmt_num(r['离职'])} 人"
                    st.caption(tail)


def _dashboard_risks(d, ana, target_days):
    """异常与风险清单：把表里对不上的地方自动列出来。"""
    st.subheader('异常与风险清单')
    rows = []
    for _, r in ana.iterrows():
        stt = r['状态']
        msgs = []
        if stt == '招聘中' and (r['周期(天)'] != r['周期(天)'] or _num(r['周期(天)'], 0) <= 0):
            msgs.append('招聘中，但没填招聘起日期')
        if stt == '招聘中' and _num(r['简历'], 0) <= 0:
            msgs.append('招聘中，还没有收到简历')
        if stt == '完成招聘' and _num(r['入职'], 0) <= 0:
            msgs.append('已完成，但入职 0 人')
        if _num(r['离职'], 0) > 0:
            msgs.append(f"入职后离职 {_fmt_num(r['离职'])} 人")
        if r['需求'] == r['需求'] and _num(r['现存'], 0) >= 0 and _num(r['现存'], 0) < _num(r['需求'], 0) \
                and stt != '完成招聘':
            msgs.append(f"现存 {_fmt_num(r['现存'])} 人 < 需求 {_fmt_num(r['需求'])} 人")
        if stt == '暂停' and _num(r['简历'], 0) > 0:
            msgs.append(f"已暂停，但已推送 {_fmt_num(r['简历'])} 份简历")
        if stt == '招聘中' and _num(r['周期(天)'], 0) > target_days:
            msgs.append(f"已招 {_fmt_num(r['周期(天)'])} 天，超过目标 {target_days:.0f} 天")
        if msgs:
            rows.append({'岗位': r['岗位'], '公司/部门': r['部门'], '状态': stt, '问题': '；'.join(msgs),
                         '需求': r['需求'], '简历': r['简历'], '面试': r['面试'],
                         '入职': r['入职'], '离职': r['离职'], '现存': r['现存']})
    if not rows:
        st.success('没有发现异常数据。')
        return
    risk = pd.DataFrame(rows)
    st.caption(f'共 {len(risk)} 个岗位存在需要核对的地方（按状态看：'
               + '、'.join(f'{k} {v}' for k, v in risk['状态'].value_counts().items()) + '）')
    st.dataframe(risk, width='stretch', hide_index=True, height=min(140 + 36 * len(risk), 520))


def page_dashboard():
    st.title('📊 招聘经营看板')
    st.caption('总览 · 时间趋势 · 组织结构 · 转化漏斗 · 周期分析 · 人员留存 · 岗位明细。'
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
        include_paused = st.checkbox('把暂停岗位也算进来', value=False, key='incl_paused',
                                     help='默认不统计「暂停」的岗位，只看看在招和已完成。')
        st.subheader('基准设置')
        target_days = st.number_input('目标招聘周期（天）', min_value=5, max_value=180, value=30, step=5,
                                      key='target_days',
                                      help='在招岗位超过这个天数就算超期；已完成岗位超过它也会被标出来。')
        slow_ratio = st.slider('比同类中位数慢多少倍算异常', 1.0, 3.0, 1.5, 0.1, key='slow_ratio',
                               help='已完成岗位的招聘周期超过「同类岗位中位数 × 这个倍数」就标为异常。')
        only_risk = st.checkbox('明细只看需要关注的岗位', value=False, key='only_risk')

    mask = (df['status'].isin(sel_status)) & (df['department'].isin(sel_dept)) & (df['category'].isin(sel_cat))
    d = df[mask].copy()
    if not include_paused:
        d = d[d['status'] != '暂停']
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
    tabs = st.tabs(['📌 招聘总览', '🏢 项目 / 公司', '📋 岗位明细', '🔻 转化与周期', '⚠️ 异常与风险', '👥 人员留存'])
    with tabs[0]:
        _dashboard_summary(d, ana, target_days)
        _dashboard_monthly(d)
    with tabs[1]:
        _dashboard_projects(d)
        _dashboard_progress(d, ana, target_days)
    with tabs[2]:
        _dashboard_detail(d, ana, only_risk, target_days)
        _dashboard_export(d, ana, target_days)
    with tabs[3]:
        _dashboard_efficiency(d, ana, target_days)
    with tabs[4]:
        _dashboard_risks(d, ana, target_days)
    with tabs[5]:
        _dashboard_retention(d, ana)


# ==================== 岗位匹配：①岗位信息 → ②批量上传 → ③智能打分 ====================

def _parse_years(v):
    try:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return 0.0
        s = ''.join(ch for ch in str(v) if ch.isdigit() or ch == '.')
        return float(s) if s else 0.0
    except Exception:
        return 0.0


def _split_list(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return []
    for sep in ('、', '，', ',', '；', ';', '/', '|'):
        v = str(v).replace(sep, '\n')
    return [x.strip() for x in str(v).split('\n') if x.strip()]


def _bulk_import_positions(positions):
    """批量导入 JD：上传一份岗位表，直接进岗位库。"""
    with st.expander('📥 批量导入 JD（Excel / CSV）', expanded=False):
        st.caption('表头支持：岗位名称（或 岗位）、部门（或 公司）、学历要求、经验要求、技能关键词、'
                   '硬性条件、岗位描述（JD 原文）。同名的岗位会跳过，不会重复添加。')
        up = st.file_uploader('选择 .xlsx / .xls / .csv', type=['xlsx', 'xls', 'csv'], key='jd_bulk')
        if up is not None and st.button('导入到岗位库', key='jd_bulk_btn'):
            try:
                raw = pd.read_csv(up) if str(up.name).lower().endswith('.csv') else pd.read_excel(up)
            except Exception as e:
                st.error(f'读取失败：{e}')
                return
            alias = {'岗位': '岗位名称', '招聘岗位': '岗位名称', '公司': '部门', '岗位部门': '部门',
                     '学历': '学历要求', '经验': '经验要求', '经验要求(年)': '经验要求',
                     '关键词': '技能关键词', '技能': '技能关键词', 'JD': '岗位描述', '岗位JD': '岗位描述'}
            raw = raw.rename(columns={str(c).strip(): alias.get(str(c).strip(), str(c).strip())
                                      for c in raw.columns})
            have = {p['name'] for p in positions}
            added, skipped = 0, 0
            for _, r in raw.iterrows():
                name = str(r.get('岗位名称', '') or '').strip()
                if not name or name.lower() == 'nan':
                    continue
                if name in have:
                    skipped += 1
                    continue
                edu_txt = str(r.get('学历要求', '') or '')
                edu = 0
                for k, v in EDU_LEVEL.items():
                    if k and k in edu_txt:
                        edu = v
                        break
                desc = r.get('岗位描述', '')
                kws = _split_list(r.get('技能关键词'))
                if not kws and desc:
                    try:
                        kws = mt.parse_jd_keywords(str(desc))
                    except Exception:
                        kws = []
                positions.append(_norm_pos({
                    'name': name,
                    'department': str(r.get('部门', '') or ''),
                    'education': edu,
                    'years': _parse_years(r.get('经验要求')),
                    'keywords': kws,
                    'hard_conditions': _split_list(r.get('硬性条件')),
                    'description': '' if pd.isna(desc) else str(desc),
                    'source': '自定义',
                }))
                have.add(name)
                added += 1
            if added:
                save_positions(positions, deleted_builtins())
            st.session_state['flash'] = (f'已导入 {added} 个岗位' + (f'，跳过重名 {skipped} 个' if skipped else '')
                                         + '，共 ' + str(len(positions)) + ' 个岗位。')
            st.rerun()


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
        jd_up = st.file_uploader('或上传 JD 文件（PDF / Word / TXT）', type=['pdf', 'docx', 'txt'], key='jd_file')
        jd_text = ''
        if jd_up is not None:
            try:
                jd_text = mt.extract_text(jd_up.name, jd_up.getvalue())
            except Exception as e:
                st.caption(f'JD 文件解析失败：{e}')
        desc = st.text_area('岗位描述（粘贴 JD）', key='jd_desc', height=130,
                            placeholder='粘贴岗位描述，便于自动解析关键词…')
        if jd_text and not (desc or '').strip():
            desc = jd_text

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

    _bulk_import_positions(positions)

    st.markdown(f'**岗位库**（共 {len(positions)} 个，均为你自己新增 / 上传的，均可删除）')
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


def _grade(score):
    s = _num(score, 0)
    return 'A' if s >= 85 else ('B' if s >= 70 else ('C' if s >= 50 else 'D'))


def render_run(run):
    """渲染某一条已保存的打分记录。"""
    rows = run.get('rows') or []
    if not rows:
        st.info('这条记录里没有结果。')
        return
    res = pd.DataFrame(rows)
    if '匹配分' in res.columns:
        res = res.sort_values('匹配分', ascending=False).reset_index(drop=True)
        res.insert(0, '等级', res['匹配分'].apply(_grade))
    fu = run.get('followup') or {}
    if fu:
        res['跟进'] = res.get('姓名', pd.Series(dtype=str)).map(lambda n: fu.get(str(n), ''))
    res.insert(0, '排名', res.index + 1)
    st.caption(f"岗位：{run.get('position', '')} ｜ 打分方式：{run.get('mode', '')} ｜ 时间：{run.get('time', '')}")

    if '匹配分' in res.columns:
        hi = res[res['匹配分'] >= 70]
        mid = res[(res['匹配分'] >= 50) & (res['匹配分'] < 70)]
        low = res[res['匹配分'] < 50]
        c = st.columns(3)
        c[0].metric('建议约面（≥70）', f'{len(hi)} 人', f'最高 {_fmt_num(res["匹配分"].max(), 1)} 分',
                    delta_color='off')
        c[1].metric('待定（50–69）', f'{len(mid)} 人', delta_color='off')
        c[2].metric('暂缓（<50）', f'{len(low)} 人', delta_color='off')
        tabs = st.tabs([f'建议约面 {len(hi)}', f'待定 {len(mid)}', f'暂缓 {len(low)}', f'全部 {len(res)}'])
        for tab, part in zip(tabs, [hi, mid, low, res]):
            with tab:
                if part.empty:
                    st.caption('这一档暂时没有候选人。')
                else:
                    st.dataframe(part, width='stretch', hide_index=True, height=min(140 + 36 * len(part), 520),
                                 column_config={
                                     '匹配分': st.column_config.ProgressColumn('匹配分', min_value=0, max_value=100,
                                                                               format='%.1f'),
                                     '排名': st.column_config.NumberColumn('排名')})
    else:
        st.dataframe(res, width='stretch', hide_index=True, height=380)

    if '姓名' in res.columns:
        with st.expander('跟进标记（已联系 / 已面试 / 已淘汰）', expanded=False):
            names = [str(x) for x in res['姓名'].tolist()]
            f1, f2, f3 = st.columns([2, 2, 1])
            who = f1.multiselect('候选人', names, key=f"fu_who_{run.get('id')}")
            stt = f2.selectbox('状态', ['未处理', '已联系', '已面试', '已淘汰'], key=f"fu_st_{run.get('id')}")
            if f3.button('保存', key=f"fu_save_{run.get('id')}") and who:
                fu2 = dict(run.get('followup') or {})
                for n in who:
                    fu2[n] = stt
                run['followup'] = fu2
                save_history([run if x.get('id') == run.get('id') else x for x in load_history()])
                st.session_state['flash'] = '已保存跟进状态。'
                st.rerun()

    details = run.get('details') or []
    det_rows = []
    for d in sorted(details, key=lambda x: -(x.get('score') or 0)):
        ai = d.get('ai') or {}
        local = d.get('local') or {}
        det_rows.append({
            '姓名': d.get('name'), '电话': d.get('phone'), '学历': d.get('education'),
            '经验': d.get('years_raw'), '匹配分': d.get('score'), '建议': d.get('tag'),
            'AI摘要': ai.get('summary', ''), '优势': '；'.join(ai.get('strengths') or []),
            '不足': '；'.join(ai.get('gaps') or []), '亮点': '；'.join(ai.get('highlights') or []),
            'AI建议': ai.get('suggestion', ''), '简历来源': d.get('source'),
            '本地规则分': (f"学历 {local.get('edu_score', 0)}/20 ｜ 经验 {local.get('exp_score', 0)}/30 ｜ "
                           f"技能 {local.get('skill_score', 0)}/40 ｜ 硬性 {local.get('hard_score', 0)}/10"),
        })
    try:
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine='openpyxl') as w:
            res.to_excel(w, index=False, sheet_name='打分结果')
            if det_rows:
                pd.DataFrame(det_rows).to_excel(w, index=False, sheet_name='候选人详情')
        st.download_button('⬇️ 导出 Excel（结果 + 候选人详情）', buf.getvalue(),
                           file_name=f"匹配打分结果_{run.get('position', '')}_{run.get('time', '').replace(':', '')}.xlsx",
                           mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                           key=f"xlsx_{run.get('id')}")
    except Exception:
        st.download_button('下载打分结果 CSV', res.to_csv(index=False).encode('utf-8-sig'),
                           file_name=f"匹配打分结果_{run.get('position', '')}.csv",
                           mime='text/csv', key=f"csv_{run.get('id')}")

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

    ledger = []
    for r in hist:
        for dt in (r.get('details') or []):
            nm = dt.get('name') or '未识别'
            ledger.append({'岗位': r.get('position', ''), '打分时间': r.get('time', ''),
                           '姓名': nm, '电话': dt.get('phone') or '—',
                           '学历': dt.get('education', ''), '经验': dt.get('years_raw', ''),
                           '匹配分': dt.get('score'), '等级': _grade(dt.get('score')),
                           '建议': dt.get('tag', ''),
                           '跟进': (r.get('followup') or {}).get(str(nm), '')})
    if ledger:
        led = pd.DataFrame(ledger).sort_values('匹配分', ascending=False).reset_index(drop=True)
        with st.expander(f'候选人台账（累计 {len(led)} 人次，刷新后仍在）', expanded=False):
            st.dataframe(led, width='stretch', hide_index=True, height=min(160 + 34 * len(led), 520))
            try:
                buf = io.BytesIO()
                with pd.ExcelWriter(buf, engine='openpyxl') as w:
                    led.to_excel(w, index=False, sheet_name='候选人台账')
                st.download_button('⬇️ 导出候选人台账 Excel', buf.getvalue(),
                                   file_name=f"候选人台账_{pd.Timestamp.today().date()}.xlsx",
                                   mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                                   key='ledger_xlsx')
            except Exception:
                st.download_button('下载候选人台账 CSV', led.to_csv(index=False).encode('utf-8-sig'),
                                   file_name='候选人台账.csv', mime='text/csv', key='ledger_csv')

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

    if st.session_state.get('_reset_widgets'):      # 删除/恢复岗位后，重置下拉框选中项
        st.session_state['_reset_widgets'] = False
        for k in ('match_pos', 'jd_del'):
            st.session_state.pop(k, None)

    if st.session_state.get('flash'):
        st.success(st.session_state.pop('flash'))

    positions = get_positions()

    # ---------- 第 1 步：岗位信息 ----------
    st.markdown('#### ① 填写 / 选择岗位信息')
    _jd_manage(positions)

    if not positions:
        st.info('岗位库是空的：请先在上面的「📥 批量导入 JD」上传岗位表，或展开「➕ 新增岗位」新增一个。')
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
_storage_badge()

if st.session_state.get('flash'):
    st.success(st.session_state.pop('flash'))
_boot_restore()
pages[page]()
