# -*- coding: utf-8 -*-
"""简历解析与岗位匹配度打分（规则引擎，无需调用外部 API）。"""
import re
import io

# ---------- 简历文本提取 ----------

def extract_text(filename, raw_bytes):
    """按扩展名提取简历文本。支持 pdf / docx / txt。"""
    name = (filename or '').lower()
    if name.endswith('.pdf'):
        return _pdf_to_text(raw_bytes)
    if name.endswith('.docx'):
        return _docx_to_text(raw_bytes)
    return _txt_to_text(raw_bytes)


def _pdf_to_text(raw_bytes):
    import pdfplumber
    parts = []
    with pdfplumber.open(io.BytesIO(raw_bytes)) as pdf:
        for page in pdf.pages:
            parts.append(page.extract_text() or '')
    return '\n'.join(parts)


def _docx_to_text(raw_bytes):
    import docx
    d = docx.Document(io.BytesIO(raw_bytes))
    lines = [p.text for p in d.paragraphs]
    for table in d.tables:
        for row in table.rows:
            lines.append(' | '.join(cell.text for cell in row.cells))
    return '\n'.join(lines)


def _txt_to_text(raw_bytes):
    for enc in ('utf-8', 'gb18030'):
        try:
            return raw_bytes.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    return raw_bytes.decode('utf-8', errors='replace')


# ---------- 通用技能词典 ----------

SKILL_DICT = [
    'Python', 'Java', 'JavaScript', 'TypeScript', 'C++', 'C#', 'Go', 'Rust', 'PHP',
    'HTML', 'CSS', 'React', 'Vue', 'Node.js', '前端开发', '后端开发', '全栈',
    'SQL', 'MySQL', 'PostgreSQL', 'Redis', 'MongoDB', '数据库', '大数据', 'Hadoop', 'Spark',
    '机器学习', '深度学习', '大模型', 'LLM', 'NLP', '自然语言处理', 'PyTorch', 'TensorFlow',
    '数据分析', '数据挖掘', '数据可视化', 'Excel', 'Power BI', 'Tableau', 'SPSS', 'R语言',
    '项目管理', '敏捷开发', '产品设计', '产品运营', '用户调研', '竞品分析', '需求分析',
    '电商运营', '店铺运营', '活动策划', '用户运营', '内容运营', '社群运营', '新媒体运营',
    '市场营销', '品牌推广', '公关', '媒介', '广告投放', '文案', '写作', '内容策划',
    '供应链', '采购', '库存管理', '物流', '仓储', '订单管理', '供应商管理',
    '商品运营', '选品', '陈列', '门店管理', '零售', '销售', '客户管理',
    '财务', '会计', '税务', '人力资源', '招聘', '培训', '行政',
    '英语', '英语六级', '英语四级', '雅思', '托福', '口语',
    'Photoshop', 'PS', 'Figma', 'Sketch', 'Axure', 'PR', 'AE', '视频剪辑', '摄影',
    '自动化测试', '测试', 'Linux', 'Docker', 'Kubernetes', 'Git', '云服务', 'AWS',
]

def _kw_in_text(kw, text):
    """技能关键词匹配：英文/含符号词用单词边界，中文词直接包含匹配。"""
    if re.search(r'[A-Za-z0-9]', kw):
        pat = r'(?<![A-Za-z0-9])' + re.escape(kw) + r'(?![A-Za-z0-9])'
        return re.search(pat, text) is not None
    return kw in text

# ---------- 学历 ----------

EDU_LEVELS = [
    ('博士', 8), ('硕士', 7), ('研究生', 7), ('本科', 6), ('学士', 6),
    ('大专', 4), ('专科', 4), ('中专', 3), ('高中', 3), ('初中', 2),
]

EDU_LEVEL_NAME = {8: '博士', 7: '硕士', 6: '本科', 4: '大专', 3: '高中/中专', 2: '初中'}


def _max_edu_level(text):
    level = 0
    hit = None
    for kw, lv in EDU_LEVELS:
        if kw in text and lv > level:
            level, hit = lv, kw
    return level, hit


def parse_resume(text):
    """从简历文本提取关键信息。"""
    text = text or ''
    result = {
        'name': '', 'phone': '', 'email': '', 'school': '',
        'education': 0, 'education_name': '未识别', 'years': 0.0, 'years_raw': '',
        'city': '', 'skills': [], 'text': text,
    }

    m = re.search(r'姓名\s*[:：]?\s*([\u4e00-\u9fa5·]{2,4})', text)
    if m:
        result['name'] = m.group(1)

    m = re.search(r'(?<!\d)1[3-9]\d{9}(?!\d)', text)
    if m:
        result['phone'] = m.group(0)

    m = re.search(r'[\w.+-]+@[\w-]+\.[\w.]+', text)
    if m:
        result['email'] = m.group(0)

    m = re.search(r'([\u4e00-\u9fa5A-Za-z]{2,}(?:大学|学院|学校))', text)
    if m:
        result['school'] = m.group(1)

    level, hit = _max_edu_level(text)
    result['education'] = level
    result['education_name'] = EDU_LEVEL_NAME.get(level, '未识别') if level else '未识别'

    m = re.search(r'(?:工作)?经验\s*[:：]?\s*(\d+(?:\.\d+)?)\s*年', text)
    if not m:
        m = re.search(r'(\d+(?:\.\d+)?)\s*年(?:以上)?(?:的)?(?:工作)?经验', text)
    if m:
        result['years'] = float(m.group(1))
        result['years_raw'] = m.group(0)

    m = re.search(r'(?:现居|所在城市|城市|base)\s*[:：]?\s*([\u4e00-\u9fa5]{2,4})', text)
    if m:
        result['city'] = m.group(1)

    if not result['name']:
        headers = ('教育背景', '工作经历', '专业技能', '个人简介', '自我评价',
                   '项目经验', '求职意向', '基本信息', '培训经历', '获奖情况', '实习经历')
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            m = re.match(r'^([\u4e00-\u9fa5·]{2,4})', line)
            if not m:
                continue
            cand, rest = m.group(1), line[len(m.group(1)):]
            if cand in headers:
                continue
            if (re.match(r'^[\s，,、;；]+(?:电话|手机|联系方式|邮箱|性别|男|女)', rest)
                    or re.match(r'^[\s，,、;；]+1[3-9]\d{9}', rest)
                    or (len(line) <= 6 and not re.search(r'[：:]', line))):
                result['name'] = cand
                break

    skills = []
    for kw in SKILL_DICT:
        if kw in skills:
            continue
        if _kw_in_text(kw, text):
            skills.append(kw)
    result['skills'] = skills
    return result


# ---------- 岗位 JD ----------

def parse_jd_keywords(jd_text, extra_keywords=None):
    """从 JD 文本提取关键词：技能词典命中 + 英文技术词 + 用户补充。"""
    text = jd_text or ''
    kws = []
    for kw in SKILL_DICT:
        if kw in kws:
            continue
        if _kw_in_text(kw, text):
            kws.append(kw)
    for m in re.finditer(r'\b[A-Za-z][A-Za-z0-9+#.\-]{1,30}\b', text):
        w = m.group(0)
        if w.lower() not in {t.lower() for t in kws}:
            kws.append(w)
    if extra_keywords:
        for kw in extra_keywords:
            kw = kw.strip()
            if kw and kw not in kws:
                kws.append(kw)
    return kws


def _jd_edu_level(text):
    level, _ = _max_edu_level(text)
    return level


def _jd_years(text):
    m = re.search(r'(\d+(?:\.\d+)?)\s*年(?:以上)?(?:的)?(?:相关)?(?:工作)?经验', text)
    return float(m.group(1)) if m else 0.0


# ---------- 匹配打分 ----------

WEIGHTS = {'education': 20, 'experience': 30, 'skill': 40, 'hard': 10}


def score_resume(jd, resume):
    """jd: dict(education, years, keywords, hard_conditions)
    resume: parse_resume() 结果。
    返回 dict：总分、各维度得分、命中/缺失技能、说明。"""
    text = resume['text'] or ''
    edu_score = 0
    if jd['education'] == 0:
        edu_score = WEIGHTS['education']  # JD 未要求学历，视为通过
    elif resume['education'] >= jd['education']:
        edu_score = WEIGHTS['education']
    elif resume['education'] == jd['education'] - 2:  # 差一档（如本科 vs 大专）
        edu_score = 12
    else:
        edu_score = 5

    exp_score = 0
    if jd['years'] <= 0:
        exp_score = WEIGHTS['experience']
    elif resume['years'] >= jd['years']:
        exp_score = WEIGHTS['experience']
    else:
        exp_score = round(WEIGHTS['experience'] * resume['years'] / jd['years'], 1)

    skill_hits, skill_miss = [], []
    for kw in jd['keywords']:
        (skill_hits if kw in text else skill_miss).append(kw)
    skill_score = round(WEIGHTS['skill'] * len(skill_hits) / max(1, len(jd['keywords'])), 1)

    hard_hits, hard_miss = [], []
    if jd['hard_conditions']:
        for hc in jd['hard_conditions']:
            hit = hc in text
            if not hit:
                pat = re.sub(r'\d+', r'\\d+', re.escape(hc))
                hit = re.search(pat, text) is not None
            (hard_hits if hit else hard_miss).append(hc)
        hard_score = round(WEIGHTS['hard'] * len(hard_hits) / len(jd['hard_conditions']), 1)
    else:
        hard_score = WEIGHTS['hard']

    total = round(edu_score + exp_score + skill_score + hard_score, 1)

    reasons = []
    reasons.append(f"学历：{resume['education_name']}，得分 {edu_score}/{WEIGHTS['education']}"
                   + ('' if jd['education'] == 0 else f"（要求{EDU_LEVEL_NAME.get(jd['education'], '')}及以上）"))
    reasons.append(f"经验：{resume['years_raw'] or '未识别'}，得分 {exp_score}/{WEIGHTS['experience']}"
                   + ('' if jd['years'] <= 0 else f"（要求 {jd['years']:g} 年）"))
    if skill_hits:
        reasons.append('命中技能：' + '、'.join(skill_hits))
    if skill_miss:
        reasons.append('缺失技能：' + '、'.join(skill_miss))
    if jd['hard_conditions'] and hard_miss:
        reasons.append('未满足硬性条件：' + '、'.join(hard_miss))

    return {
        'total': total,
        'edu_score': edu_score,
        'exp_score': exp_score,
        'skill_score': skill_score,
        'hard_score': hard_score,
        'skill_hits': skill_hits,
        'skill_miss': skill_miss,
        'reasons': reasons,
    }


# ---------- 内置岗位（结构化） ----------

JD_POSITIONS = [
    {
        'name': '全栈实习生', 'department': '北京PGS', 'education': 6, 'years': 0.0,
        'keywords': ['Python', 'JavaScript', 'HTML', 'CSS', 'React', 'Vue', 'SQL', '数据库', 'Git', '全栈'],
        'hard_conditions': ['每周实习至少4天', '实习3个月以上'],
        'description': '岗位：全栈实习生。负责参与公司产品前后端功能开发与维护。要求本科及以上学历，计算机相关专业，每周实习至少4天，实习3个月以上。熟悉 Python、JavaScript、HTML/CSS，了解前端框架 React 或 Vue，掌握 SQL 与数据库基础，熟练使用 Git。',
    },
    {
        'name': '电商运营管培生', 'department': '运营', 'education': 6, 'years': 0.0,
        'keywords': ['数据分析', 'Excel', '电商运营', '活动策划', '用户运营', '文案', '写作', '沟通'],
        'hard_conditions': [],
        'description': '岗位：电商运营管培生。负责店铺日常运营、活动策划与数据复盘。要求本科及以上学历，专业不限，经验不限。熟悉 Excel 与数据分析，了解电商平台运营逻辑，具备活动策划与用户运营意识，文案写作能力佳，沟通表达能力强。',
    },
    {
        'name': '媒介专员PR', 'department': '领势', 'education': 6, 'years': 1.0,
        'keywords': ['公关', '媒介', '内容策划', '文案', '写作', '新媒体运营', '品牌推广', '沟通'],
        'hard_conditions': [],
        'description': '岗位：媒介专员PR。负责媒体关系维护、内容策划与品牌传播。要求本科及以上学历，新闻传播、市场营销相关专业优先，1年以上相关经验。具备公关、媒介、内容策划经验，文案写作与沟通能力强，熟悉新媒体运营。',
    },
    {
        'name': '商品运维专员', 'department': '优沃森', 'education': 4, 'years': 1.0,
        'keywords': ['Excel', '库存管理', '供应链', '商品运营', '数据分析', '物流', '采购'],
        'hard_conditions': [],
        'description': '岗位：商品运维专员。负责商品上下架、库存管理与日常数据维护。要求大专及以上学历，1年以上相关经验。熟悉 Excel 与库存管理流程，了解供应链基础，做事细致，责任心强。',
    },
    {
        'name': '大模型开发实习生', 'department': '北京PGS', 'education': 7, 'years': 0.0,
        'keywords': ['Python', 'PyTorch', '大模型', '深度学习', '机器学习', 'NLP', '自然语言处理', 'Transformer'],
        'hard_conditions': ['实习6个月以上'],
        'description': '岗位：大模型开发实习生。参与大模型训练、微调与推理优化工作。要求硕士及以上学历（优秀本科生亦可），计算机相关专业，实习6个月以上。熟悉 Python、PyTorch，了解大模型、深度学习、NLP 与 Transformer 原理，有相关项目经验优先。',
    },
]


def position_label(p):
    """岗位在选项里的显示名。"""
    return f"{p.get('name', '')}（{p.get('department', '') or '未填部门'}）"
# ---------- DeepSeek 智能分析 ----------

def build_jd_text(pos):
    """把结构化岗位转成供 AI / 展示用的岗位要求文本。"""
    lines = [f"岗位名称：{pos.get('name', '')}"]
    if pos.get('department'):
        lines.append(f"部门：{pos['department']}")
    edu = EDU_LEVEL_NAME.get(pos.get('education', 0), '不限')
    lines.append(f"学历要求：{edu}及以上" if edu != '不限' else '学历要求：不限')
    years = pos.get('years', 0) or 0
    lines.append(f"经验要求：{years:g} 年及以上" if years > 0 else '经验要求：不限')
    if pos.get('keywords'):
        lines.append('技能关键词：' + '、'.join(pos['keywords']))
    if pos.get('hard_conditions'):
        lines.append('硬性条件：' + '、'.join(pos['hard_conditions']))
    desc = (pos.get('description') or '').strip()
    if desc:
        lines.append('岗位描述：' + desc)
    return '\n'.join(lines)


def _parse_ai_json(content):
    """把 AI 返回内容解析为 dict，容忍 Markdown 代码块等杂质。"""
    import json
    import re
    if isinstance(content, dict):
        return content
    text = content.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r'\{.*\}', text, re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    raise ValueError('AI 返回内容不是有效 JSON：' + text[:200])


def deepseek_analyze(api_key, jd_text, resume_text, base_url='https://api.deepseek.com',
                     model='deepseek-chat', timeout=180):
    """调用 DeepSeek（OpenAI 兼容接口）对简历做智能匹配分析。

    返回 dict：score(0-100), summary, strengths, gaps, highlights, suggestion。
    失败抛 RuntimeError。
    """
    import json
    import urllib.request
    import urllib.error

    system = (
        '你是一位资深的招聘HR和人才评估专家。你会收到“岗位要求”和“候选人简历”，'
        '请专业、客观地进行人岗匹配评估，只输出一个 JSON 对象（不要输出其他文字、不要用 Markdown 代码块），字段如下：\n'
        '{"score": 0到100的整数, "summary": "一句话总体评价", '
        '"strengths": ["优点1", "优点2"], "gaps": ["不足1", "不足2"], '
        '"highlights": ["亮点1", "亮点2"], "suggestion": "是否建议约面及后续关注点的建议"}'
    )
    user = f'【岗位要求】\n{jd_text}\n\n【候选人简历】\n{resume_text[:12000]}'
    url = base_url.rstrip('/') + '/chat/completions'
    headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {api_key}'}

    def call(payload):
        req = urllib.request.Request(url, data=json.dumps(payload).encode('utf-8'),
                                     headers=headers, method='POST')
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode('utf-8'))

    payload = {
        'model': model,
        'messages': [
            {'role': 'system', 'content': system},
            {'role': 'user', 'content': user},
        ],
        'temperature': 0.3,
        'max_tokens': 1500,
        'response_format': {'type': 'json_object'},
    }
    try:
        body = call(payload)
    except urllib.error.HTTPError as e:
        detail = e.read().decode('utf-8', errors='replace')[:500]
        if 'response_format' in detail and 'json_object' in detail:
            payload.pop('response_format', None)  # 部分兼容服务不支持 json_object
            try:
                body = call(payload)
            except urllib.error.HTTPError as e2:
                d2 = e2.read().decode('utf-8', errors='replace')[:500]
                raise RuntimeError(f'DeepSeek API 请求失败（HTTP {e2.code}）：{d2}') from e2
        else:
            raise RuntimeError(f'DeepSeek API 请求失败（HTTP {e.code}）：{detail}') from e
    except Exception as e:
        raise RuntimeError(f'DeepSeek API 连接失败：{e}') from e

    try:
        content = body['choices'][0]['message']['content']
    except (KeyError, IndexError) as e:
        raise RuntimeError(f'DeepSeek API 返回结构异常：{str(body)[:300]}') from e

    result = _parse_ai_json(content)
    score = result.get('score')
    if score is None:
        score = 50
    try:
        result['score'] = max(0, min(100, int(round(float(score)))))
    except (TypeError, ValueError):
        result['score'] = 50
    for key in ('strengths', 'gaps', 'highlights'):
        if not isinstance(result.get(key), list):
            result[key] = []
    result.setdefault('summary', '')
    result.setdefault('suggestion', '')
    return result


def score_tag(score):
    """按分数给出筛选建议。"""
    if score >= 70:
        return '建议约面'
    if score >= 50:
        return '待定'
    return '暂缓'