# 🎯 招聘 HR 工作台（Streamlit）

面向招聘 HR 的轻量工具，两个模块：

| 模块 | 功能 |
| --- | --- |
| 📊 数据看板 | 招聘漏斗、核心 KPI、部门/状态/周期分析；内置数据管理：上传合并、在线增删改行/列、导出 |
| 📄 岗位匹配 | 一条流程：填写岗位信息 → 批量上传简历 → DeepSeek 智能打分排名 |

**岗位匹配**按三步走完：
1. **① 填写 / 选择岗位信息**：自己填写岗位（岗位名称、部门、学历要求、经验要求、技能关键词、硬性条件），也可粘贴岗位描述自动解析；内置 5 个岗位模板，自定义岗位可删除，选择后用于本次打分。
2. **② 批量上传简历**：一次可上传多份 PDF / Word / TXT，或粘贴文本；自动提取姓名、电话、学历、经验、技能。
3. **③ 开始匹配打分**：
   - **智能分析（DeepSeek）**：调用 DeepSeek API 逐份分析，输出 0-100 匹配分 + 总体评价 + 优势/不足/亮点 + 约面建议；打分结果按分数排名，≥70 建议约面、50–69 待定、<50 暂缓，可直接下载 CSV。
   - **本地规则打分（无需 API）**：不依赖任何网络与 AI，用学历20%+经验30%+技能40%+硬性10% 规则打分，AI 调用失败时也会自动回退到这里。

## 关于 DeepSeek API

- 在“③ 打分设置”里填入 DeepSeek API Key（到 [platform.deepseek.com](https://platform.deepseek.com) 注册申请，新用户有免费额度）。
- 默认官方接口 `https://api.deepseek.com` + 模型 `deepseek-chat`（也可用 `deepseek-reasoner`）；如果用的是火山方舟等兼容服务，可自行改 API 地址和模型/接入点 ID。
- DeepSeek 官方接口在国内网络可直接访问，不需要梯子；本地规则模式则完全不联网。

## 绑定 Key（不用每次手动输入）

两种方式任选：

1. **云端绑定（推荐，永久生效）**：在 Streamlit Cloud 的 应用管理页 → Settings → Secrets 里配置：
   ```
   DEEPSEEK_API_KEY = "sk-你的key"
   ```
   保存后网页每次打开都会自动读取，打分设置里会显示「✅ 已绑定 Key（来自 Streamlit Secrets）」，不用再输入。

2. **本地记住（填一次自动带出）**：在“③ 打分设置”填入 Key，勾选「💾 记住 Key 到本地」，点一次“开始打分”后自动保存到 `data/deepseek_key.txt`，下次打开自动带出（本地运行时长期有效；云端应用重启后文件会清空，仍建议用方式 1）。

> ⚠️ 安全提示：`deepseek_key.txt` 已被 `.gitignore` 忽略，请勿把它传到 GitHub；密钥泄露请到 DeepSeek 平台重置。

## 本地运行

```bash
pip install -r requirements.txt
streamlit run app.py
```

浏览器打开 http://localhost:8501 即可使用。

## 部署到 Streamlit Community Cloud（免费，无需服务器）

1. 注册 [github.com](https://github.com) → 新建仓库（如 `recruit-dashboard`）。
2. 把 `app.py`、`data_utils.py`、`matching.py`、`requirements.txt`、`.streamlit/`、`data/` 全部上传到仓库根目录（GitHub 页面 Add file → Upload files 即可）。
3. 打开 [share.streamlit.io](https://share.streamlit.io) 用 GitHub 登录 → Create app → 选仓库，Main file 填 `app.py` → Deploy。
4. 完成后得到 `https://xxx.streamlit.app/` 网址，收藏即可日常使用。更新代码后 Streamlit 会自动重新部署。

## 项目结构

```
recruit-dashboard/
├── app.py            # 主程序（数据看板 + 岗位匹配流程）
├── data_utils.py     # 数据清洗与指标计算
├── matching.py       # 简历解析 + 本地规则打分 + DeepSeek 智能分析
├── data/
│   ├── 岗位招聘详情.xlsx   # 内置示例数据（可替换）
│   ├── user_positions.json   # 用户自定义岗位（自动生成）
│   └── recruitment_store.json # 看板数据存储（自动生成）
├── .streamlit/config.toml # 主题与上传大小配置
└── requirements.txt
```