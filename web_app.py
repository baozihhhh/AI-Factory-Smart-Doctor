import streamlit as st
from utils import DeepSeekClient
from processor import LogAnalyzer
from knowledge_engine import (
    save_uploaded_file, list_uploaded_docs,
    build_index, search_knowledge, has_index,
)
from pdf_export import generate_diagnosis_pdf
from trend_analysis import (
    load_history, save_history,
    compute_error_frequency, compute_level_distribution, build_trend_prompt,
)
import concurrent.futures
import time
import os
import re
import pandas as pd

st.set_page_config(page_title="AI 工业医生 - RAG 2.0", layout="wide")

# ── 工业级全局 CSS 注入 ──
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* ── 全局基调 ── */
    html, body, [class*="css"] {
        font-family: 'Inter', 'Monaco', 'Consolas', monospace !important;
        letter-spacing: 0.01em;
    }
    .stApp {
        background-color: #0E1117;
    }
    h1 {
        font-size: 1.4rem !important;
        font-weight: 600 !important;
        color: #E8EAED !important;
        letter-spacing: 0.02em;
    }
    h2 {
        font-size: 1.15rem !important;
        font-weight: 500 !important;
        color: #C8CCD0 !important;
    }
    h3 {
        font-size: 0.95rem !important;
        font-weight: 500 !important;
        color: #A0A8B4 !important;
        text-transform: none;
        letter-spacing: 0.04em;
    }
    p, span, label, .stMarkdown {
        font-size: 0.875rem !important;
        line-height: 1.65 !important;
        color: #BFC5CD !important;
    }
    .stCaption p {
        font-size: 0.78rem !important;
        color: #6B7280 !important;
    }

    /* ── 侧边栏 ── */
    section[data-testid="stSidebar"] {
        background-color: #0A0E14 !important;
        border-right: 1px solid #1E2530 !important;
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #C8CCD0 !important;
        text-transform: none !important;
    }
    section[data-testid="stSidebar"] .stMarkdown p {
        color: #8892A0 !important;
        font-size: 0.82rem !important;
    }

    /* ── 标签页导航栏 ── */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #141920;
        border-radius: 8px;
        padding: 4px;
        gap: 4px;
        border: 1px solid #1E2530;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 6px;
        color: #6B7280;
        font-size: 0.82rem !important;
        font-weight: 500;
        padding: 8px 20px;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        background-color: #007BFF !important;
        color: #FFFFFF !important;
        border: none;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #E8EAED;
    }
    .stTabs [data-baseweb="tab-panel"] {
        padding-top: 1.5rem;
    }

    /* ── 按钮 ── */
    .stButton > button {
        font-family: 'Inter', 'Monaco', 'Consolas', monospace !important;
        font-size: 0.82rem !important;
        font-weight: 500;
        border-radius: 6px;
        border: 1px solid #2A3040;
        transition: all 0.2s ease;
    }
    .stButton > button[kind="primary"] {
        background-color: #007BFF !important;
        border-color: #007BFF !important;
        color: #FFFFFF !important;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #0069D9 !important;
        box-shadow: 0 0 12px rgba(0,123,255,0.3);
    }
    .stButton > button:not([kind="primary"]) {
        background-color: #1A1F2B !important;
        color: #A0A8B4 !important;
    }
    .stButton > button:not([kind="primary"]):hover {
        background-color: #252D3A !important;
        color: #E8EAED !important;
    }

    /* ── 收藏星星按钮 ── */
    .fav-btn .stButton > button {
        background: transparent !important;
        border: none !important;
        font-size: 0.9rem !important;
        padding: 2px 6px !important;
        min-height: 0 !important;
        line-height: 1 !important;
    }

    /* ── 输入框/上传区 ── */
    .stTextArea textarea,
    .stTextInput input {
        background-color: #141920 !important;
        border: 1px solid #2A3040 !important;
        border-radius: 6px !important;
        color: #E8EAED !important;
        font-family: 'Inter', 'Monaco', 'Consolas', monospace !important;
        font-size: 0.84rem !important;
    }
    .stTextArea textarea:focus,
    .stTextInput input:focus {
        border-color: #007BFF !important;
        box-shadow: 0 0 0 1px #007BFF !important;
    }

    /* ── 提示框（info/warning/error/success） ── */
    .stAlert {
        border-radius: 6px !important;
        border: none !important;
        font-size: 0.84rem !important;
    }
    div[data-testid="stAlert"] > div {
        padding: 0.7rem 1rem !important;
    }

    /* ── st.divider ── */
    hr {
        border-color: #1E2530 !important;
        margin: 0.8rem 0 !important;
    }

    /* ── toggle 开关 ── */
    .stToggle label span {
        font-size: 0.82rem !important;
        color: #8892A0 !important;
    }

    /* ── 进度条 ── */
    .stProgress > div > div > div {
        background-color: #007BFF !important;
    }

    /* ── 诊断结果卡片样式 ── */
    .diag-card {
        background: #141920;
        border: 1px solid #1E2530;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 12px;
        position: relative;
    }
    .diag-card .log-text {
        font-family: 'Courier New', monospace;
        font-size: 0.8rem;
        color: #A0A8B4;
        background: #0E1117;
        padding: 8px 12px;
        border-radius: 4px;
        border-left: 3px solid #2A3040;
        margin: 8px 0;
        word-break: break-all;
    }
    .diag-card .level-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 4px;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.05em;
    }
    .level-critical { background: rgba(220,53,69,0.15); color: #FF6B7A; border: 1px solid rgba(220,53,69,0.3); }
    .level-warning  { background: rgba(255,193,7,0.12); color: #FFD166; border: 1px solid rgba(255,193,7,0.25); }
    .level-normal   { background: rgba(40,167,69,0.12); color: #6BCB77; border: 1px solid rgba(40,167,69,0.25); }

    /* ── KPI 卡片网格 ── */
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
        gap: 12px;
        margin: 12px 0 20px 0;
    }
    .kpi-card {
        background: #141920;
        border: 1px solid #1E2530;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .kpi-card .kpi-value {
        font-size: 1.6rem;
        font-weight: 700;
        margin: 4px 0;
        font-family: 'Inter', sans-serif;
    }
    .kpi-card .kpi-label {
        font-size: 0.72rem;
        color: #6B7280;
        letter-spacing: 0.04em;
        font-weight: 500;
    }
    .kpi-card .kpi-desc {
        font-size: 0.7rem;
        color: #4B5563;
        margin-top: 4px;
    }
    .kpi-neutral  .kpi-value { color: #E8EAED; }
    .kpi-danger   .kpi-value { color: #FF6B7A; }
    .kpi-success  .kpi-value { color: #6BCB77; }
    .kpi-warning  .kpi-value { color: #FFD166; }
    .kpi-info     .kpi-value { color: #60A5FA; }

    /* ── 侧边栏状态指示灯 ── */
    @keyframes breathe {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    .status-dot {
        display: inline-block;
        width: 7px;
        height: 7px;
        border-radius: 50%;
        margin-right: 6px;
        position: relative;
        top: -1px;
    }
    .dot-green  { background: #4B7A5A; box-shadow: 0 0 6px rgba(75,122,90,0.4); animation: breathe 3s ease-in-out infinite; }
    .dot-blue   { background: #4A6A8A; box-shadow: 0 0 6px rgba(74,106,138,0.4); animation: breathe 3s ease-in-out infinite 0.5s; }
    .dot-gray   { background: #4B5563; }

    .status-line {
        font-size: 0.78rem !important;
        color: #6B7280 !important;
        padding: 3px 0;
    }

    /* ── 文档列表 ── */
    .doc-item {
        display: flex;
        align-items: center;
        padding: 8px 12px;
        background: #141920;
        border: 1px solid #1E2530;
        border-radius: 6px;
        margin-bottom: 6px;
        font-size: 0.82rem;
        color: #A0A8B4;
    }

    /* ── 隐藏 Streamlit 默认 footer/header ── */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }

    /* ── 聊天消息 ── */
    .stChatMessage {
        background: #141920 !important;
        border: 1px solid #1E2530 !important;
        border-radius: 8px !important;
    }

    /* ── PDF 导出区 ── */
    .pdf-export-section {
        background: #141920;
        border: 1px solid #1E2530;
        border-radius: 8px;
        padding: 16px 20px;
        margin-top: 16px;
        text-align: center;
    }
    /* ── PDF 下载按钮蓝化 ── */
    .pdf-export-section .stDownloadButton > button {
        background-color: #007BFF !important;
        border-color: #007BFF !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        border-radius: 6px !important;
        padding: 10px 20px !important;
        transition: all 0.2s ease !important;
    }
    .pdf-export-section .stDownloadButton > button:hover {
        background-color: #0069D9 !important;
        box-shadow: 0 0 14px rgba(0,123,255,0.35) !important;
    }
    /* ── 演示用例按钮（低调虚线框） ── */
    .demo-sample-btn .stButton > button {
        background: transparent !important;
        border: 1px dashed #3A4050 !important;
        color: #6B7280 !important;
        font-size: 0.76rem !important;
        font-weight: 400 !important;
        padding: 4px 14px !important;
        min-height: 0 !important;
        border-radius: 4px !important;
    }
    .demo-sample-btn .stButton > button:hover {
        border-color: #6B7280 !important;
        color: #A0A8B4 !important;
        background: rgba(255,255,255,0.03) !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_client():
    return DeepSeekClient()


# ── 初始化 session_state ──
defaults = {
    "diagnosis_results": [],
    "chat_history": [],
    "all_logs": "",
    "diagnosing": False,
    "uploaded_logs": [],
    "log_count": 0,
    "history_records": [],
    "viewing_history": None,
    "favorite_cases": [],
    "favorited_ids": set(),
    "multi_expert_enabled": False,
    "trend_ai_analysis": "",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# 启动时从磁盘加载历史记录
if not st.session_state.history_records:
    disk_history = load_history()
    if disk_history:
        st.session_state.history_records = disk_history

# 知识库状态：从文件系统同步
if "kb_indexed" not in st.session_state:
    st.session_state.kb_indexed = has_index()
if "rag_enabled" not in st.session_state:
    st.session_state.rag_enabled = True

# ── 侧边栏（精简：仅状态 + 核心控制） ──
with st.sidebar:
    st.markdown('<p style="font-size:1.1rem; font-weight:600; color:#E8EAED; margin:16px 0 4px 0;">AI 工业医生</p>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:0.72rem; color:#4B5563; margin:0 0 16px 0; letter-spacing:0.06em;">日志智能诊断平台 / RAG 2.0</p>', unsafe_allow_html=True)
    st.markdown("---")

    if st.button("开始新一轮诊断", type="primary", use_container_width=True):
        st.session_state.diagnosis_results = []
        st.session_state.chat_history = []
        st.session_state.all_logs = ""
        st.session_state.diagnosing = False
        st.session_state.uploaded_logs = []
        st.session_state.log_count = 0
        st.session_state.viewing_history = None
        st.rerun()

    st.markdown("---")

    # 极简状态指示灯（带辅助小字）
    _card = 'background:rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.1); border-radius:4px; padding:8px 12px; margin-bottom:6px;'
    _sub = 'font-size:0.65rem; color:#4B5563; margin:2px 0 0 13px; padding:0;'
    st.markdown('<p style="font-size:0.78rem; font-weight:500; color:#6B7280; letter-spacing:0.04em; margin-bottom:10px;">系统状态</p>', unsafe_allow_html=True)
    st.markdown(f'<div style="{_card}"><p class="status-line" style="margin:0 !important;"><span class="status-dot dot-green"></span>DeepSeek API 已连接</p><p style="{_sub}">延迟: ~30ms</p></div>', unsafe_allow_html=True)
    st.markdown(f'<div style="{_card}"><p class="status-line" style="margin:0 !important;"><span class="status-dot dot-blue"></span>待检测条目: {st.session_state.log_count} 条</p><p style="{_sub}">支持 .txt / .log 格式</p></div>', unsafe_allow_html=True)

    # 知识库状态（实时同步）
    st.session_state.kb_indexed = has_index()
    _doc_count = len(list_uploaded_docs())
    if st.session_state.kb_indexed:
        st.markdown(f'<div style="{_card}"><p class="status-line" style="margin:0 !important;"><span class="status-dot dot-blue"></span>知识库: 已就绪</p><p style="{_sub}">已加载: {_doc_count} 篇文档</p></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div style="{_card}"><p class="status-line" style="margin:0 !important;"><span class="status-dot dot-gray"></span>知识库: 未构建</p><p style="{_sub}">已上传: {_doc_count} 篇文档</p></div>', unsafe_allow_html=True)

    # 会诊模式状态
    if st.session_state.multi_expert_enabled:
        st.markdown(f'<div style="{_card}"><p class="status-line" style="margin:0 !important;"><span class="status-dot dot-blue"></span>会诊模式: 已开启</p><p style="{_sub}">电气 / 机械 / 综合 三专家并行</p></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div style="{_card}"><p class="status-line" style="margin:0 !important;"><span class="status-dot dot-gray"></span>会诊模式: 关闭</p><p style="{_sub}">单专家流式诊断</p></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # RAG 开关
    st.session_state.rag_enabled = st.toggle("开启 RAG 知识库增强", value=st.session_state.rag_enabled)

    # 多专家会诊开关
    st.session_state.multi_expert_enabled = st.toggle("多专家会诊模式", value=st.session_state.multi_expert_enabled)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("清除历史记录", use_container_width=True):
        st.session_state.diagnosis_results = []
        st.session_state.chat_history = []
        st.session_state.all_logs = ""
        st.session_state.diagnosing = False
        st.session_state.uploaded_logs = []
        st.session_state.log_count = 0
        st.session_state.history_records = []
        st.session_state.viewing_history = None
        st.session_state.favorite_cases = []
        st.session_state.favorited_ids = set()
        st.session_state.trend_ai_analysis = ""
        # 清除磁盘历史
        from config import HISTORY_FILE_PATH
        if os.path.exists(HISTORY_FILE_PATH):
            os.remove(HISTORY_FILE_PATH)
        st.rerun()

    if st.session_state.history_records:
        st.markdown("---")
        st.markdown('<p style="font-size:0.78rem; font-weight:500; color:#6B7280; letter-spacing:0.04em; margin-bottom:8px;">诊断历史</p>', unsafe_allow_html=True)
        for idx, record in enumerate(st.session_state.history_records):
            if st.button(f"{record['time']}  |  {record['count']} 条日志",
                         key=f"history_{idx}", use_container_width=True):
                st.session_state.viewing_history = idx
                st.rerun()

# ── 主界面标题（视觉锚点） ──
st.markdown("""
<div style="text-align:center; margin-bottom:40px; padding:22px 0 12px 0;">
    <p style="font-size:2.5rem; font-weight:700; color:#E8EAED; margin:0 0 8px 0; line-height:1.2; letter-spacing:0.02em;">
        AI 工业医生：日志智能诊断平台 <span style="color:#007BFF;">(RAG 2.0)</span>
    </p>
    <p style="font-size:1rem; color:#808495; margin:0; letter-spacing:0.03em; font-weight:400;">
        Powered by DeepSeek &amp; LangChain &nbsp;|&nbsp; 您的工厂安全卫士
    </p>
</div>
""", unsafe_allow_html=True)

tab_diag, tab_kb, tab_cases, tab_trend = st.tabs(["故障诊断", "工程师知识库", "经典案例库", "运行状况趋势"])

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 标签页 1：故障诊断
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_diag:
    # 空状态欢迎
    if not st.session_state.diagnosing and not st.session_state.diagnosis_results:
        if st.session_state.viewing_history is None:
            st.info("系统已就绪。请上传日志文件或直接粘贴内容开始诊断。")
            st.markdown("<br>", unsafe_allow_html=True)

    # ── 数据导入 ──
    if not st.session_state.diagnosing and not st.session_state.diagnosis_results and st.session_state.viewing_history is None:
        st.markdown("### 数据导入")

        # 载入示例按钮
        _SAMPLE_LOGS = """2024-03-15 08:23:11 [ERROR] ERR_0042 伺服驱动器X轴过载报警 电流值: 12.8A 阈值: 10.0A
2024-03-15 08:23:45 [WARN] WARN_017 液压系统压力波动 当前: 18.2MPa 标准: 20.0MPa
2024-03-15 08:24:02 [ERROR] ERR_0088 主轴编码器信号丢失 持续时间: 350ms
2024-03-15 08:25:30 [INFO] 冷却系统温度正常 当前: 42.3°C
2024-03-15 08:26:15 [WARN] WARN_005 PLC通信延迟 响应时间: 280ms 阈值: 200ms
2024-03-15 08:27:00 [ERROR] FAULT_12 安全光幕触发 区域: B3 状态: 遮挡"""

        # 右上角低调演示按钮
        _demo_col1, _demo_col2 = st.columns([4, 1])
        with _demo_col2:
            st.markdown('<div class="demo-sample-btn">', unsafe_allow_html=True)
            if st.button("载入演示用例", key="demo_sample"):
                analyzer = LogAnalyzer()
                lines = [line.strip() for line in _SAMPLE_LOGS.strip().split("\n") if line.strip()]
                valid_logs = [log for log in lines if analyzer.is_valid_log(log)]
                st.session_state.uploaded_logs = valid_logs
                st.session_state.log_count = len(valid_logs)
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            uploaded_file = st.file_uploader(
                "上传日志文件", type=["txt", "log"],
                help="支持 .txt 和 .log 格式", label_visibility="visible"
            )
        if uploaded_file:
            content = uploaded_file.read().decode("utf-8")
            logs = [line.strip() for line in content.split("\n") if line.strip()]
            analyzer = LogAnalyzer()
            valid_logs = [log for log in logs if analyzer.is_valid_log(log)]
            st.session_state.uploaded_logs = valid_logs
            st.session_state.log_count = len(valid_logs)
            st.success(f"已扫描到 {len(valid_logs)} 条有效日志条目")

        with col2:
            manual_input = st.text_area("或直接粘贴日志内容", height=150,
                                        placeholder="在此粘贴日志条目...")
        if manual_input:
            lines = [line.strip() for line in manual_input.split("\n") if line.strip()]
            analyzer = LogAnalyzer()
            valid_logs = [log for log in lines if analyzer.is_valid_log(log)]
            st.session_state.uploaded_logs = valid_logs
            st.session_state.log_count = len(valid_logs)

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("立即开始诊断", type="primary", use_container_width=True,
                      disabled=st.session_state.log_count == 0):
            st.session_state.diagnosing = True
            st.rerun()

    # ── 诊断进行中 ──
    if st.session_state.diagnosing and not st.session_state.diagnosis_results:
        client = get_client()
        analyzer = LogAnalyzer()
        logs = st.session_state.uploaded_logs
        results = []
        all_logs_text = ""

        progress_bar = st.progress(0, text="正在诊断中...")
        stream_container = st.container()

        # ── 多专家会诊：专家定义 ──
        EXPERT_PROMPTS = {
            "电气专家": "你是一位拥有20年经验的工业电气维修专家，擅长PLC、伺服驱动器、变频器、传感器等电气系统故障诊断。请从电气系统角度分析以下日志，给出故障根因、风险等级和排查步骤。",
            "机械专家": "你是一位拥有20年经验的工业机械维修专家，擅长机械传动、液压气动、轴承齿轮、结构件等机械系统故障诊断。请从机械系统角度分析以下日志，给出故障根因、风险等级和排查步骤。",
            "综合专家": "你是一位拥有20年经验的工业自动化综合维修专家，擅长从系统层面分析故障，综合考虑电气、机械、软件、工艺等多方面因素。请从综合系统角度分析以下日志，给出故障根因、风险等级和排查步骤。",
        }

        def _call_expert(expert_name, system_prompt, log_text, knowledge_ctx):
            """单个专家的非流式 API 调用。"""
            msgs = [
                {"role": "system", "content": system_prompt + (f"\n\n参考资料：\n{knowledge_ctx}" if knowledge_ctx else "")},
                {"role": "user", "content": f"请分析以下工业日志：\n{log_text}"},
            ]
            resp = client.client.chat.completions.create(model="deepseek-chat", messages=msgs)
            return expert_name, resp.choices[0].message.content

        for i, log in enumerate(logs):
            level = analyzer.classify_error(log)
            all_logs_text += f"{log}\n"

            # 知识库检索（受 RAG 开关控制）
            knowledge_context = ""
            rag_ref_count = 0
            if st.session_state.rag_enabled and st.session_state.kb_indexed:
                kb_results = search_knowledge(client, log, top_k=3)
                if kb_results:
                    rag_ref_count = len(kb_results)
                    knowledge_context = "\n---\n".join(
                        [f"[来源: {r['doc']}]\n{r['text']}" for r in kb_results]
                    )

            # 实时输出：先显示日志和分级
            with stream_container:
                level_class = "level-critical" if level == "严重" else ("level-warning" if level == "警告" else "level-normal")
                st.markdown(f'<div class="diag-card"><div class="log-text">{log}</div><span class="level-badge {level_class}">{level}</span></div>', unsafe_allow_html=True)
                result_placeholder = st.empty()
                result_placeholder.info("AI 正在分析...")

            expert_results = {}

            if st.session_state.multi_expert_enabled:
                # ── 多专家并行会诊（非流式） ──
                result_placeholder.info("多专家会诊中，请稍候...")
                try:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                        futures = {
                            executor.submit(_call_expert, name, prompt, log, knowledge_context): name
                            for name, prompt in EXPERT_PROMPTS.items()
                        }
                        for future in concurrent.futures.as_completed(futures):
                            ename, eanswer = future.result()
                            expert_results[ename] = eanswer
                    # 以综合专家结果作为主结果
                    ai_result = expert_results.get("综合专家", list(expert_results.values())[0] if expert_results else "会诊完成")
                except Exception:
                    ai_result = client.get_ai_response(
                        f"请分析以下工业日志：\n{log}",
                        knowledge_context=knowledge_context
                    )
            else:
                # ── 单专家流式输出 ──
                try:
                    system_msg = "你是一位拥有20年经验的工业机器人与数控机床维修专家。你只负责分析工业日志。你的回答必须包含：1. 故障根因；2. 风险等级；3. 具体的排查步骤。严禁回答与工业维修无关的话题。"
                    if knowledge_context:
                        system_msg += f"\n\n以下是从工程师知识库中检索到的参考资料，请结合这些经验进行分析：\n{knowledge_context}"

                    stream = client.client.chat.completions.create(
                        model="deepseek-chat",
                        messages=[
                            {"role": "system", "content": system_msg},
                            {"role": "user", "content": f"请分析以下工业日志：\n{log}"}
                        ],
                        stream=True
                    )
                    ai_result = ""
                    for chunk in stream:
                        delta = chunk.choices[0].delta.content
                        if delta:
                            ai_result += delta
                            if level == "严重":
                                result_placeholder.error(ai_result)
                            elif level == "警告":
                                result_placeholder.warning(ai_result)
                            else:
                                result_placeholder.info(ai_result)
                except Exception:
                    ai_result = client.get_ai_response(
                        f"请分析以下工业日志：\n{log}",
                        knowledge_context=knowledge_context
                    )

            # 更新占位符显示最终结果
            if level == "严重":
                result_placeholder.error(ai_result)
            elif level == "警告":
                result_placeholder.warning(ai_result)
            else:
                result_placeholder.info(ai_result)

            with stream_container:
                st.divider()

            result_item = {"log": log, "level": level, "result": ai_result, "id": i,
                           "rag_ref_count": rag_ref_count}
            if expert_results:
                result_item["expert_results"] = expert_results
            results.append(result_item)
            progress_bar.progress((i + 1) / len(logs),
                                  text=f"已完成 {i+1}/{len(logs)}")

        progress_bar.empty()
        st.session_state.diagnosis_results = results
        st.session_state.all_logs = all_logs_text
        st.session_state.diagnosing = False

        st.session_state.history_records.append({
            "time": time.strftime("%m-%d %H:%M"),
            "count": len(results),
            "results": results
        })
        # 持久化到磁盘
        save_history(st.session_state.history_records)
        st.rerun()

    # ── 诊断结果展示 ──
    if st.session_state.diagnosis_results:
        # ── KPI 卡片看板 ──
        st.markdown("### 诊断总览")
        diag_results = st.session_state.diagnosis_results
        total_lines = len(diag_results)
        error_count = sum(1 for r in diag_results if r['level'] == '严重')
        warn_count = sum(1 for r in diag_results if r['level'] == '警告')
        normal_count = sum(1 for r in diag_results if r['level'] == '正常')
        high_priority = error_count
        total_rag_refs = sum(r.get('rag_ref_count', 0) for r in diag_results)
        rag_display = str(total_rag_refs) if st.session_state.rag_enabled else "关闭"

        kpi_html = f"""
        <div class="kpi-grid">
            <div class="kpi-card kpi-neutral">
                <div class="kpi-label">扫描总数</div>
                <div class="kpi-value">{total_lines}</div>
                <div class="kpi-desc">已分析日志条目</div>
            </div>
            <div class="kpi-card kpi-danger">
                <div class="kpi-label">严重故障</div>
                <div class="kpi-value">{error_count}</div>
                <div class="kpi-desc">{"需立即处理" if error_count > 0 else "无严重故障"}</div>
            </div>
            <div class="kpi-card kpi-warning">
                <div class="kpi-label">警告</div>
                <div class="kpi-value">{warn_count}</div>
                <div class="kpi-desc">{"建议关注" if warn_count > 0 else "无警告"}</div>
            </div>
            <div class="kpi-card kpi-success">
                <div class="kpi-label">正常</div>
                <div class="kpi-value">{normal_count}</div>
                <div class="kpi-desc">运行正常</div>
            </div>
            <div class="kpi-card kpi-danger">
                <div class="kpi-label">高优先级</div>
                <div class="kpi-value">{high_priority}</div>
                <div class="kpi-desc">{"需升级处理" if high_priority > 0 else "无需升级"}</div>
            </div>
            <div class="kpi-card kpi-info">
                <div class="kpi-label">RAG 引用</div>
                <div class="kpi-value">{rag_display}</div>
                <div class="kpi-desc">{"知识库已激活" if st.session_state.rag_enabled else "RAG 已关闭"}</div>
            </div>
        </div>
        """
        st.markdown(kpi_html, unsafe_allow_html=True)

        st.markdown("### 详细结果")
        for item in st.session_state.diagnosis_results:
            level_class = "level-critical" if item['level'] == "严重" else ("level-warning" if item['level'] == "警告" else "level-normal")

            with st.container():
                col_log, col_fav = st.columns([19, 1])
                with col_log:
                    st.markdown(f"""
                    <div class="diag-card">
                        <span class="level-badge {level_class}">{item['level']}</span>
                        <div class="log-text">{item['log']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col_fav:
                    fav_key = f"fav_{item['id']}"
                    is_faved = item['id'] in st.session_state.favorited_ids
                    if is_faved:
                        if st.button("\u2b50", key=fav_key, help="已收藏 - 点击移除"):
                            st.session_state.favorited_ids.discard(item['id'])
                            st.session_state.favorite_cases = [
                                c for c in st.session_state.favorite_cases if c.get('id') != item['id']
                            ]
                            st.rerun()
                    else:
                        if st.button("\u2606", key=fav_key, help="点击收藏"):
                            st.session_state.favorite_cases.append(item)
                            st.session_state.favorited_ids.add(item['id'])
                            st.rerun()

                if item['level'] == "严重":
                    st.error(item['result'])
                elif item['level'] == "警告":
                    st.warning(item['result'])
                else:
                    st.info(item['result'])

                # 多专家会诊结果展示
                if "expert_results" in item and item["expert_results"]:
                    expert_names = list(item["expert_results"].keys())
                    expert_tabs = st.tabs(expert_names)
                    for et, ename in zip(expert_tabs, expert_names):
                        with et:
                            st.markdown(item["expert_results"][ename])

                st.divider()

        # PDF 导出按钮
        st.markdown('<div class="pdf-export-section">', unsafe_allow_html=True)
        pdf_bytes = generate_diagnosis_pdf(st.session_state.diagnosis_results, st.session_state.rag_enabled)
        st.download_button(
            label="\u2b07 导出技术诊断单（PDF）",
            data=pdf_bytes,
            file_name=f"诊断报告_{time.strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

        # 追问历史
        if st.session_state.chat_history:
            st.markdown("### 追问对话")
            for chat in st.session_state.chat_history:
                with st.chat_message(chat["role"]):
                    st.write(chat["content"])

        # 追问输入框
        if prompt := st.chat_input("输入追问内容，进一步了解诊断详情..."):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            client = get_client()
            context = f"已分析的日志：\n{st.session_state.all_logs}\n\n用户追问：{prompt}"
            response = client.get_ai_response(context)
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            st.rerun()

    # ── 查看历史记录 ──
    elif st.session_state.viewing_history is not None:
        record = st.session_state.history_records[st.session_state.viewing_history]
        st.markdown(f"### 历史记录 &mdash; {record['time']}")
        for item in record['results']:
            level_class = "level-critical" if item['level'] == "严重" else ("level-warning" if item['level'] == "警告" else "level-normal")
            st.markdown(f"""
            <div class="diag-card">
                <span class="level-badge {level_class}">{item['level']}</span>
                <div class="log-text">{item['log']}</div>
            </div>
            """, unsafe_allow_html=True)
            if item['level'] == "严重":
                st.error(item['result'])
            elif item['level'] == "警告":
                st.warning(item['result'])
            else:
                st.info(item['result'])
            st.divider()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 标签页 2：工程师知识库
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_kb:
    st.markdown("### 知识库管理")
    st.caption("上传技术手册与维修文档，构建向量索引后可在诊断时自动检索相关知识。")
    st.markdown("<br>", unsafe_allow_html=True)

    # 上传区（不触发 rerun，避免阻断文档列表显示）
    kb_files = st.file_uploader(
        "上传技术文档", type=["pdf", "txt"],
        accept_multiple_files=True,
        help="支持 PDF 和 TXT 格式的技术手册及维修记录",
        key="kb_uploader"
    )
    if kb_files:
        for f in kb_files:
            save_uploaded_file(f)
        st.toast(f"已保存 {len(kb_files)} 个文档")

    st.markdown("<br>", unsafe_allow_html=True)

    # 已上传文档列表（始终从磁盘实时扫描）
    st.markdown("### 已上传文档")
    docs = list_uploaded_docs()
    if docs:
        to_delete = None
        for doc_name in docs:
            col_name, col_del = st.columns([8, 1])
            with col_name:
                st.markdown(f'<div class="doc-item">{doc_name}</div>', unsafe_allow_html=True)
            with col_del:
                if st.button("X", key=f"del_{doc_name}"):
                    to_delete = doc_name

        if to_delete is not None:
            from config import RAW_DOCS_DIR
            file_path = os.path.join(RAW_DOCS_DIR, to_delete)
            try:
                os.remove(file_path)
            except PermissionError:
                import gc
                gc.collect()
                os.remove(file_path)
            st.toast(f"已删除: {to_delete}")
            st.rerun()
    else:
        st.info("暂无文档。请上传技术手册或维修记录以开始使用知识库。")

    st.markdown("<br>", unsafe_allow_html=True)

    # 构建索引
    if st.button("构建 / 更新索引", type="primary", use_container_width=True):
        st.toast("正在提取文档特征并构建索引...")
        client = get_client()
        progress = st.progress(0, text="正在处理文档...")

        def update_progress(pct, msg):
            progress.progress(pct, text=msg)

        update_progress(0.1, "正在解析文档...")
        count = build_index(client, progress_callback=update_progress)
        progress.empty()
        if count > 0:
            st.session_state.kb_indexed = True
            st.toast(f"知识库就绪！已向量化 {count} 个文本块。")
            st.rerun()
        else:
            st.toast("未找到可处理的文档，请先上传文件。")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 标签页 3：经典案例库
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_cases:
    st.markdown("### 经典案例库")
    st.caption("从诊断结果中收藏的典型故障案例，便于日后参考与复盘。")
    st.markdown("<br>", unsafe_allow_html=True)

    if st.session_state.favorite_cases:
        to_remove = None
        for idx, case in enumerate(st.session_state.favorite_cases):
            level_class = "level-critical" if case['level'] == "严重" else ("level-warning" if case['level'] == "警告" else "level-normal")
            with st.container():
                col_case, col_unfav = st.columns([19, 1])
                with col_case:
                    st.markdown(f"""
                    <div class="diag-card">
                        <span style="font-size:0.72rem; color:#4B5563; font-weight:500;">案例 #{idx + 1}</span>
                        <span class="level-badge {level_class}" style="margin-left:12px;">{case['level']}</span>
                        <div class="log-text">{case['log']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col_unfav:
                    if st.button("X", key=f"unfav_{idx}", help="移除收藏"):
                        to_remove = idx

                if case['level'] == "严重":
                    st.error(case['result'])
                elif case['level'] == "警告":
                    st.warning(case['result'])
                else:
                    st.info(case['result'])
                st.divider()

        if to_remove is not None:
            removed = st.session_state.favorite_cases.pop(to_remove)
            st.session_state.favorited_ids.discard(removed.get('id'))
            st.toast("已从案例库移除")
            st.rerun()
    else:
        st.info("暂无收藏案例。在诊断结果中点击收藏按钮即可保存典型案例。")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 标签页 4：运行状况趋势
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_trend:
    st.markdown("### 运行状况趋势")
    st.caption("基于历史诊断数据的故障趋势分析与预测性维护建议。")
    st.markdown("<br>", unsafe_allow_html=True)

    history = st.session_state.history_records

    if not history:
        st.info("暂无历史诊断数据。完成诊断后，趋势数据将自动生成。")
    else:
        # KPI 指标
        total_sessions = len(history)
        total_items = sum(r.get("count", 0) for r in history)

        kpi_trend_html = f"""
        <div class="kpi-grid">
            <div class="kpi-card kpi-neutral">
                <div class="kpi-label">历史诊断次数</div>
                <div class="kpi-value">{total_sessions}</div>
                <div class="kpi-desc">累计诊断轮次</div>
            </div>
            <div class="kpi-card kpi-info">
                <div class="kpi-label">累计分析条目</div>
                <div class="kpi-value">{total_items}</div>
                <div class="kpi-desc">已分析日志总数</div>
            </div>
        </div>
        """
        st.markdown(kpi_trend_html, unsafe_allow_html=True)

        # 故障代码频率
        error_freq = compute_error_frequency(history)
        if error_freq:
            st.markdown("### 故障代码频率分布")
            freq_df = pd.DataFrame(error_freq, columns=["故障代码", "出现次数"])
            freq_df = freq_df.set_index("故障代码")
            st.bar_chart(freq_df)
        else:
            st.info("暂未检测到标准故障代码（如 ERR_001, WARN_042 等）。")

        # 风险等级分布
        level_dist = compute_level_distribution(history)
        if any(v > 0 for v in level_dist.values()):
            st.markdown("### 风险等级分布")
            dist_df = pd.DataFrame(list(level_dist.items()), columns=["风险等级", "数量"])
            dist_df = dist_df.set_index("风险等级")
            st.bar_chart(dist_df)

        st.markdown("<br>", unsafe_allow_html=True)

        # AI 趋势分析
        if st.button("生成趋势分析报告", type="primary", use_container_width=True):
            error_freq_data = compute_error_frequency(history)
            level_dist_data = compute_level_distribution(history)
            trend_prompt = build_trend_prompt(error_freq_data, level_dist_data)

            client = get_client()
            trend_placeholder = st.empty()
            trend_placeholder.info("正在生成趋势分析报告...")

            try:
                stream = client.client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "system", "content": "你是一位资深工业设备运维专家，擅长从故障数据中发现趋势和隐患。"},
                        {"role": "user", "content": trend_prompt},
                    ],
                    stream=True,
                )
                trend_text = ""
                for chunk in stream:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        trend_text += delta
                        trend_placeholder.markdown(trend_text)
                st.session_state.trend_ai_analysis = trend_text
            except Exception as e:
                trend_placeholder.warning(f"趋势分析生成失败: {e}")

        # 显示缓存的趋势分析
        elif st.session_state.trend_ai_analysis:
            st.markdown("### AI 趋势分析报告")
            st.markdown(st.session_state.trend_ai_analysis)
