"""PDF 诊断报告导出模块。"""
import os
import time
from io import BytesIO
from fpdf import FPDF
from config import FONTS_DIR


def _resolve_font_path() -> str | None:
    """优先使用系统微软雅黑，回退到 assets/fonts/ 下备用字体。"""
    # Windows 系统字体
    sys_font = r"C:\Windows\Fonts\msyh.ttc"
    if os.path.exists(sys_font):
        return sys_font
    # 回退到项目内置字体
    for name in ("msyh.ttf", "NotoSansSC-Regular.ttf", "SimHei.ttf"):
        path = os.path.join(FONTS_DIR, name)
        if os.path.exists(path):
            return path
    return None


def _safe_text(text: str) -> str:
    """清理文本中可能导致 fpdf 渲染异常的字符。"""
    if not text:
        return ""
    # 替换零宽字符和特殊控制字符
    for ch in ("\u200b", "\u200c", "\u200d", "\ufeff"):
        text = text.replace(ch, "")
    return text


def generate_diagnosis_pdf(diagnosis_results: list[dict], rag_enabled: bool) -> bytes:
    """根据诊断结果列表生成中文 PDF 报告，返回 bytes。"""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=25)
    # 设置充足的页面边距，防止 Not enough horizontal space
    pdf.set_left_margin(15)
    pdf.set_right_margin(15)

    # 加载中文字体
    font_path = _resolve_font_path()
    font_family = "zh"
    if font_path:
        pdf.add_font(font_family, "", font_path, uni=True)
    else:
        font_family = "Helvetica"

    pdf.add_page()
    # multi_cell 可用宽度 = 页面宽度 - 左边距 - 右边距
    usable_w = pdf.w - pdf.l_margin - pdf.r_margin

    # ── 标题 ──
    pdf.set_font(font_family, size=18)
    pdf.cell(usable_w, 14, _safe_text("AI 工业医生 - 技术诊断报告"),
             new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(4)

    # ── 元信息 ──
    pdf.set_font(font_family, size=9)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(usable_w, 6, _safe_text(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}"),
             new_x="LMARGIN", new_y="NEXT")
    pdf.cell(usable_w, 6, _safe_text(f"RAG 知识库增强: {'已开启' if rag_enabled else '已关闭'}"),
             new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)

    # ── 诊断总览 ──
    total = len(diagnosis_results)
    critical = sum(1 for r in diagnosis_results if r["level"] == "严重")
    warning = sum(1 for r in diagnosis_results if r["level"] == "警告")
    normal = sum(1 for r in diagnosis_results if r["level"] == "正常")

    pdf.set_text_color(0, 0, 0)
    pdf.set_font(font_family, size=13)
    pdf.cell(usable_w, 10, _safe_text("一、诊断总览"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font(font_family, size=10)
    pdf.cell(usable_w, 7,
             _safe_text(f"扫描总数: {total}    严重: {critical}    警告: {warning}    正常: {normal}"),
             new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)

    # ── 逐条详细结果 ──
    pdf.set_font(font_family, size=13)
    pdf.cell(usable_w, 10, _safe_text("二、详细诊断结果"), new_x="LMARGIN", new_y="NEXT")

    for idx, item in enumerate(diagnosis_results, 1):
        pdf.ln(3)
        pdf.set_font(font_family, size=10)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(usable_w, 7, _safe_text(f"[{idx}] 风险等级: {item['level']}"),
                 new_x="LMARGIN", new_y="NEXT")

        # 日志原文
        pdf.set_font(font_family, size=8)
        pdf.set_text_color(80, 80, 80)
        pdf.multi_cell(usable_w, 5, _safe_text(f"日志: {item['log']}"))

        # AI 分析
        pdf.set_font(font_family, size=9)
        pdf.set_text_color(30, 30, 30)
        pdf.multi_cell(usable_w, 5, _safe_text(f"分析: {item['result']}"))

        # RAG 引用数
        rag_count = item.get("rag_ref_count", 0)
        if rag_count > 0:
            pdf.set_font(font_family, size=8)
            pdf.set_text_color(100, 100, 100)
            pdf.cell(usable_w, 5, _safe_text(f"RAG 引用: {rag_count} 条"),
                     new_x="LMARGIN", new_y="NEXT")

        pdf.ln(2)

    # ── 页脚声明 ──
    pdf.ln(10)
    pdf.set_font(font_family, size=7)
    pdf.set_text_color(150, 150, 150)
    pdf.multi_cell(usable_w, 4, _safe_text(
        "声明: 本报告由 AI 工业医生系统自动生成，仅供工程技术人员参考。"
        "最终维修决策应结合现场实际情况，由专业人员确认后执行。"))

    buf = BytesIO()
    pdf.output(buf)
    return buf.getvalue()
