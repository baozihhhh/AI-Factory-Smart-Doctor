# AI 工业医生：日志智能诊断平台

## 项目简介
基于 DeepSeek API 的工业日志智能分析系统，提供本地分级 + AI 诊断的双重保障。

## 项目架构

### 核心模块
- **utils.py** - DeepSeekClient 类，封装 API 调用与异常处理
- **processor.py** - LogAnalyzer 类，负责日志读取、验证和本地分级
- **app.py** - 命令行入口，适合快速测试
- **web_app.py** - Streamlit 网页端，提供可视化交互界面

### 数据文件
- **data/raw_logs.txt** - 原始工业日志数据
- **.env** - 存储 DEEPSEEK_API_KEY

## 运行方法

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 配置 API Key
编辑 `.env` 文件，填入你的 DeepSeek API Key：
```
DEEPSEEK_API_KEY=your_key_here
```

### 3. 运行方式

**方式一：一键启动（推荐）**
```bash
双击根目录下的"启动AI工业医生.bat"
```

**方式二：命令行版本**
```bash
python app.py
```

**方式三：网页版本**
```bash
streamlit run web_app.py
```

## 功能特性

### 1. 强化 AI 角色定位
- 20年工业维修专家人设
- 输出包含：故障根因、风险等级、排查步骤
- 严格限定工业维修领域

### 2. 输入拦截机制
- 自动验证日志格式（ERR, STAT, Line 等关键词）
- 拦截非标准日志，防止 AI 幻觉

### 3. 系统鲁棒性
- API 认证失败提示
- 网络异常捕获
- 文件读取异常处理

### 4. 可视化分级
- 🚨 RED 等级 - 严重错误
- ⚠️ YELLOW 等级 - 警告
- ℹ️ 正常信息

## 技术栈
- Python 3.x
- OpenAI SDK (DeepSeek API)
- Streamlit
- python-dotenv
