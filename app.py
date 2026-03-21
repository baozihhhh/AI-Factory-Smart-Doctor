import sys
sys.stdout.reconfigure(encoding='utf-8')

from utils import DeepSeekClient
from processor import LogAnalyzer

if __name__ == "__main__":
    client = DeepSeekClient()
    analyzer = LogAnalyzer()

    logs = analyzer.read_logs("data/raw_logs.txt")

    for log in logs:
        level = analyzer.classify_error(log)
        print(f"\n原始日志: {log}")
        print(f"本地分级: {level}")

        ai_result = client.get_ai_response(log)
        print(f"AI分析: {ai_result}")
