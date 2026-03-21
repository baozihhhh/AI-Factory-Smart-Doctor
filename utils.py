import os
from dotenv import load_dotenv
from openai import OpenAI

class DeepSeekClient:
    def __init__(self):
        load_dotenv()
        self.client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com"
        )

    def get_ai_response(self, prompt, knowledge_context=""):
        try:
            system_msg = "你是一位拥有20年经验的工业机器人与数控机床维修专家。你只负责分析工业日志。你的回答必须包含：1. 故障根因；2. 风险等级；3. 具体的排查步骤。严禁回答与工业维修无关的话题。"
            if knowledge_context:
                system_msg += f"\n\n以下是从工程师知识库中检索到的参考资料，请结合这些经验进行分析：\n{knowledge_context}"
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            error_msg = str(e)
            if "401" in error_msg or "Authentication" in error_msg:
                return "[错误] API 认证失败，请检查 .env 文件"
            elif "connection" in error_msg.lower() or "network" in error_msg.lower():
                return "[错误] 网络连接异常，请检查网络设置后再试"
            else:
                return f"[错误] API调用失败: {error_msg}"
