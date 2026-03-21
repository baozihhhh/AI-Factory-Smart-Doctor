class LogAnalyzer:
    def read_logs(self, file_path):
        try:
            with open(file_path, "r") as f:
                return [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            return []

    def is_valid_log(self, content):
        keywords = ["ERR", "STAT", "Line", "temp", "pressure", "vibration", "flow", "motor", "bar", "hz", "pct"]
        return any(keyword.lower() in content.lower() for keyword in keywords)

    def classify_error(self, log_line):
        if "RED" in log_line or "temp_high" in log_line or "overload" in log_line:
            return "严重"
        elif "YELLOW" in log_line:
            return "警告"
        else:
            return "正常"
