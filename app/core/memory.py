from collections import defaultdict

# mỗi session_id có 1 lịch sử chat
chat_memory = defaultdict(list)

MAX_TURNS = 6  # 3 lượt user + AI
