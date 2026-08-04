docs = [
    "JasonLearn 是一个 8 周 AI 学习计划，每天学习 1 小时。",
    "RAG 的核心是先检索资料，再让模型基于资料回答。",
    "Embedding 可以把文本转换成向量，用于语义相似度搜索。",
    "Chunking 是把长文档切成适合检索的小片段。",
    "Agent 是能调用工具、维护状态并完成多步骤任务的 AI 程序。"
]

question = "Agent 是什么？"

def score(text, query):
    score = 0
    for word in query:
        if word in text:
            score += 1
    return score

ranked = sorted(docs, key=lambda d: score(d, question), reverse=True)
context = "\n".join(ranked[:2])

prompt = f"""
请只根据下面资料回答问题。
如果资料中没有答案，就回答“资料中没有说明”。

资料：
{context}

问题：
{question}
"""

print(prompt)