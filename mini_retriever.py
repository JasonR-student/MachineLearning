import math


# 我们暂时人工指定文本中值得关注的关键词。
# 后续使用 Embedding 模型后，就不需要手动定义这些关键词了。
KEYWORDS = ["退款", "到账", "工作日", "登录", "密码", "发票"]

# 模拟知识库中的三段资料。
DOCUMENTS = [
    "退款审核后，通常在三个工作日内到账。",
    "忘记登录密码时，可以通过手机号重置。",
    "电子发票会在付款完成后生成。",
]


def vectorize(text):
    """
    将文本转换成由 0 和 1 组成的向量。

    例如：
    文本包含“退款”，对应位置就是 1；
    文本不包含“退款”，对应位置就是 0。
    """
    vector = []

    for keyword in KEYWORDS:
        if keyword in text:
            vector.append(1)
        else:
            vector.append(0)

    return vector


def cosine_similarity(vector_a, vector_b):
    """
    计算两个向量的余弦相似度。

    返回值越接近 1，表示两个向量越相似；
    返回值越接近 0，表示两个向量越不相似。
    """

    # 计算两个向量的点积。
    dot_product = sum(
        a * b for a, b in zip(vector_a, vector_b)
    )

    # 计算两个向量各自的长度。
    length_a = math.sqrt(sum(a * a for a in vector_a))
    length_b = math.sqrt(sum(b * b for b in vector_b))

    # 避免除以 0。
    if length_a == 0 or length_b == 0:
        return 0

    return dot_product / (length_a * length_b)


question = "退款多久能够到账？"
question_vector = vectorize(question)

results = []

# 将问题分别与知识库中的每段资料进行比较。
for document in DOCUMENTS:
    document_vector = vectorize(document)
    score = cosine_similarity(question_vector, document_vector)

    results.append({
        "document": document,
        "score": score,
    })

# 按相似度从高到低排列。
results.sort(
    key=lambda item: item["score"],
    reverse=True,
)

for result in results:
    print(f"相似度：{result['score']:.3f}")
    print(f"资料：{result['document']}")
    print()