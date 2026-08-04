from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


# 每个元组代表一条训练样本：
# 第一个元素是用户问题，第二个元素是人工标注的正确类别。
# samples：列表 list
# ├── 第 0 项：元组 tuple
# │   ├── 第 0 项："退款多久到账"
# │   └── 第 1 项："knowledge"
# └── 第 1 项：元组 tuple
#     ├── 第 0 项："帮我取消订单"
#     └── 第 1 项："action"
samples = [
    ("退款一般多久到账", "knowledge"),
    ("忘记密码应该怎么办", "knowledge"),
    ("电子发票如何开具", "knowledge"),
    ("会员有哪些权益", "knowledge"),
    ("商品保修期是多久", "knowledge"),
    ("退货需要满足什么条件", "knowledge"),
    ("优惠券使用规则是什么", "knowledge"),
    ("运费是如何计算的", "knowledge"),
    ("运费的阶梯公式是什么", "knowledge"),
    ("有哪些基本会员", "knowledge"),
    ("订阅操作教程有哪些", "knowledge"),
    ("驾驶员驾车有哪些基本流程", "knowledge"),

    ("帮我查询订单状态", "action"),
    ("取消我的订单", "action"),
    ("修改订单收货地址", "action"),
    ("查看包裹物流位置", "action"),
    ("帮我申请退款", "action"),
    ("把发票发送到我的邮箱", "action"),
    ("查询我的账户余额", "action"),
    ("提醒我明天付款", "action"),
    ("提醒我晚上八点开会", "action"),
    ("查看丢失手机的实时定位", "action")
]


# 将问题和标签分别取出来。
texts = [sample[0] for sample in samples]   #列表推导式
labels = [sample[1] for sample in samples]


# 将数据分成训练集和测试集。
# 训练集用于让模型学习，测试集用于检查模型是否能处理没见过的问题。
train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts,
    labels,
    test_size=0.25,       # 25% 的数据作为测试集
    random_state=47,      # 固定随机结果，方便重复实验
    stratify=labels,      # 保证两种标签都出现在训练集和测试集中
)

# Sklearn Pipeline 机器学习预处理流水线
# Pipeline 会依次执行两个步骤：
# 1. 将中文文本转换成 TF-IDF 数字特征。
# 2. 使用逻辑回归学习这些特征与标签之间的关系。
model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            analyzer="char",       # 按汉字分析，不依赖中文分词工具
            ngram_range=(1, 2),    # 同时学习单字和相邻的两个字
        ),
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,         # 允许模型有足够次数完成训练
            random_state=42,
        ),
    ),
])


# fit 表示训练：模型从训练数据中学习分类规律。
model.fit(train_texts, train_labels)

# predict 表示预测：让模型判断测试问题的类别。
predicted_labels = model.predict(test_texts)

# 对比预测标签和正确标签。
accuracy = accuracy_score(test_labels, predicted_labels)
print(f"测试集准确率：{accuracy:.2%}\n")

for text, expected, predicted in zip(
    test_texts, test_labels, predicted_labels
):
    print(f"问题：{text}")
    print(f"正确标签：{expected}")
    print(f"预测标签：{predicted}")
    print()


# 使用训练好的模型处理新的用户问题。
new_questions = [
    "会员订阅规则是什么",
    "帮我预定早上九点的机票",
    "给我定位实时位置的教程",
]

new_predictions = model.predict(new_questions)

print("新问题预测结果：")

for question, label in zip(new_questions, new_predictions):
    print(f"{question} -> {label}")