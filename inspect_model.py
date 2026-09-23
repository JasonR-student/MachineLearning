from anti_fraud_dataset import SAMPLES
from anti_fraud_router import create_model


texts = [question for question, label in SAMPLES]
labels = [label for question, label in SAMPLES]

model = create_model()
model.fit(texts, labels)

# 从 Pipeline 中取出两个已经训练好的对象。
tfidf = model.named_steps["tfidf"]
classifier = model.named_steps["classifier"]

print("模型学到的特征数量：", len(tfidf.vocabulary_))
print("类别顺序：", classifier.classes_)
print("权重矩阵形状：", classifier.coef_.shape)

question = "陌生客服让我转账安全吗"

# 将问题转换成数字向量。
question_vector = tfidf.transform([question])

# 取得每个特征的名字。
feature_names = tfidf.get_feature_names_out()

# 找到问题向量中不为 0 的位置。
feature_indexes = question_vector.nonzero()[1]

print("\n这个问题包含的部分特征：")

for index in feature_indexes[:20]:
    feature = feature_names[index]
    value = question_vector[0, index]
    print(f"{feature!r} -> {value:.4f}")

print("\n五类预测概率：")

probabilities = classifier.predict_proba(question_vector)[0]

for label, probability in zip(
    classifier.classes_,
    probabilities,
):
    print(f"{label:<16} {probability:.2%}")