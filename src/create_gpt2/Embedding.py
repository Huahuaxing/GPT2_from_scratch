import numpy as np

class Embedding:
    def __init__(self, vocab_size, embed_dim, max_seq_len):
        self.token_embed = np.random.randn(vocab_size, embed_dim) * 0.02
        self.pos_embed = np.random.randn(max_seq_len, embed_dim) * 0.02

    def forward(self, token_ids):
        seq_len = token_ids.shape[-1]
        tok_emb = self.token_embed[token_ids]
        pos_emb = self.pos_embed[:seq_len]
        return tok_emb + pos_emb


if __name__ == "__main__":
# 超参数
    vocab_size = 1000
    embed_dim = 16
    max_seq_len = 32

    embedding_layer = Embedding(vocab_size, embed_dim, max_seq_len)

    # 原始英文句子
    sentence = "I love building gpt models"
    words = sentence.split()
    print("原始英文句子：", sentence)
    print("分词结果：", words)

    # 简易模拟tokenizer：手动给每个单词分配一个token id（仅演示，不是真实BPE）
    word2id = {word: idx for idx, word in enumerate(words)}
    # 转为token id数组，增加batch维度 [batch=1, seq_len]
    token_ids = np.array([[word2id[w] for w in words]])
    print("\nToken IDs：", token_ids)
    print(f"token_ids shape: {token_ids.shape}")

    # Embedding前向计算
    emb_out = embedding_layer.forward(token_ids)
    print(f"\nEmbedding输出shape (batch, seq_len, embed_dim): {emb_out.shape}")
    print("第一个token（I）的嵌入向量：")
    print(emb_out[0, 0])