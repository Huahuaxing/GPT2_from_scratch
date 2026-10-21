import numpy as np

class LayerNorm:
    def __init__(self, dim, eps=1e-5):
        self.gamma = np.ones(dim)
        self.beta = np.zeros(dim)
        self.eps = eps

    def forward(self, x):
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        return self.gamma * (x - mean) / np.sqrt(var + self.eps) + self.beta


if __name__ == "__main__":
    # 构造输入，沿用上面embedding输出的shape：[batch, seq_len, embed_dim]
    batch_size = 1
    seq_len = 5
    embed_dim = 16

    # 模拟embedding输出作为LayerNorm输入
    x = np.random.randn(batch_size, seq_len, embed_dim) * 0.02

    ln = LayerNorm(dim=embed_dim)
    out = ln.forward(x)

    print("输入 x shape:", x.shape)
    print("LayerNorm输出 shape:", out.shape)
    print("\n输入第一个token向量：")
    print(x[0,0])
    print("\n归一化后第一个token向量：")
    print(out[0,0])