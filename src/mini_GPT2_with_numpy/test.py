import numpy as np

def cross_entropy_loss(logits, targets):
    print(f"【输入】logits shape = {logits.shape}      (batch, seq_len, vocab_size)")
    print(f"【输入】targets shape = {targets.shape}    (batch, seq_len)")
    batch, seq_len, vocab_size = logits.shape

    logits_flat = logits.reshape(-1, vocab_size)
    print(f"\nlogits_flat shape = {logits_flat.shape}    (batch*seq_len, vocab_size)，压平batch和序列维度")

    targets_flat = targets.reshape(-1)
    print(f"targets_flat shape = {targets_flat.shape}  (batch*seq_len,)，标签同步压平成一维")

    max_logits = logits_flat.max(axis=-1, keepdims=True)
    print(f"\nmax_logits shape = {max_logits.shape}      (batch*seq_len, 1)，每行取最大值，用于数值稳定")

    exp_part = np.exp(logits_flat - max_logits)
    print(f"exp_part shape = {exp_part.shape}           (batch*seq_len, vocab_size)")

    sum_exp = exp_part.sum(axis=-1, keepdims=True)
    print(f"sum_exp shape = {sum_exp.shape}             (batch*seq_len, 1)，每行exp求和")

    log_softmax = logits_flat - max_logits - np.log(sum_exp)
    print(f"log_softmax shape = {log_softmax.shape}     (batch*seq_len, vocab_size)，每行是log(softmax)")

    # 取出真实标签位置对应的 logp
    selected_logp = log_softmax[np.arange(len(targets_flat)), targets_flat]
    print("\nselected_logp:", selected_logp)
    print(f"selected_logp shape = {selected_logp.shape} (batch*seq_len,)，只取目标token对应的对数概率")

    loss = -selected_logp.mean()
    print(f"loss 是标量，loss = {loss:.4f}")
    return loss

if __name__ == "__main__":
    np.random.seed(42)  # 固定随机种子，每次运行结果一致，方便对照
    batch = 2
    seq_len = 2
    vocab_size = 5
    logits = np.random.randn(batch, seq_len, vocab_size)
    targets = np.array([[0, 2], [1, 4]])
    print("===== 开始计算交叉熵 =====")
    loss = cross_entropy_loss(logits, targets)