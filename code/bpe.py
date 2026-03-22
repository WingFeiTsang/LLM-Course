from collections import Counter

class BPE:
    """
    一个简单的 BPE 分词器实现。
    使用特殊符号 '_' 表示空格，以避免与单词内合并混淆。
    """
    def __init__(self):
        self.merges = []          # 合并规则列表，按训练顺序存储 (left, right)
        self.vocab = set()        # 词汇表（字符串集合）
        # 注意：实际实现中通常还会维护 token 到 ID 的映射，但这里为了直观，直接使用字符串

    def fit(self, corpus, vocab_size):
        """
        训练 BPE 模型。

        Args:
            corpus: list of strings，每个字符串是一个文档（可包含空格）。
            vocab_size: 目标词汇表大小（包括初始字符）。
        """
        # 将空格替换为特殊符号 '_'，这样空格也作为一个普通字符参与合并
        corpus = [doc.replace(' ', '_') for doc in corpus]

        # 将每个文档拆分为字符列表
        docs = [list(doc) for doc in corpus]

        # 初始词汇表：所有出现过的字符
        chars = set()
        for doc in docs:
            chars.update(doc)
        self.vocab = set(chars)

        # 如果目标词汇表大小不大于初始大小，则无法合并
        if vocab_size <= len(self.vocab):
            print("目标词汇表太小，无需合并。")
            return

        # 循环合并，直到达到目标词汇表大小
        while len(self.vocab) < vocab_size:
            # 统计所有相邻字符对的频率
            pair_counts = Counter()
            for doc in docs:
                for i in range(len(doc) - 1):
                    pair = (doc[i], doc[i + 1])
                    pair_counts[pair] += 1

            if not pair_counts:
                break  # 没有可合并的对了

            # 找出出现频率最高的相邻对
            most_common_pair = max(pair_counts, key=pair_counts.get)
            left, right = most_common_pair

            # 合并后的新符号
            new_token = left + right

            # 更新词汇表
            self.vocab.add(new_token)

            # 记录合并规则
            self.merges.append((left, right))

            # 用新符号替换所有文档中的 left+right
            new_docs = []
            for doc in docs:
                new_doc = []
                i = 0
                while i < len(doc):
                    if i < len(doc) - 1 and doc[i] == left and doc[i + 1] == right:
                        new_doc.append(new_token)
                        i += 2
                    else:
                        new_doc.append(doc[i])
                        i += 1
                new_docs.append(new_doc)
            docs = new_docs

            # 可选：打印进度
            print(f"合并: {left} + {right} -> {new_token}，当前词汇表大小: {len(self.vocab)}")

        print("训练完成。")

    def encode(self, text):
        """
        将文本编码为 token 列表（字符串列表）。

        Args:
            text: 输入字符串（可包含空格）。

        Returns:
            list of strings: 分词结果。
        """
        # 替换空格
        text = text.replace(' ', '_')
        # 初始化为字符列表
        tokens = list(text)

        # 按照训练时的合并顺序依次应用规则
        for left, right in self.merges:
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == left and tokens[i + 1] == right:
                    new_tokens.append(left + right)  # 合并
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        return tokens

    def decode(self, tokens):
        """
        将 token 列表解码回原始文本。

        Args:
            tokens: list of strings，分词结果。

        Returns:
            str: 还原后的文本（空格已恢复）。
        """
        # 构建反向映射：合并后的 token -> (left, right)
        expansion_map = {}
        for left, right in self.merges:
            merged = left + right
            expansion_map[merged] = (left, right)

        # 递归展开一个 token 为基础字符
        def expand(token):
            if token in expansion_map:
                left, right = expansion_map[token]
                return expand(left) + expand(right)
            else:
                return token

        # 展开所有 token
        expanded_chars = []
        for token in tokens:
            expanded_chars.extend(expand(token))

        # 拼接成字符串，并将 '_' 替换回空格
        text = ''.join(expanded_chars)
        text = text.replace('_', ' ')
        return text


# ------------------ 使用示例 ------------------
if __name__ == "__main__":
    # 训练语料（每个字符串作为一个文档，不包含跨文档合并）
    corpus = [
        "low",
        "lower",
        "new",
        "newer"
    ]

    # 创建 BPE 实例并训练
    bpe = BPE()
    bpe.fit(corpus, vocab_size=10)   # 初始字符有 l,o,w,e,r,n,_（空格）共7个，目标10，可合并3次

    print("\n合并规则:", bpe.merges)
    print("词汇表:", sorted(bpe.vocab))

    # 测试编码和解码
    test_text = "new"
    tokens = bpe.encode(test_text)
    print(f"\n编码 '{test_text}': {tokens}")

    decoded = bpe.decode(tokens)
    print(f"解码后: '{decoded}'")

    # 处理未登录词
    test_text2 = "lowest"
    tokens2 = bpe.encode(test_text2)
    print(f"\n编码 '{test_text2}': {tokens2}")
    print(f"解码后: '{bpe.decode(tokens2)}'")
