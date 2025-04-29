# Transformer 模型数据处理、训练与推理指南

本文档提供了关于如何处理数据集以及如何训练和推理基于Transformer架构的神经机器翻译模型的详细指导。

## 1. 数据集处理

### 1.1 数据集结构

本项目使用HuggingFace的datasets库加载双语翻译数据集。以下是数据处理的主要步骤：

#### 1.1.1 加载数据集

```python
from datasets import load_dataset

# 从HuggingFace库加载数据集
ds_raw = load_dataset(
    f"{config['datasource']}", 
    config.get('dataset_config', None),
    split='train',
    trust_remote_code=config['trust_remote_code']
)
```

默认配置使用`iwslt2017`数据集，配置为中文到英文的翻译(`zh-en`)。可以通过修改`config.py`中的配置来使用其他数据集或语言对。

#### 1.1.2 创建分词器

本项目使用HuggingFace的tokenizers库创建和使用分词器：

```python
def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        # 创建新的分词器
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        
        # 针对中文的特殊处理
        if lang == "zh":
            # 中文字符级别分词
            from tokenizers.pre_tokenizers import Metaspace
            tokenizer.pre_tokenizer = Metaspace(replacement="▁", add_prefix_space=True)
        else:
            # 其他语言使用空格分词
            tokenizer.pre_tokenizer = Whitespace()
            
        # 定义特殊词元并训练分词器
        trainer = WordLevelTrainer(
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], 
            min_frequency=2
        )
        tokenizer.train_from_iterator(
            get_all_sentences(ds, lang), 
            trainer=trainer
        )
        tokenizer.save(str(tokenizer_path))
    else:
        # 加载已有的分词器
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer
```

对于中文和其他语言使用了不同的预分词处理方法，确保各种语言都能得到合适的处理。

#### 1.1.3 划分训练集和验证集

```python
# 将数据集划分为训练集和验证集
train_ds_size = int(0.9 * len(ds_raw))
val_ds_size = len(ds_raw) - train_ds_size
train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])
```

默认将90%的数据用于训练，10%用于验证。

### 1.2 数据集类的实现(BilingualDataset)

`BilingualDataset`类处理原始数据并生成适合模型训练的格式：

```python
class BilingualDataset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        
        # 特殊词元
        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)
        
    def __getitem__(self, idx):
        # 获取源语言和目标语言的文本
        src_target_pair = self.ds[idx]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        # 将文本转换为词元ID
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # 处理填充和添加特殊词元
        # ...
```

主要功能包括：

1. 获取源语言和目标语言文本
2. 将文本转换为词元ID序列
3. 添加特殊词元([SOS], [EOS], [PAD])
4. 确保所有序列长度统一(seq_len)
5. 创建编码器输入、解码器输入和标签
6. 创建注意力掩码

#### 1.2.1 词元处理和填充

```python
# 添加<s>和</s>词元，以及填充
encoder_input = torch.cat(
    [
        self.sos_token,
        torch.tensor(enc_input_tokens, dtype=torch.int64),
        self.eos_token,
        torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64),
    ],
    dim=0,
)

# 解码器输入(仅添加<s>)
decoder_input = torch.cat(
    [
        self.sos_token,
        torch.tensor(dec_input_tokens, dtype=torch.int64),
        torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
    ],
    dim=0,
)

# 标签(仅添加</s>)
label = torch.cat(
    [
        torch.tensor(dec_input_tokens, dtype=torch.int64),
        self.eos_token,
        torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
    ],
    dim=0,
)
```

这里实现了Transformer训练所需的特殊处理:
- 编码器输入: `[SOS] + 源文本 + [EOS] + [PAD]...`
- 解码器输入: `[SOS] + 目标文本 + [PAD]...`
- 标签: `目标文本 + [EOS] + [PAD]...`

#### 1.2.2 注意力掩码的创建

```python
return {
    "encoder_input": encoder_input,  # (seq_len)
    "decoder_input": decoder_input,  # (seq_len)
    "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1, 1, seq_len)
    "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), # (1, seq_len) & (1, seq_len, seq_len)
    "label": label,  # (seq_len)
    "src_text": src_text,
    "tgt_text": tgt_text,
}
```

两种掩码:
1. **编码器掩码**: 防止模型关注填充词元
2. **解码器掩码**: 结合填充掩码和因果掩码(确保模型只能看到当前位置之前的词元)

```python
def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0
```

因果掩码使用上三角矩阵实现，确保位置i只能关注位置0~i。

## 2. Transformer模型架构

### 2.1 模型组件

该项目实现了一个完整的Transformer架构，遵循"Attention Is All You Need"论文的设计。主要组件包括:

1. **输入嵌入 (InputEmbeddings)**: 将词元ID转换为向量表示
2. **位置编码 (PositionalEncoding)**: 添加位置信息
3. **多头自注意力 (MultiHeadAttentionBlock)**: 允许模型关注输入序列的不同部分
4. **前馈网络 (FeedForwardBlock)**: 对每个位置独立应用的全连接层
5. **残差连接 (ResidualConnection)**: 帮助训练深层网络
6. **编码器 (Encoder)**: 处理源语言序列
7. **解码器 (Decoder)**: 生成目标语言序列
8. **投影层 (ProjectionLayer)**: 将解码器输出映射到词汇表

### 2.2 构建Transformer

```python
def build_transformer(src_vocab_size, tgt_vocab_size, src_seq_len, tgt_seq_len, 
                     d_model=512, N=6, h=8, dropout=0.1, d_ff=2048):
    # 创建嵌入层
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # 创建位置编码层
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)
    
    # 创建编码器块
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # 创建解码器块
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    # 创建编码器和解码器
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))
    
    # 创建投影层
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # 创建Transformer模型
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)
    
    # 初始化参数
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
            
    return transformer
```

模型参数:
- `d_model`: 嵌入维度(默认512)
- `N`: 编码器和解码器层数(默认6)
- `h`: 注意力头数(默认8)
- `d_ff`: 前馈网络维度(默认2048)
- `dropout`: Dropout比率(默认0.1)

## 3. 模型训练

### 3.1 训练配置

训练配置在`config.py`中定义:

```python
def get_config():
    return {
        "batch_size": 16,
        "num_epochs": 2,
        "lr": 10**-4,
        "seq_len": 350,
        "d_model": 512,
        "datasource": 'iwslt2017',
        "lang_src": "zh",
        "lang_tgt": "en",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": "latest",
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel",
        "dataset_config": "iwslt2017-zh-en",
        "trust_remote_code": True
    }
```

主要参数:
- `batch_size`: 每批数据大小
- `num_epochs`: 训练轮数
- `lr`: 学习率
- `seq_len`: 序列最大长度
- `datasource`: 数据集名称
- `lang_src`/`lang_tgt`: 源语言/目标语言代码

### 3.2 训练过程

训练过程在`train.py`中实现:

```python
def train_model(config):
    # 准备数据和模型
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    
    # TensorBoard记录
    writer = SummaryWriter(config['experiment_name'])
    
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)
    
    # 损失函数(带标签平滑)
    loss_fn = nn.CrossEntropyLoss(
        ignore_index=tokenizer_src.token_to_id('[PAD]'), 
        label_smoothing=0.1
    ).to(device)
    
    # 训练循环
    for epoch in range(initial_epoch, config['num_epochs']):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        for batch in batch_iterator:
            # 前向传播和反向传播
            # ...
```

训练过程包括:
1. 将数据移至正确设备(CPU/GPU/MPS)
2. 前向传播计算损失
3. 反向传播更新权重
4. 定期验证和保存模型

### 3.3 训练期间的验证

```python
def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step, writer, num_examples=2):
    model.eval()
    
    # 进行验证预测
    with torch.no_grad():
        for batch in validation_ds:
            encoder_input = batch["encoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)
            
            # 贪婪解码生成预测
            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)
            
            # 解码为文本
            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())
            
            # 输出示例
            # ...
        
    # 计算评估指标
    if writer:
        # 字符错误率(CER)
        metric = torchmetrics.CharErrorRate()
        cer = metric(predicted, expected)
        writer.add_scalar('validation cer', cer, global_step)
        
        # 词错误率(WER)
        metric = torchmetrics.WordErrorRate()
        wer = metric(predicted, expected)
        writer.add_scalar('validation wer', wer, global_step)
        
        # BLEU评分
        metric = torchmetrics.BLEUScore()
        bleu = metric(predicted, expected)
        writer.add_scalar('validation BLEU', bleu, global_step)
```

验证过程中计算了翻译质量的多种指标:
- **字符错误率(CER)**: 衡量字符级别的错误
- **词错误率(WER)**: 衡量单词级别的错误
- **BLEU评分**: 机器翻译领域的标准评估指标

## 4. 模型推理

### 4.1 贪婪解码

推理时使用贪婪解码策略生成翻译:

```python
def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # 预计算编码器输出
    encoder_output = model.encode(source, source_mask)
    # 初始化解码器输入为SOS词元
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    
    while True:
        if decoder_input.size(1) == max_len:
            break

        # 创建解码器掩码
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # 计算解码器输出
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # 获取下一个词元(选择概率最高的)
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
        )

        # 如果生成了EOS词元，停止生成
        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)
```

贪婪解码的步骤:
1. 计算编码器输出
2. 从SOS词元开始生成
3. 在每一步选择概率最高的下一个词元
4. 如果生成EOS或达到最大长度，停止生成

### 4.2 更高级的解码策略

对于更高质量的翻译，可以考虑实现以下高级解码策略:

1. **束搜索(Beam Search)**: 保留k个最可能的序列，而不仅仅是最可能的一个
2. **采样解码(Sampling Decoding)**: 从概率分布中采样而不是选择最高概率
3. **Top-k采样**: 只从概率最高的k个词元中采样
4. **Top-p(核采样)**: 从累积概率达到p的最小词元集合中采样

## 5. 模型使用示例

### 5.1 训练新模型

```python
from config import get_config
from train import train_model

# 加载默认配置
config = get_config()

# 可以修改配置
config['batch_size'] = 32
config['num_epochs'] = 5
config['lang_src'] = 'zh'
config['lang_tgt'] = 'en'

# 开始训练
train_model(config)
```

### 5.2 翻译新文本

```python
def translate(sentence, model, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len, device):
    # 将模型设为评估模式
    model.eval()
    
    # 对源语言句子进行分词
    encoder_input = tokenizer_src.encode(sentence).ids
    encoder_input = torch.cat([
        torch.tensor([tokenizer_src.token_to_id('[SOS]')], dtype=torch.int64),
        torch.tensor(encoder_input, dtype=torch.int64),
        torch.tensor([tokenizer_src.token_to_id('[EOS]')], dtype=torch.int64),
    ], dim=0).unsqueeze(0)
    
    # 创建源语言掩码
    encoder_mask = (encoder_input != tokenizer_src.token_to_id('[PAD]')).unsqueeze(0).unsqueeze(0).int().to(device)
    encoder_input = encoder_input.to(device)
    
    # 使用贪婪解码进行翻译
    model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, seq_len, device)
    
    # 解码为目标语言文本
    translation = tokenizer_tgt.decode(model_out.detach().cpu().numpy())
    
    # 后处理
    translation = translation.replace('[SOS]', '').replace('[EOS]', '').replace('[PAD]', '').strip()
    
    return translation
```

## 6. 结论

本指南详细介绍了如何使用Transformer架构实现神经机器翻译系统的完整流程，包括数据处理、模型构建、训练和推理。通过遵循这些步骤，您可以创建自己的翻译系统或将其应用于其他序列到序列的任务。

更多高级主题可以考虑:
1. 实现更高级的解码策略(束搜索等)
2. 使用更大的预训练模型(如T5、BART等)
3. 添加数据增强技术提高模型鲁棒性
4. 实现模型量化以提高推理速度
5. 探索多语言翻译模型的训练方法 