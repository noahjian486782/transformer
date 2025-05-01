from model import build_transformer
from dataset import BilingualDataset, causal_mask
from config import get_config, get_weights_file_path, latest_weights_file_path

#import torchtext.datasets as datasets
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import LambdaLR

import warnings
from tqdm import tqdm
import os
from pathlib import Path

# Huggingface datasets and tokenizers
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

import torchmetrics
from torch.utils.tensorboard import SummaryWriter

def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # Precompute the encoder output and reuse it for every step
    encoder_output = model.encode(source, source_mask)
    # Initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        # build mask for target
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # calculate output
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # get next token
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
        )

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)

def beam_search_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device, beam_size=5):
    """
    Beam Search解码实现
    Args:
        model: Transformer模型
        source: 输入序列张量
        source_mask: 输入序列掩码
        tokenizer_src: 源语言tokenizer
        tokenizer_tgt: 目标语言tokenizer
        max_len: 最大生成长度
        device: 计算设备
        beam_size: beam search的宽度
    Returns:
        最佳翻译序列
    """
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')
    
    # 预计算编码器输出，并对所有候选复用
    encoder_output = model.encode(source, source_mask)  # (1, seq_len, d_model)
    
    # 候选序列：(分数, 序列, 是否完成)
    # 初始候选仅包含开始符号[SOS]
    candidates = [(0.0, [sos_idx], False)]
    
    for _ in range(max_len):
        # 遍历当前所有候选序列
        candidates_new = []
        
        # 对每个候选序列进行扩展
        for score, seq, is_finished in candidates:
            # 如果序列已经完成（遇到EOS或达到最大长度），保留不变
            if is_finished:
                candidates_new.append((score, seq, is_finished))
                continue
            
            # 将候选序列转为张量，准备输入解码器
            decoder_input = torch.tensor([seq], dtype=torch.long, device=device)  # (1, current_len)
            
            # 构建解码器掩码
            decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
            
            # 计算解码器输出
            out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)
            # 获取最后一个位置的预测
            prob = model.project(out[:, -1])  # (1, vocab_size)
            
            # 获取topk个最有可能的下一个词
            log_probs, next_words = torch.topk(torch.log_softmax(prob, dim=-1), beam_size)
            
            # 扩展每个候选词
            for i in range(beam_size):
                word = next_words[0, i].item()
                log_prob = log_probs[0, i].item()
                
                # 计算新的得分：之前的得分加上新词的对数概率
                new_score = score + log_prob
                new_seq = seq + [word]
                
                # 检查是否完成
                new_is_finished = (word == eos_idx)
                
                candidates_new.append((new_score, new_seq, new_is_finished))
        
        # 保留最好的beam_size个候选
        candidates = sorted(candidates_new, key=lambda x: x[0], reverse=True)[:beam_size]
        
        # 如果所有候选都已完成，提前结束
        if all(is_finished for _, _, is_finished in candidates):
            break
    
    # 返回得分最高的序列
    best_score, best_seq, _ = candidates[0]
    return torch.tensor(best_seq, dtype=torch.long, device=device)

def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step, writer, config, num_examples=2):
    model.eval()
    count = 0

    source_texts = []
    expected = []
    predicted = []

    try:
        # get the console window width
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        # If we can't get the console width, use 80 as default
        console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device) # (b, seq_len)
            encoder_mask = batch["encoder_mask"].to(device) # (b, 1, 1, seq_len)

            # check that the batch size is 1
            assert encoder_input.size(
                0) == 1, "Batch size must be 1 for validation"

            # decode method
            model_out = beam_search_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device, beam_size=config.get('beam_size', 5))

            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)
            
            # Print the source, target and model output
            print_msg('-'*console_width)
            print_msg(f"{f'SOURCE: ':>12}{source_text}")
            print_msg(f"{f'TARGET: ':>12}{target_text}")
            print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

            if count == num_examples:
                print_msg('-'*console_width)
                break
    
    # initialize evaluation metrics variables
    bleu = None
    char_bleu = None
    word_bleu = None
    cer = None
    wer = None
    
    if writer:
        # Evaluate the character error rate
        # Compute the char error rate 
        metric = torchmetrics.CharErrorRate()
        cer = metric(predicted, expected)
        writer.add_scalar('validation cer', cer, global_step)
        writer.flush()

        # Compute the word error rate
        metric = torchmetrics.WordErrorRate()
        wer = metric(predicted, expected)
        writer.add_scalar('validation wer', wer, global_step)
        writer.flush()

        # check if the source language is Chinese
        is_chinese = config['lang_src'] == 'zh' or config['lang_tgt'] == 'zh'
        
        # for Chinese, we calculate character-level BLEU and word-level BLEU
        if is_chinese:
            # 1. character-level BLEU (split text into individual characters)
            char_predicted = [' '.join(list(text)) for text in predicted]
            char_expected = [' '.join(list(text)) for text in expected]
            
            # character-level BLEU calculation
            char_metric = torchmetrics.BLEUScore(n_gram=4)
            char_bleu = char_metric(char_predicted, char_expected)
            writer.add_scalar('validation char_BLEU', char_bleu, global_step)
            
            # 2. use standard BLEU score 
            metric = torchmetrics.BLEUScore()
            word_bleu = metric(predicted, expected)
            writer.add_scalar('validation word_BLEU', word_bleu, global_step)
            
            # use character-level BLEU as the main metric
            bleu = char_bleu
        else:
            # non-Chinese language uses standard BLEU
            metric = torchmetrics.BLEUScore()
            bleu = metric(predicted, expected)
            word_bleu = bleu  # non-Chinese情况下，word_bleu就是标准BLEU
            writer.add_scalar('validation BLEU', bleu, global_step)
        
        writer.flush()
    
    # return all evaluation metrics
    return bleu, source_texts, expected, predicted, char_bleu, word_bleu, cer, wer

def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]

def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        # Most code taken from: https://huggingface.co/docs/tokenizers/quicktour
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        
        # 针对中文的分词处理    
        if lang == "zh":
            # Chinese character-level tokenization, do not use space tokenization
            from tokenizers.pre_tokenizers import Metaspace
            # try to use a compatible way to create Metaspace
            try:
                tokenizer.pre_tokenizer = Metaspace(replacement="▁", add_prefix_space=True)
            except TypeError:
                # if the above call fails, try without the add_prefix_space parameter
                tokenizer.pre_tokenizer = Metaspace(replacement="▁")
        else:
            # other languages use space tokenization
            tokenizer.pre_tokenizer = Whitespace()
            
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_ds(config):
    # It only has the train split, so we divide it overselves
    ds_raw = load_dataset(
        f"{config['datasource']}", 
        config.get('dataset_config', None),  # use the new dataset_config parameter
        split='train',
        trust_remote_code=config['trust_remote_code']  #    directly access the trust_remote_code parameter
    )
    
    # determine the actual source and target languages
    src_lang = config['lang_src'] 
    tgt_lang = config['lang_tgt']

    # Build tokenizers
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, src_lang)
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, tgt_lang)

    # Keep 90% for training, 10% for validation
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    # create datasets
    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, config['seq_len'])

    # Find the maximum length of each sentence in the source and target sentence
    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_tgt}')
    

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config["seq_len"], config['seq_len'], d_model=config['d_model'])
    return model

def train_model(config):
    # Define the device
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
    print("Using device:", device)
    if (device == 'cuda'):
        print(f"Device name: {torch.cuda.get_device_name(device.index)}")
        print(f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB")
    elif (device == 'mps'):
        print(f"Device name: <mps>")
    else:
        print("NOTE: If you have a GPU, consider using it for training.")
        print("      On a Windows machine with NVidia GPU, check this video: https://www.youtube.com/watch?v=GMSjDTU8Zlc")
        print("      On a Mac machine, run: pip3 install --pre torch torchvision torchaudio torchtext --index-url https://download.pytorch.org/whl/nightly/cpu")
    device = torch.device(device)

    # Make sure the weights folder exists
    Path(f"{config['datasource']}_{config['model_folder']}").mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    # Tensorboard
    writer = SummaryWriter(config['experiment_name']) #define the location of the log.

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    # 添加warmup学习率调度器
    warmup_steps = config.get('warmup_steps', 4000)  # 默认使用4000步

    def lr_schedule(step):
        # 实现预热学习率调度：在warmup_steps内线性增加，之后按step的平方根反比例衰减
        if step == 0:
            return 1e-8  # 避免初始为0
        if step < warmup_steps:
            return step / warmup_steps
        return (warmup_steps ** 0.5) / (step ** 0.5)  # 先增加后衰减
    
    scheduler = LambdaLR(optimizer, lr_lambda=lr_schedule)

    # If the user specified a model to preload before training, load it
    initial_epoch = 0
    global_step = 0
    preload = config['preload']
    model_filename = latest_weights_file_path(config) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None
    if model_filename:
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
        
        # 恢复调度器状态，如果存在的话
        if 'scheduler_state_dict' in state:
            scheduler.load_state_dict(state['scheduler_state_dict'])
            print('Scheduler state restored')
    else:
        print('No model to preload, starting from scratch')

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch, config['num_epochs']):
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        for batch in batch_iterator:

            encoder_input = batch['encoder_input'].to(device) # (b, seq_len)
            decoder_input = batch['decoder_input'].to(device) # (B, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # (B, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device) # (B, 1, seq_len, seq_len)

            # Run the tensors through the encoder, decoder and the projection layer
            encoder_output = model.encode(encoder_input, encoder_mask) # (B, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (B, seq_len, d_model)
            proj_output = model.project(decoder_output) # (B, seq_len, vocab_size)

            # Compare the output with the label
            label = batch['label'].to(device) # (B, seq_len)

            # Compute the loss using a simple cross entropy
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            # Log the loss
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            # Backpropagate the loss
            loss.backward()

            # Update the weights
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            # 更新学习率
            scheduler.step()
            # 记录学习率
            writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

            global_step += 1

        # Run validation at the end of every epoch
        print(f"\n▶ Validation after epoch {epoch}")
        bleu_score, source_texts, target_texts, predicted_texts, char_bleu, word_bleu, cer, wer = run_validation(
            model, 
            val_dataloader,
            tokenizer_src,
            tokenizer_tgt,
            config['seq_len'],
            device,
            lambda msg: print(msg),
            global_step,
            writer,
            config,
            num_examples=2
        )
        
        # print BLEU score
        if bleu_score is not None:
            print(f"BLEU score: {bleu_score:.4f}")
            # save detailed evaluation results to a separate file
            # if the directory does not exist, create it
            detailed_eval_dir = Path(f"{config['datasource']}_{config['model_folder']}/detailed_evaluation")
            detailed_eval_dir.mkdir(parents=True, exist_ok=True)
            
            # save detailed evaluation results to a separate file
            detailed_eval_path = detailed_eval_dir / f"detailed_eval_epoch_{epoch}.txt"
            with open(detailed_eval_path, "w", encoding="utf-8") as f:
                for i, (src, tgt, pred) in enumerate(zip(source_texts, target_texts, predicted_texts)):
                    f.write(f"Example {i+1}:\n")
                    f.write(f"Source: {src}\n")
                    f.write(f"Target: {tgt}\n")
                    f.write(f"Predicted: {pred}\n")
                    f.write("\n")
                
                # add evaluation metrics information - ensure all metrics are defined
                if cer is not None:
                    f.write(f"Character Error Rate: {cer:.4f}\n")
                if wer is not None:
                    f.write(f"Word Error Rate: {wer:.4f}\n")
                
                # if the source or target language is Chinese, add character-level and word-level BLEU
                if config['lang_src'] == 'zh' or config['lang_tgt'] == 'zh':
                    if char_bleu is not None:
                        f.write(f"Character-level BLEU: {char_bleu:.4f}\n")
                    if word_bleu is not None:
                        f.write(f"Word-level BLEU: {word_bleu:.4f}\n")
                    f.write(f"Main BLEU (Character-level): {bleu_score:.4f}\n")
                else:
                    f.write(f"BLEU Score: {bleu_score:.4f}\n")
            
            # record BLEU score to log file
            with open(f"{config['datasource']}_{config['model_folder']}/bleu_log.txt", "a", encoding="utf-8") as f:
                if config['lang_src'] == 'zh' or config['lang_tgt'] == 'zh' and char_bleu is not None and word_bleu is not None:
                    f.write(f"Epoch: {epoch}, Char-BLEU: {char_bleu:.4f}, Word-BLEU: {word_bleu:.4f}\n")
                else:
                    f.write(f"Epoch: {epoch}, BLEU: {bleu_score:.4f}\n")
    
        # Save the model at the end of every epoch
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step,
            'scheduler_state_dict': scheduler.state_dict()  # 保存调度器状态
        }, model_filename)


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)
