
        # Run validation at the end of every epoch
        print(f"\n▶ Validation after epoch {epoch}")
        bleu_score, source_texts, target_texts, predicted_texts, char_bleu, word_bleu, cer, wer, nltk_bleu = run_validation(
            model, 
            val_dataloader,
            tokenizer_src,