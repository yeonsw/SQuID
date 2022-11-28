from transformers import DPRQuestionEncoderTokenizerFast, BertTokenizerFast, AutoTokenizer

def get_tokenizer(init_checkpoint):
    tokenizer = \
        AutoTokenizer \
            .from_pretrained(
                init_checkpoint, \
            )
    return tokenizer
