import re
from tokenize_text import tokenize_text as tk_txt

class SimpleTokenizerV1():
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s, i in vocab.items()} # Inverse vocabulary that maps token IDs back to the original text tokens

    def encode(self, text):
        """Processes input text into token IDs"""
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
    
    def decode(self,ids):
        """Converts token IDs back into text"""
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"(\'])', r'\1', text) # removes spaces before the specified punctuation
        return text
    
if __name__ == "__main__":
    
    DOC_PATH = '/Users/omoultos/coding/build_LLMs/the-verdict.txt'
    vocab = tk_txt(DOC_PATH)
    tokenizer = SimpleTokenizerV1(vocab)

    text = """"It's the last he painted, you know,"Mrs. Gisburn said with pardonable pride."""
    ids = tokenizer.encode(text)
    
    print(ids)
    print(tokenizer.decode(ids))    


