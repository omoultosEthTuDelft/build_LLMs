import re

def tokenize_text(doc: str):
    with open(doc, 'r', encoding='utf-8') as f:
        raw_text = f.read()

    preprocessed = re.split(r'([,.!;?:_"()\']|--|\s)', raw_text)
    # item.strip(): Removes leading and trailing whitespace from item.
    # if item.strip(): Ensures only non-empty strings are included in the final list.
    preprocessed = [item.strip() for item in preprocessed if item.strip()] 

    all_tokens = sorted(set(preprocessed))  # "set" removes duplicates
    all_tokens.extend(["<|endoftext|>", "<|unk|>"])
    vocab_size = len(all_tokens) # number of unique words 

    vocab = {token: integer for integer, token in enumerate(all_tokens)}
    
    switch_print_vocab = False 
    if switch_print_vocab:
        print(f'dictionary size: {len(vocab.items())}')
        # for i, item in enumerate(vocab.items()):
        #     print(item)
        #     if i>= 50: break
        for i, item in enumerate(list(vocab.items())[-5:]): print(item)
     
    switch_num_characters = False
    if switch_num_characters:
        print(f'number of none-space characters {len(preprocessed)}')
        print(f'total number of characters {len(raw_text)}')
        # print(raw_text[:30])
        print(f'first 30 words from the tokenized text:\n{preprocessed[:30]}')

    switch_vocab_size = False
    if switch_vocab_size:
        print(vocab_size)
        print(all_tokens)

    return raw_text, vocab



if __name__ == '__main__':
    DOC_PATH = '/Users/omoultos/coding/build_LLMs/the-verdict.txt'
    text_data, vocab = tokenize_text(DOC_PATH)
    # print(text_data)






    

