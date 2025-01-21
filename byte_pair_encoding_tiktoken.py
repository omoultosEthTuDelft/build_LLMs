# from importlib.metadata import version
import tiktoken

# print(f'tiktoken version: {version("tiktoken")}')
print(f'tiktoken version: {tiktoken.__version__}')

tokenizer = tiktoken.get_encoding('gpt2')

# text = "Hello, do you like tea? <|endoftext|> In the sunlit terraces" "of someunknownPlace."
# text = "<|endoftext|>"
# text = 'ab a b, !.,-_'
# enc_text = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
# strings = tokenizer.decode(integers)


DOC_PATH = '/Users/omoultos/coding/build_LLMs/the-verdict.txt'
with open(DOC_PATH, 'r', encoding='utf-8') as f:
    raw_text = f.read()

enc_text = tokenizer.encode(raw_text)
# print(enc_text)
print(len(enc_text))

enc_sample = enc_text[50:] # remove the first 50 tokens
# print(tokenizer.decode(enc_text[:10]))

# Create input-target pairs
context_size = 4 # how many tokens are included in the input
x = enc_sample[:context_size] # input tokens
y = enc_sample[1:context_size+1] # targets, which are the inputs shifted by 1
print(f'x: {x}')
print(f'y:      {y}')

for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(context, '---->', desired)

print(40*'*')
print(40*'*')

for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(tokenizer.decode(context), '---->', tokenizer.decode([desired]))
