with open('/Users/omoultos/coding/build_LLMs/the-verdict.txt', 'r', encoding='utf-8') as f:
    raw_text = f.read()

print(f'total # of characters {len(raw_text)}')
print(raw_text[:99])