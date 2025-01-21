import re

def count_characters_in_document(doc: str):
    with open(doc, 'r', encoding='utf-8') as f:
        raw_text = f.read()

    preprocessed = re.split(r'([,.!;?:_"()\']|--|\s)', raw_text)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]

    print(f'number of none-space characters {len(preprocessed)}')
    print(f'total number of characters {len(raw_text)}')
    # print(raw_text[:99])
    print(preprocessed[:30])

DOC_PATH = '/Users/omoultos/coding/build_LLMs/the-verdict.txt'
count_characters_in_document(DOC_PATH)