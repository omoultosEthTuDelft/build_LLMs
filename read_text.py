
def count_characters_in_document(doc: str):
    with open(doc, 'r', encoding='utf-8') as f:
        raw_text = f.read()

    print(f'total # of characters {len(raw_text)}')
    print(raw_text[:99])

DOC_PATH = '/Users/omoultos/coding/build_LLMs/the-verdict.txt'
count_characters_in_document(DOC_PATH)