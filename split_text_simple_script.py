import re

text = 'Test, 1, sofia--, alex? otto; text!'

output = re.split(r'([,.!;?:_"()\']|--|\s)', text)
output_without_spaces = [item for item in output if item.strip()]

print(output)
print(output_without_spaces)