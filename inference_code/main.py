import shutil
from ocr_inference_utils_code import *
from regex_utils_code import *

shutil.rmtree("results/")

parse_doc(['Description[6].pdf'], 'results/', backend='pipeline')
parse_doc(['AVTIP25000219.pdf'], 'results/', backend='pipeline')

with open("results/AVTIP25000219/auto/AVTIP25000219.md", "r", encoding="utf-8") as f:
    markdown_text_doc1 = f.read()

print(markdown_text_doc1)

with open("/content/results/Description[6]/auto/Description[6].md", "r", encoding="utf-8") as f:
    markdown_text_doc2 = f.read()

print(markdown_text_doc2)

result = validate_documents(markdown_text_doc1, markdown_text_doc2)
print(result)
