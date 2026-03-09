import codecs

text = codecs.open('check_ocv.txt', 'r', 'utf-16le').read()
with open('check_ocv_utf8.txt', 'w', encoding='utf-8') as f:
    f.write(text)
