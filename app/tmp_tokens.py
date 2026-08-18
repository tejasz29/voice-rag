from sentence_transformers import SentenceTransformer

m = SentenceTransformer("all-MiniLM-L6-v2")
ids = m.tokenizer("कॉर्पोरेशन क्या है?")["input_ids"]
print("ids:", ids)
print("tokens:", m.tokenizer.convert_ids_to_tokens(ids))