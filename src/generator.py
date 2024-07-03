from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class ResponseGenerator:
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

    def generate(self, query, documents):
        # Code to generate a response based on the query and retrieved documents
        context = " ".join(documents)
        inputs = self.tokenizer(query + self.tokenizer.sep_token + context, return_tensors='pt', truncation=True, max_length=512)
        outputs = self.model.generate(**inputs)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return response
