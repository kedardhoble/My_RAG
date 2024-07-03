import torch
from transformers import AutoTokenizer, AutoModel
import faiss
import numpy as np
import os

class DocumentRetriever:
    def __init__(self, model_path, data_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)
        self.data_path = data_path
        self.index, self.documents = self.build_index()

    def build_index(self):
        documents = []
        embeddings = []
        for file_name in os.listdir(self.data_path):
            with open(os.path.join(self.data_path, file_name), 'r') as file:
                text = file.read()
                documents.append(text)
                inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
                with torch.no_grad():
                    outputs = self.model(**inputs)
                embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().numpy())

        index = faiss.IndexFlatL2(len(embeddings[0]))
        index.add(np.array(embeddings))

        return index, documents

    def retrieve(self, query, top_k=5):
        inputs = self.tokenizer(query, return_tensors='pt', truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        query_embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        
        distances, indices = self.index.search(np.array([query_embedding]), top_k)
        retrieved_docs = [self.documents[idx] for idx in indices[0]]

        return retrieved_docs
