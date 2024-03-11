from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings
import os


class FaissEngine:

    def __init__(
            self,
            path: str = "faiss"
    ):
        self.path = path
        self.llm = AzureOpenAIEmbeddings()

    # 保存
    def save_vector_store(
            self,
            text_chunks,
            path: str = "faiss"
    ):
        db = FAISS.from_texts(text_chunks, self.llm)
        db.save_local(path)

    # 加载
    def load_vector_store(
            self,
            path: str = "faiss"
    ):
        return FAISS.load_local(path, self.llm)
