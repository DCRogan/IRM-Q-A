from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_openai import AzureChatOpenAI
import os


class LlmEngine:

    def __init__(
            self,
            model_name: str = "gpt-3.5-turbo",
            streaming: bool = False
    ):
        self.model_name = model_name
        self.streaming = streaming
        self.openai = AzureChatOpenAI(model_name="gpt-35-turbo", temperature=0.3)
    # 拆分文本
    def get_text_chunks(
            self,
            text,
            chunk_size: int = 768,
            chunk_overlap: int = 200
    ):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            # chunk_size=768,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        return text_splitter.split_text(text)

    # 获取检索型问答链
    def get_qa_chain(
            self,
            vector_store
    ):
        prompt_template = """基于以下已知内容，简洁和专业的来回答用户的问题。
                                                如果无法从中得到答案，清说"根据已知内容无法回答该问题"
                                                答案请使用中文。
                                                已知内容:
                                                {context}
                                                问题:
                                                {question}"""

        prompt = PromptTemplate(template=prompt_template,
                                input_variables=["context", "question"])

        return RetrievalQA.from_llm(llm=self.openai,
                                    retriever=vector_store.as_retriever(),
                                    prompt=prompt)

    # 获取对话式问答链
    def get_history_chain(
            self,
            vector_store
    ):
        prompt_template = """基于以下已知内容，简洁和专业的来回答用户的问题。
                                                如果无法从中得到答案，清说"根据已知内容无法回答该问题"
                                                答案请使用中文。
                                                已知内容:
                                                {context}
                                                问题:
                                                {question}"""

        prompt = PromptTemplate(template=prompt_template,
                                input_variables=["context", "question"])

        return ConversationalRetrievalChain.from_llm(llm=self.openai,
                                                     retriever=vector_store.as_retriever(),
                                                     combine_docs_chain_kwargs={'prompt': prompt})
