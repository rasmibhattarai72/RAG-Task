from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from app import vectorstore, chat_model

class QARetrieval:
    def __init__(self, vectorstore, chat_model, search_kwargs=None):
        """
        Initialize the QARetrieval with the given vectorstore and chat model.
        """
        if search_kwargs is None:
            search_kwargs = {'k': 3}
        
        self.retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)
        self.chat_model = chat_model
        self.custom_prompt_template = """Use the following pieces of information to answer the user's question.
                        If you don't know the answer, just say that you don't know, don't try to make up an answer.

                        Context: {context}
                        Question: {question}

                        Only return the helpful answer below and nothing else.
                        Helpful answer:
                        """
        self.prompt = self.set_custom_prompt()

    def set_custom_prompt(self):
        """
        Creates a custom prompt template for QA retrieval.
        """
        prompt = PromptTemplate(template=self.custom_prompt_template,
                                input_variables=['context', 'question'])
        return prompt

    def perform_qa(self, query):
        """
        Perform the QA retrieval for the given query.
        """
        qa = RetrievalQA.from_chain_type(llm=self.chat_model,
                                         chain_type="stuff",
                                         retriever=self.retriever,
                                         return_source_documents=False,
                                         chain_type_kwargs={"prompt": self.prompt})

        response = qa.invoke({"query": query})
        return response

# Create an instance of QARetrieval
qa_retrieval = QARetrieval(vectorstore=vectorstore, chat_model=chat_model)

# Prompting the user to enter the question
question = input("Please enter your question: ")

# Performing the QA retrieval
response = qa_retrieval.perform_qa(question)


print("Response:", response)
