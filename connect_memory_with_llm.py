

from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import ChatHuggingFace
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress info and warning logs
import warnings
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings("ignore", category=DeprecationWarning)


from dotenv import load_dotenv, find_dotenv


load_dotenv(find_dotenv())

# Step 1: Setup LLM (Mistral with HuggingFace)
HF_TOKEN=os.environ.get("HF_TOKEN")
HUGGINGFACE_REPO_ID="mistralai/Mistral-7B-Instruct-v0.3"

def load_llm(huggingface_repo_id):
    base_llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        task="conversational",
        temperature=0.5,
        max_new_tokens=512,
        huggingfacehub_api_token=HF_TOKEN
    )

    chat_llm = ChatHuggingFace(llm=base_llm)
    return chat_llm




# Step 2: Connect LLM with FAISS and Create chain

DB_FAISS_PATH="vectorstore/db_faiss"

CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer. 
Dont provide anything out of the given context

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

def set_custom_prompt(custom_prompt_template):
    prompt=PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt


#Load Database
DB_FAISS_PATH="vectorstore/db_faiss"
embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db=FAISS.load_local(DB_FAISS_PATH,embedding_model,allow_dangerous_deserialization=True)

#Create QA Chain

qa_chain=RetrievalQA.from_chain_type(
    llm=load_llm(HUGGINGFACE_REPO_ID),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k':3}),
    return_source_documents=True,
    chain_type_kwargs={'prompt':set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

# Now invoke with a single query
user_query=input("Write Query Here: ")
response=qa_chain.invoke({'query': user_query})
print("RESULT: ", response["result"])
print("SOURCE DOCUMENTS: ", response["source_documents"])