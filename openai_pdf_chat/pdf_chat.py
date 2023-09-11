import pickle  # to save embeddings
import os
from langchain.chat_models import ChatOpenAI  # to deal with chat GPT
from langchain.chains.question_answering import load_qa_chain  # to deal with chat GPT
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter

def main(incoming_msg):
    data_file_path = os.path.join("data_", "data.pkl")

    # script_dir = os.path.dirname(os.path.abspath(__file__))
    # data_dir = os.path.join(script_dir, "data_")
    # data_file = os.path.join(script_dir, "data_", "data.pkl")

    # loader = DirectoryLoader(data_dir, glob="**/*.txt")
    # documents = loader.load()
    # text_splitter = CharacterTextSplitter(chunk_size=10000, chunk_overlap=0)
    # docs = text_splitter.split_documents(documents)
# Load the data.pkl file
    with open(data_file_path, "rb") as f:
        db = pickle.load(f)

    # True --> we worked this file befor



    # incoming_msg = incoming_msg.strip()
    if incoming_msg:
        # print(get_display(incoming_msg))
        print("****", incoming_msg, "****")
        # print("****",type(incoming_msg), "****")
        docs = db.similarity_search(query=incoming_msg, k=1)
        llm = ChatOpenAI(model_name="gpt-3.5-turbo")
        chain = load_qa_chain(llm=llm, chain_type="stuff")
        response = chain.run(question=incoming_msg, input_documents=docs)
        return response