from flask import Flask, request, jsonify
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def get_text_from_files(txt_files):
    text = ""
    for txt_file in txt_files:
        with open(txt_file, 'r', encoding='utf-8') as file:
            text += file.read()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain



def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)
    return response["output_text"]


app = Flask(__name__)


@app.route('/api/append',methods=['POST'])
def append_to_file():
    company = request.json.get('company')
    rounds = request.json.get('rounds')
    description = request.json.get('description')
    others = request.json.get('others')
    print(company,rounds,description,others)
    if not company or not rounds:
        return jsonify({'error':'Content is required. '}),400
    print(company,rounds)
    file_path = "./data/file.txt"
    try:
        with open(file_path,'a') as file:
            file.write("\n")
            file.write("The company name is "+company+"\n")
            file.write(company+" "+"has "+str(rounds)+" rounds"+"\n")
            file.write("The following points give the desciption of the rounds of the company.\n")
            for i in range(len(description)):
                file.write("In Round "+str(i+1)+" topics from which questions asked: "+description[i]['inputField1']+"\n")
                file.write("In Round "+str(i+1)+" list of questions asked: "+description[i]['inputField2']+"\n")
                file.write("For Round "+str(i+1)+" the preparation strategies are: "+description[i]['inputField3']+"\n")
            file.write(others)
            file.write("\n")
        return jsonify({'message': 'Content appended successfully.'}),200
    except Exception as e:
        return jsonify({'error':str(e)}),500

@app.route('/api/getAns',methods=['POST'])
def getAns():
    directory_path = "./data"
    text_files = [os.path.join(directory_path, file) for file in os.listdir(directory_path) if file.endswith('.txt')]
    combined_text = get_text_from_files(text_files)
    print("DVFGDFGEGFDGDFSG>>>>>>>>>>>>>>"+combined_text)
    query = request.json.get("query")
    text_chunks = get_text_chunks(combined_text)
    print(text_chunks)
    get_vector_store(text_chunks)
    if query:
        text=user_input(query)
        print(text)
        return jsonify({'message':text}),200
    print(query)

    return jsonify({'message': 'Content appended successfully.'}),200



if __name__ == '__main__':
    app.run(debug=True)