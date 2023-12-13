import streamlit as st
from PIL import Image
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
import base64
from dotenv import load_dotenv

def main():
    load_dotenv()
    st.set_page_config(page_title="Askus")
    # Logo
    logo_path = "C:\\Users\\pc\\source\\repos\\langchain-ask-pdf\\logo3.png"
    
    title = f'<img src="data:image/png;base64,{base64.b64encode(open(logo_path, "rb").read()).decode()}" alt="Logo" width="200" style="vertical-align: middle">'
    st.markdown(title, unsafe_allow_html=True)


    #Sidebar
    image = Image.open('esi.png')

    st.sidebar.image(image)

    # About
    st.sidebar.title("About Askus:")
    # Paragraph
    st.sidebar.write("L'application Aksus développée par Langchain et utilisant Streamlit est conçue pour vous aider à obtenir des réponses rapides et précises concernant les procédures administratives de différents services. Que vous ayez des questions sur les impôts, les permis, les demandes de visa ou d'autres sujets administratifs, cette application est votre guichet unique pour obtenir les informations nécessaires.")
    st.sidebar.title("""Made by: 
                     Mohamed AZZAM & Taoufiq EL-ABED""")


    # Le fichier
    pdf = "C:\\Users\\pc\\source\\repos\\langchain-ask-pdf\\cni1.pdf"
    
    # Lire & extraire  text
    if pdf is not None:
      pdf_reader = PdfReader(pdf)
      text = ""
      for page in pdf_reader.pages:
        text += page.extract_text()
        
      # Diviser le texte en chunks
      text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
      )
      chunks = text_splitter.split_text(text)
      
      # Creation des embeddings
      embeddings = OpenAIEmbeddings(openai_api_key = "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")
      knowledge_base = FAISS.from_texts(chunks, embeddings)
      
      # Affichage de la requete du user
      user_question = st.text_input("Ask a question about your service:")
      if user_question:
        docs = knowledge_base.similarity_search(user_question)
        
        llm = OpenAI(openai_api_key="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")
        chain = load_qa_chain(llm, chain_type="stuff")
        with get_openai_callback() as cb:
          response = chain.run(input_documents=docs, question=user_question)
          print(cb)
           
        st.write(response)
    
    hide_st_style= """
          <style> footer {visibility: hidden;}
          </style>     
    """
    st.markdown(hide_st_style, unsafe_allow_html=True)
    

if __name__ == '__main__':
    main()
