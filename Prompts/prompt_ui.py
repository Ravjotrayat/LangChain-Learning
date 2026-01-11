from langchain_openai import ChatOpenAI
#from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate

#load_dotenv()

st.header('Research Tool')
paper_input = st.selectbox("Select Research paper Name",["Attention is all you need","Transformers for Natural Language Processing","BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding," "GPT-3: Language Models are Few-Shot Learners","Diffusion Models Beat GANs on Image Synthesis"])

style_input = st.selectbox("Selct explanation Style",["Beginner-Friendly","Technical","code-oriented","Mathematical"])

length_input = st.selectbox("Select Length of Explanation",["Short(1-2 paragraphs)","medium(3-4 paragraphs)","Long(detailed explanation)"])

#Template....

template = PromptTemplate(
    template="""


""", input_variables=["paper_input","style_input","length_input"]
)

#Fill the placeholder
prompt = template.invoke({
    "paper_input": paper_input,
    "style_input": style_input,
    "length_input": length_input
})

if st.button('Summarize'):
    result = model.invoke(prompt)
    st.write(result.content)


