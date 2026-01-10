# Using HuggingFaceEndpoint because we are using api call.
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint 
from dotenv import load_dotenv
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="zai-org/GLM-4.7",
    task="text-generation",
    huggingfacehub_api_token=""
)

model = ChatHuggingFace(llm=llm)

result = model.invoke("HI, my name is Ravjot Singh?")
print(result.content)
