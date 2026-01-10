from langchain_huggingface import ChatHuggingFace,huggingfacepipeline

llm = huggingfacepipeline.from_model_id(
    model_id="zai-org/GLM-4.7",
    task="text-generation",
    model_kwargs=dict(temperature=0.5, max_new_tokens=100)
    )

model = ChatHuggingFace(llm=llm)

result = model.invoke("HI, my name is Ravjot Singh?")
print(result.content)