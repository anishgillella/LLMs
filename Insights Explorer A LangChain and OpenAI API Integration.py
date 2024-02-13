## Integrating the code with openai API
import os
from langchain.llms import OpenAI
from requirements import openai_api
import streamlit as st
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SimpleSequentialChain
from langchain.chains import SequentialChain
from langchain.memory import ConversationBufferMemory

#Streamlit framework
#st.title("LangChain Demo with OpenAI API")
st.title("Celebrity Search Result")
input_text = st.text_input("Search the topic you prefer")

os.environ["OPENAI_API_KEY"] = openai_api

#Declaring the LLM
llm = OpenAI(temperature=0.7)

#Prompt Template
first_prompt = PromptTemplate(
    input_variables = ['name'],
    template = "What does {name} do?",

)
#Memory
person_memory_input = ConversationBufferMemory(input_key='name',memory_key='chat_history')
dob_memory = ConversationBufferMemory(input_key='person',memory_key='chat_history')
descr_memory = ConversationBufferMemory(input_key='dob',memory_key='description_history')

chain = LLMChain(llm=llm, prompt= first_prompt, verbose= True,output_key="person",memory=person_memory_input)

second_prompt = PromptTemplate(
    input_variables = ['person'],
    template = "When was {person} born?"
)


chain_2 = LLMChain(llm=llm, prompt= second_prompt, verbose= True,output_key="dob",memory=dob_memory)

#parent_chain = SimpleSequentialChain(chains=[chain, chain_2],verbose=True)
#SimpleSequentialChain only shows the last output

third_prompt = PromptTemplate(
    input_variables = ['dob'],
    template = "Mention 5 major events happened around {dob}?"
)

chain_3 = LLMChain(llm=llm, prompt= third_prompt, verbose= True,output_key="description", memory=descr_memory)

#To show the entire information
parent_chain = SequentialChain(chains=[chain, chain_2,chain_3],
                               input_variables=["name"], 
                               output_variables=['person', 'dob','description'],
                                 verbose=True)



#chain.run()
if input_text:
    st.write(parent_chain.run({'name' : input_text}))

    with st.expander('Person Name'):
        st.info(person_memory_input.buffer)

    with st.expander('Major Events'):
        st.info(descr_memory.buffer)