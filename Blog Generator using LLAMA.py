import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers

##Function to get response from my LLama model 2 
def getLLamaresponse(input_text,no_words,blog_style):

    ### LLama 2 model
    llm = CTransformers(model = 'llama/llama-2-7b-chat.ggmlv3.q2_K.bin',
                        model_type = 'llama',
                        config = {'max_new_tokens':256,
                                  'temperature' : 0.01})
    
    ##Prompt Template
    template = """
                Write a blog for {blog_style} job profile for a topic {input_text} within {no_words} words.
                """

    prompt = PromptTemplate(input_variables = ['blog_style','input_text','no_words'],
                            template = template)
    
    #Generate the response from LLama 2 model
    response = llm(prompt.format(style = blog_style,text = input_text,n_words = no_words ))
    print(response)
    return response



st.set_page_config(page_title = "Generate blogs",
                   page_icon = 'AG',
                   layout = 'centered',
                   initial_sidebar_state = "collapsed")

st.header("Generate blogs")
input_text = st.text_input("Enter the Blog Topic")

##Creating 2 more columns for additional 2 fields
col1,col2 = st.columns([5,5])

with col1:
    no_words = st.text_input("No of Words")
with col2:
    blog_style = st.selectbox("Writing the blog for ",('Researchers','Data Scientists','Common People'),index = 0)

submit = st.button('Generate')


##Final response
if submit:
    st.write(getLLamaresponse(input_text,no_words,blog_style))