import streamlit as st
import sqlite3
from dataclasses import dataclass, fields
from typing import List
import warnings
import inspect
from tqdm import tqdm
from datasets import load_dataset
import os
from utils import query_raven
import base64

# Suppress warnings
warnings.filterwarnings('ignore')

# Dataclass for schema representation there
@dataclass
class Record:
    agent_name: str
    customer_email: str
    customer_order: str
    customer_phone: str
    customer_sentiment: str

# Database initialization here
def initialize_db():
    conn = sqlite3.connect('extracted.db')
    cursor = conn.cursor()
    table_name = "customer_information"
    columns = """
    id INTEGER PRIMARY KEY, 
    agent_name TEXT, 
    customer_email TEXT, 
    customer_order TEXT, 
    customer_phone TEXT, 
    customer_sentiment TEXT
    """
    quoted_table_name = f'"{table_name}"'
    cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name={quoted_table_name}")
    if cursor.fetchone():
        print(f"Table {table_name} already exists.")
    else:
        cursor.execute(f'''CREATE TABLE {quoted_table_name} ({columns})''')
        print(f"Table {table_name} created successfully.")
    conn.commit()
    conn.close()

# Function to execute SQL
def execute_sql(sql: str):
    table_name = "customer_information"
    conn = sqlite3.connect('extracted.db')
    cursor = conn.cursor()
    cursor.execute(sql)
    results = cursor.fetchall()
    conn.close()
    return results

# Function to update knowledge
def update_knowledge(results_list: List[Record]):
    conn = sqlite3.connect('extracted.db')
    cursor = conn.cursor()
    table_name = "customer_information"
    column_names = "agent_name, customer_email, customer_order, customer_phone, customer_sentiment"
    placeholders = ", ".join(["?"] * 5)
    sql = f"INSERT INTO {table_name} ({column_names}) VALUES ({placeholders})"
    for record in results_list:
        record_values = tuple(getattr(record, f.name) for f in fields(record))
        cursor.execute(sql, record_values)
    conn.commit()
    conn.close()

# we will use this function to load and process our unstructured data Load 
def process_dialogues():
    cwd = os.getcwd()
    dialogue_data = load_dataset(cwd + "/customer_service_chatbot", cache_dir="./cache")["train"]
    
    for i in tqdm(range(0, 10)):
        data = dialogue_data[i]
        dialogue_string = data["conversation"].replace("\n\n", "\n")
        
        prompt = "\n" + dialogue_string
        signature = inspect.signature(update_knowledge)
        docstring = update_knowledge.__doc__
        dataclass_schema_representation = '''
        @dataclass
        class Record:
            agent_name : str
            customer_email : str
            customer_order : str
            customer_phone : str
            customer_sentiment : str
        '''
        raven_prompt = f'''{dataclass_schema_representation}\nFunction:\n{update_knowledge.__name__}{signature}\n """{docstring}"""\n\n\nUser Query:{prompt}<human_end>'''
        raven_call = query_raven(raven_prompt)
        exec(raven_call)

# Converting  SQL results to natural language here but no need finally!
def results_to_natural_language(results, question):

    database_result = f"{results}"

    # Construct the full prompt
    full_prompt = f"""
    <s> [INST]
    {database_result}
     Use the information in the prompt to provide an insightful answer to the question.
    Example:
    User Query: Give me the names and phone numbers of the ones who are frustrated and the order numbers?
    Answer: After analysis, I found 3 frustrated customers. Below are the details:
    Question:
    {question} [/INST]
    """
    
    # Querying Raven with the constructed prompt
    grounded_response = query_raven(full_prompt.format(question = question))

    return grounded_response

# Streamlit UI
def main():
    st.set_page_config(page_title="Customer Service Query System", layout="wide")
    @st.cache_data
    def get_base64_of_bin_file(bin_file):
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()

    def set_png_as_page_bg(png_file1):
        bin_str1 = get_base64_of_bin_file(png_file1)
        page_bg_img = f'''
        <style>
        [data-testid="stAppViewContainer"] {{
            background-image: url("data:image/png;base64,{bin_str1}");
            background-size: 150%;
            background-repeat: no-repeat;
            background-attachment: fixed;
            color: white;
        }}
        </style>
        '''
        st.markdown(page_bg_img, unsafe_allow_html=True)

    set_png_as_page_bg("background_1.png")


    # Sidebar
    st.sidebar.title("Navigation")
    st.sidebar.markdown("Use the buttons below to navigate through the application.")
    
    if st.sidebar.button("Process Dialogues"):
        with st.spinner("Processing dialogues..."):
            process_dialogues()
        st.sidebar.success("Dialogues processed and database updated.")

    st.sidebar.markdown("---")
    
    # Let's use some css
    st.markdown("""
        <style>
        .big-font {
            font-size: 20px !important;
        }
        .stTextInput > div > label {
        font-size: 30px !important;
        }
        .stTextInput > div > div > input {
            font-size: 20px !important;
        }
        </style>
        """, unsafe_allow_html=True)
    

    # I m starting the app from her
    st.title("Customer Service LLM")

    # Welcome text 
    st.markdown("""
        <div class="big-font">
            This application allows you to interact with a customer service database using natural language queries.
        </div>
        """, unsafe_allow_html=True)
    
    user_query = st.text_input("", "", key="query_input", placeholder="Type your query here...")

    if st.button("Submit Query"):
        if user_query:
            st.markdown("### Generated SQL Query")
            
            # Formulate the prompt for Raven
            signature = inspect.signature(execute_sql)
            docstring = execute_sql.__doc__
            schema_representation = """
            CREATE TABLE customer_information (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_name TEXT,
                customer_email TEXT,
                customer_order TEXT,
                customer_phone TEXT,
                customer_sentiment TEXT
            );
            """
            raven_prompt = f'''{schema_representation}\nFunction:\n{execute_sql.__name__}{signature}\n"""{docstring}"""\n\n\nUser Query:{user_query}<human_end>'''
            
            # Get the SQL command from Raven
            raven_call = query_raven(raven_prompt)
            st.code(raven_call, language='sql')
            
            # Execute the generated SQL command
            results = eval(raven_call)
            st.markdown("### SQL Raw Results")
            st.table(results)
            
            # Convert results to natural language
            #natural_language_response = results_to_natural_language(results, user_query)
            
            # Display the natural language response
            #st.markdown("### Natural Language Response")
            #st.write(natural_language_response)
        else:
            st.warning("Please enter a query to submit.")

if __name__ == '__main__':
    initialize_db()
    main()
