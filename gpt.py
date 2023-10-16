import PyPDF2
from uuid import uuid4
from tqdm.auto import tqdm
import openai
import pinecone
import datetime
from time import sleep
from langchain.text_splitter import RecursiveCharacterTextSplitter
from IPython.display import Markdown
from local_secrets import pinecone_api_key, pinecone_environment    




pdf_paths = \
[r'ASHA Manuals/ASHA_Handbook-Mobilizing_for_Action_on_Violence_against_Women_English.pdf', 
r'ASHA Manuals/book-no-1.pdf', 
r'ASHA Manuals/book-no-2.pdf',
r'ASHA Manuals/book-no-3.pdf',
r'ASHA Manuals/book-no-4.pdf',
r'ASHA Manuals/book-no-5.pdf',
r'ASHA Manuals/book-no-6.pdf',
r'ASHA Manuals/book-no-7.pdf']

# read every pdf and get chunks
chunks = []
for pdf_path in pdf_paths:
    pdf_file = open(pdf_path, "rb")
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    page_count = len(pdf_reader.pages)
    print("Page count:", page_count)

    def pdf_len(text):
        return len(text)

    print("Reading...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
        length_function=pdf_len,
        separators=["\n\n", "\n", " ", ""]
    )

    for i in tqdm(range(len(pdf_reader.pages))):
        page = pdf_reader.pages[i]
        text = page.extract_text()
        texts = text_splitter.split_text(text)
        chunks.extend([{
            'id': str(uuid4()),
            'text': texts[j],
            'chunk': j,
            'page': i,
            'pdf': pdf_path,
        } for j in range(len(texts))])

import os
#openai.api_key = os.environ.get("OPENAI_API_KEY")


embed_model = "text-embedding-ada-002"

res = openai.Embedding.create(
    input=[
        "Sample document text goes here",
        "there will be several phrases in each batch"
    ], engine=embed_model
)

index_name = 'chw' #instead of doing this, define it like in pineconeDB (Ashok) using the class

print('Connecting to Pinecone')
pinecone.init(
    api_key=pinecone_api_key,  # jaycy account since trash dash is taken
    environment=pinecone_environment  # next to API key in console
)
if index_name not in pinecone.list_indexes():
    print(f'Creating {index_name} index')
    pinecone.create_index(
        index_name,
        dimension=len(res['data'][0]['embedding']),
        metric='dotproduct'
    )

index = pinecone.Index(index_name)
pinecone.describe_index(index_name)

def create_embeddings():
    batch_size = 100
    print("Remembering everything...")
    for i in tqdm(range(0, len(chunks), batch_size)):
        i_end = min(len(chunks), i+batch_size)
        meta_batch = chunks[i:i_end]
        ids_batch = [x['id'] for x in meta_batch]
        texts = [x['text'] for x in meta_batch]
        try:
            res = openai.Embedding.create(input=texts, engine=embed_model)
        except:
            done = False
            while not done:
                sleep(5)
                try:
                    res = openai.Embedding.create(input=texts, engine=embed_model)
                    done = True
                except:
                    pass
        embeds = [record['embedding'] for record in res['data']]
        meta_batch = [{
            'text': x['text'],
            'chunk': x['chunk'],
            'page': x['page'] + 1,
            'pdf_path': x['pdf']
        } for x in meta_batch]
        to_upsert = list(zip(ids_batch, embeds, meta_batch))
        index.upsert(vectors=to_upsert)

# create_embeddings() #PLEASE DONT DUPE. PLEASE DO NOT UNCOMMENT UNLESS YOU KNOW WHAT YOU'RE DOING

print("finished making embeddings")

query = '''
A female child, age 5, presents with diarrhea.
'''

res = openai.Embedding.create(
    input=[query],
    engine=embed_model
)

# retrieve from Pinecone
xq = res['data'][0]['embedding']

# get relevant contexts (including the questions)
res = index.query(xq, top_k=int(1e10), include_metadata=True)

print(res) # this is the extra context to make our bot smarter


#context: put in as context, put into query 
contexts = []
for item in res['matches']:
    quote = item['metadata']['text']
    page = int(item['metadata']['page'])
    pdf_path = item['metadata']['pdf_path']
    contexts.append(f"Quote: {quote} \nPage Number: {page} \nPDF Path: {pdf_path}")

augmented_query = "\n\n---\n\n".join(contexts)+"\n\n-----\n\n"+query

print(augmented_query) # add the additional context to our prompt

# exit()

#given before any query message
primer = """
You are a careful advisor to a community health worker (CHW). She is in rural Bihar. She carries supplies for diagnosis such as a thermometer and a rapid strep test. She carries treatments such as ORS, zinc and paracetamol. Your job is to (in order) tell her questions to ask the patient; exams to perform (such as look in throat, take temperature; propose one or more diagnoses; and create a treatment plan. The treatment plan often involves going to the clinic to see a doctor and get prescription medicines.
The patient, age xx presented with xx 
The relevant clinical guidelines are:
START GUIDELINES
***
END GUIDELINES
Answer solely based on these guidelines. If you cannot find an answer to a question, reply “I do not know.”
If there are facts that would change diagnosis or treatment, be sure to ask questions or perform physical exams. Be sure to have completed questions before starting exams, and finish exams before starting diagnosis and treatment.
What questions would you recommend the CHW ask? Reply at the 8th grade reading level.
"""


print('starting up the bot')

res = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": primer},
        {"role": "user", "content": augmented_query}
    ]
)

ans = res["choices"][0]["message"]["content"]
print(ans)