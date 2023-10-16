import openai
from timeit import timeit 
import os
from PineconeDB import Database
from local_secrets import pinecone_api_key, pinecone_environment

db = Database(
    index_name = 'chw',
    pinecone_api_key = pinecone_api_key,
    pinecone_environment = pinecone_environment,
    embed_model = "text-embedding-ada-002"
)

query = '''
A female child, age 5, presents with diarrhea.
'''

res = db.retrieve(query)

print(res) # this is the extra context to make our bot smarter

contexts = []
for item in res['matches']:
    quote = item['metadata']['text']
    page = int(item['metadata']['page'])
    pdf_path = item['metadata']['pdf_path']
    contexts.append(f"Quote: {quote} \nPage Number: {page} \nPDF Path: {pdf_path}")

augmented_query = "\n\n---\n\n".join(contexts)+"\n\n-----\n\n"+query

print(augmented_query) # add the additional context to our prompt

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

res = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": primer},
        {"role": "user", "content": augmented_query}
    ]
)

ans = res["choices"][0]["message"]["content"]
print(ans)