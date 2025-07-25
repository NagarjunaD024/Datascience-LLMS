'''
running through terminal

cd DataScience_GPT.py 
'''
import openai
client = openai.OpenAI()

result = client.chat.completions.create(
    model='gpt-4o',
    messages=[{
        'role':'user', 
        'content':'What is meaning of life?.'
        }],
    max_tokens=1024, 
    stop='happily ever after', 
    temperature=1, 
    presence_penalty=0.5,
    logit_bias={14844:-100},
    n=2)
print(result.choices[0].message.content)