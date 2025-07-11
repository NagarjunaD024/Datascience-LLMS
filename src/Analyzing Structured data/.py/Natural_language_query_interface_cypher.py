import argparse
import openai
import pandas as pd
import time
import re
from IPython.display import Markdown, display
from google.colab import files
import sqlite3

from openai import OpenAI

from google.colab import drive
from google.colab import userdata
drive.mount('/content/drive')

client = OpenAI(api_key= userdata.get('secretName'))

def create_prompt(question):
    """ Generate prompt to translate question into Cypher query.

    Args:
        question: question about data in natural language.

    Returns:
        prompt for question translation.
    """
    parts = []
    parts += ['Neo4j Database:']
    parts += ['Node labels: Movie, Person']
    parts += ['Relationship types: ACTED_IN, DIRECTED,']
    parts += ['FOLLOWS, PRODUCED, REVIEWED, WROTE']
    parts += ['Property keys: born, name, rating, released']
    parts += ['roles, summary, tagline, title']
    parts += [question]
    parts += ['Cypher Query:']
    return '\n'.join(parts)

def call_llm(prompt):
    """ Query large language model and return answer.

    Args:
        prompt: input prompt for language model.

    Returns:
        Answer by language model.
    """
    for nr_retries in range(1, 4):
        try:
            response = client.chat.completions.create(
                model='gpt-4o',
                messages=[
                    {'role':'user', 'content':prompt}
                    ]
                )
            return response.choices[0].message.content
        except:
            time.sleep(nr_retries * 2)
    raise Exception('Cannot query OpenAI model!')

if __name__ == '__main__':

   """
  for Script Invoking from command line with argparse ex- cd Natural_Language_Query_Interface_Cypher.ipynb.py "How many movies keanu reeves acted are stored?"
  """


    parser = argparse.ArgumentParser()
    parser.add_argument('question', type=str, help='A question about movies')
    args = parser.parse_args()

    prompt = create_prompt(args.question)
    print('--- Prompt ---')
    print(prompt)

    answer = call_llm(prompt)
    print('--- Answer ---')
    print(answer)

    query = re.findall('```cypher(.*)```', answer, re.DOTALL)[0]
    print('--- Query ---')
    print(query)