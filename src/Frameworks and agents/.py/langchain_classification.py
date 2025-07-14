
!pip install langchain
!pip install langchain-openai

from langchain_openai import ChatOpenAI
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.runnables.passthrough import RunnablePassthrough

import pandas as pd
from google.colab import userdata

from google.colab import drive, userdata,  output
drive.mount('/content/drive')

def create_chain():
    """ Creates chain for text classification.

    Returns:
        a chain for text classification.
    """
    prompt = ChatPromptTemplate.from_template(
        '{text}\n'
        'Is the sentiment positive or negative?\n'
        'Answer ("Positive"/"Negative")\n')
    llm = ChatOpenAI( openai_api_key= userdata.get('OPENAI_API_KEY'),
        model='gpt-4o', temperature=0,
        max_tokens=1)
    parser = StrOutputParser()
    chain = ({'text':RunnablePassthrough()} | prompt | llm | parser)
    return chain

if __name__ == '__main__':
  """
  for Script Invoking from command line with argparse ex- python3 langchain_classification.py  '/content/drive/My Drive/Colab Notebooks/DataScience+GPT/Data/textanalysis/reviews.csv'
                                      "
  """

    parser = argparse.ArgumentParser()
    parser.add_argument('file_path', type=str, help='Path to input .csv file')
    args = parser.parse_args()

    df = pd.read_csv(args.file_path)
    chain = create_chain()

    results = chain.batch(list(df['text']))
    df['class'] = results
    df.to_csv('result.csv')