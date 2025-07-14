

!pip install llama-index
!pip install transformers
!pip install python-pptx
!pip install torch 'transformers<4.50' python-pptx Pillow

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
import openai

from google.colab import drive, userdata,  output
drive.mount('/content/drive')

import os
from google.colab import userdata

# This line fetches the secret you just saved and sets it as an environment variable
# for your current Colab session.
try:
    os.environ['OPENAI_API_KEY'] = userdata.get('OPENAI_API_KEY')
    print("OPENAI_API_KEY has been set successfully!")
except Exception as e:
    print("ERROR: Could not set the API key.")
    print("Please make sure you have saved your key in Colab Secrets with the name 'OPENAI_API_KEY'.")
    raise e

if __name__ == '__main__':

  """
  for Script Invoking from command line with argparse ex- python3 llama_extractingtext_multipleformats.py '/content/drive/My Drive/Colab Notebooks/DataScience+GPT/Data/Multiple_format_docs'
                                                                    'How much did the Plantain unit make in 2023?'

  """

    parser = argparse.ArgumentParser()
    parser.add_argument('datadir', type=str, help='Path to data directory')
    parser.add_argument('question', type=str, help='A question to answer')
    args = parser.parse_args()

    documents = SimpleDirectoryReader(args.datadir).load_data()
    index = VectorStoreIndex.from_documents(documents)
    engine = index.as_query_engine()
    answer = engine.query(args.question)
    print(answer)