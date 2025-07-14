
!pip install langchain
!pip install langchain-openai
!pip install langchainhub
!pip install google-search-results
!pip install langchain openai sqlalchemy
!pip install -U langchain-community

from langchain import hub
prompt = hub.pull('hwchase17/react')
print(prompt.template)

from langchain.agents.load_tools import load_tools
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_openai import ChatOpenAI

from google.colab import drive, userdata,  output
drive.mount('/content/drive')

if __name__ == '__main__':

  """
  for Script Invoking from command line with argparse ex- python3 langchain_agents.py serpapi_api_key '/content/drive/My Drive/Colab Notebooks/DataScience+GPT/Data/structured_data/games.db'
                                                                                     'What was the most sold game in Europe, and how is it played?'

  """

    parser = argparse.ArgumentParser()
    parser.add_argument('serpaikey', type=str, help='SERP API access key')
    parser.add_argument('dbpath', type=str, help='Path to SQLite database')
    parser.add_argument('question', type=str, help='A question to answer')
    args = parser.parse_args()

    llm = ChatOpenAI(temperature=0, model='gpt-4o')
    db = SQLDatabase.from_uri(f'sqlite:///{args.dbpath}')
    extra_tools = load_tools(
        ['serpapi'], serpapi_api_key=args.serpaikey, llm=llm)

    agent = create_sql_agent(
        llm=llm, db=db, verbose=True,
        agent_type='openai-tools',
        extra_tools=extra_tools)
    agent.invoke({'input':args.question})

llm = ChatOpenAI(api_key = userdata.get('OPENAI_API_KEY'),temperature=0, model='gpt-4o')
db_path = '/content/drive/My Drive/Colab Notebooks/DataScience+GPT/Data/structured_data/games.db'
db = SQLDatabase.from_uri(f'sqlite:///{db_path}')
extra_tools = load_tools(
    ['serpapi'], serpapi_api_key= "4efe4e98e408d433c0d2a842e320df23743b16d813d5fce429b5665367c7193c", llm=llm)

agent = create_sql_agent(
    llm=llm, db=db, verbose=True,
    agent_type='openai-tools',
    extra_tools=extra_tools)
agent.invoke({'input': 'What was the most sold game in Europe, and how is it played?'})