
!pip install cohere

import cohere

if __name__ == '__main__':

  """
  for Script Invoking from command line with argparse ex- python3 Copy_of_cohere.py ""What are webconnectors in context of Large Language models?."

  """

    parser = argparse.ArgumentParser()
    parser.add_argument('ai_key', type=str, help='Cohere access key')
    parser.add_argument('question', type=str, help='Answer this question')
    args = parser.parse_args()

    client = cohere.Client(args.ai_key)

    prompt = f'Answer this question: {args.question}'
    result = client.chat(prompt, connectors=[{'id': 'web-search'}])

    print(f'Answer: {result.text}')
    print(f'Web searches: {result.search_results}')
    print(f'Web results: {result.documents}')