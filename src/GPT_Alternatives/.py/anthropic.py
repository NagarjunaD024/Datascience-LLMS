
!pip install anthropic

import argparse
from anthropic import Anthropic


if __name__ == '__main__':

  """
  for Script Invoking from command line with argparse ex- python3 Copy_of_Anthropic.py "What is constitutional AI?"

  """

    parser = argparse.ArgumentParser()
    parser.add_argument('ai_key', type=str, help='Anthropic access key')
    parser.add_argument('question', type=str, help='A question for Claude')
    args = parser.parse_args()

    anthropic = Anthropic(api_key=args.ai_key)

    completion = anthropic.messages.create(
        model='claude-3-5-sonnet-20241022',
        max_tokens=100,
        messages=[
            {
                'role':'user',
                'content':args.question
             }])

    print(completion.content)