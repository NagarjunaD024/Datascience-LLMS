

!pip install google-generativeai

import google.generativeai as genai

if __name__ == '__main__':

   """
  for Script Invoking from command line with argparse ex- python3 Copy_of_gemini.py 'What is the meaning of life?'

  """

    parser = argparse.ArgumentParser()
    parser.add_argument('api_key', type=str, help='Google API key')
    parser.add_argument('question', type=str, help='Question to answer')
    args = parser.parse_args()

    genai.configure(api_key=args.api_key)

    model = genai.GenerativeModel('gemini-1.5-flash')
    reply = model.generate_content(args.question)

    print(reply.text)