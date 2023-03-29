import json
from nltk import word_tokenize
import re
from gensim.models import doc2vec


def greet(message):
    print(message)


def preprocess(text):
    text = text.lower()
    text = word_tokenize(text)
    text = [word for word in text if re.match("[^.?!,;:]", word)]
    return text


json_file = json.loads(open("jeopardy.json", "rb").read())
tokenized_json_sentences = [doc2vec.TaggedDocument(preprocess(record["question"]), [str(i)]) for i, record in enumerate(json_file)]

greet("Hello! I'm Ada, a question answering bot who knows answers to "
      "all the questions from the 'Jeopardy!' game.")

model = doc2vec.Doc2Vec(vector_size=300,
                        window=5,
                        min_count=2,
                        workers=10,
                        epochs=50)

model.build_vocab(tokenized_json_sentences)
model.train(tokenized_json_sentences, total_examples=model.corpus_count, epochs=model.epochs)

while True:
    query = input("Ask me something!\n")
    print("Let's play!")
    preprocessed_input = preprocess(query)

    vector = model.infer_vector(preprocessed_input)
    similar_docs = model.dv.most_similar([vector], topn=1)
    most_similar_index, similarity_probability = similar_docs[0]
    print(f"I know this question: its number is {most_similar_index}. I'm {int(round(similarity_probability, 2) * 100)}% sure of this.")
    print(f"The answer is {json_file[int(most_similar_index)]['answer']}")

    again = input("Do you want to ask me again? (yes/no)\n")

    if again == "no":
        print("It was nice to play with you! Goodbye!")
        break

