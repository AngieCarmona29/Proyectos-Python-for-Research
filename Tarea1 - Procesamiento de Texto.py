import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.metrics.distance import edit_distance
import string

class FAQAssistant:
    def __init__(self, faq_file):
        self.faq_data = self.load_faq(faq_file) 
        self.stop_words = set(stopwords.words('english'))

    def load_faq(self, file_path):
        faq = {}
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            sections = content.split("\n\n")
            for section in sections:
                question, answer = section.split("\n", 1)
                faq[question.strip()] = answer.strip()
        return faq

    def clean_and_tokenize(self, text):
        text = text.lower()
        text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in self.stop_words]
        return tokens

    def find_best_match(self, user_question):
        user_tokens = self.clean_and_tokenize(user_question)
        user_text = " ".join(user_tokens)

        best_match = None
        min_distance = float('inf')

        for question in self.faq_data.keys():
            question_tokens = self.clean_and_tokenize(question)
            question_text = " ".join(question_tokens)
            distance = edit_distance(user_text, question_text)
            if distance < min_distance:
                min_distance = distance
                best_match = question

        return best_match

    def answer_question(self, user_question):
        best_match = self.find_best_match(user_question)
        if best_match:
            return self.faq_data[best_match]
        else:
            return "No pude encontrar una respuesta a tu pregunta."

def main():
    print("Asistente de Preguntas Frecuentes (FAQ) de cultura.")
    print("Escribe 'salir' para terminar la conversación.\n")

    faq_file = "cultura.txt"
    assistant = FAQAssistant(faq_file)

    while True:
        user_input = input("Q:")
        if user_input.lower() == 'salir':
            print("Saliendo del asistente...")
            break

        response = assistant.answer_question(user_input)
        print(f"A: {response}\n")

if __name__ == "__main__":
    main()