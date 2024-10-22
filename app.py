from flask import Flask, request, render_template
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os

# Suppress symlink warning
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

# Initialize the Flask app
app = Flask(__name__)

# Load pre-trained Sentence-BERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Sample FAQs
faqs = [
    {"id": 1, "question": "What is SARAS AI Institute?", "answer": "SARAS AI Institute is an organization focused on research and education in the field of artificial intelligence. We provide training programs, workshops, and research opportunities to advance the use of AI technologies."},
    {"id": 2, "question": "What programs do you offer?", "answer": "We offer a variety of programs including AI research internships, hands-on technical workshops, certifications in machine learning, and postgraduate courses in AI technologies."},
    {"id": 3, "question": "How can I apply?", "answer": "You can apply by visiting our official website, navigating to the Admissions section, and filling out the application form for the program of your choice."},
    {"id": 4, "question": "What is the application deadline?", "answer": "Application deadlines vary by program. Please refer to the specific program details on our website to find out the exact deadlines."},
    {"id": 5, "question": "Do I need prior experience in AI to apply?", "answer": "While prior experience in AI or machine learning is helpful, we offer beginner programs that do not require any background in the field. Our advanced courses, however, may require prior knowledge of programming and machine learning."},
    {"id": 6, "question": "Where is SARAS AI Institute located?", "answer": "SARAS AI Institute is located in New York City, USA. We also offer online programs for international participants."},
    {"id": 7, "question": "What are the tuition fees?", "answer": "Tuition fees vary depending on the program you choose. Please visit the program-specific page on our website for detailed information on costs and financial aid options."},
    {"id": 8, "question": "Is financial aid available?", "answer": "Yes, we offer financial aid and scholarships for qualified candidates. Details can be found in the Financial Aid section of our website."},
    {"id": 9, "question": "Can international students apply?", "answer": "Yes, international students are welcome to apply. We offer both on-site and online programs, allowing global participation."},
    {"id": 10, "question": "What career opportunities are available after completing a program?", "answer": "Our graduates have gone on to work in a variety of fields including AI research, data science, software development, and tech startups. We provide career guidance and networking opportunities to help students succeed after graduation."}
]

# Extract FAQ questions
faq_questions = [faq['question'] for faq in faqs]

# Encode FAQ questions into embeddings
faq_embeddings = model.encode(faq_questions)

# Function to split a multi-query sentence
def split_queries(user_query):
    queries = re.split(r'(?i)\band\b|\bor\b|[.,;]', user_query)
    return [q.strip() for q in queries if q.strip()]

# Function to find the closest FAQ for each query
def find_closest_faq(query):
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, faq_embeddings)
    closest_faq_idx = similarities.argmax()
    return faqs[closest_faq_idx]['answer']

# Main function to handle multi-query input
def handle_multi_query(user_query):
    sub_queries = split_queries(user_query)
    responses = [find_closest_faq(query) for query in sub_queries]
    return " ".join(responses)

# Flask route to display the homepage
@app.route('/')
def home():
    return render_template('index.html')

# Flask route to handle form submission
@app.route('/get_answer', methods=['POST'])
def get_answer():
    user_query = request.form['query']
    response = handle_multi_query(user_query)
    return render_template('index.html', query=user_query, response=response)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
