import streamlit as st
import os
import PyPDF2
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from fuzzywuzzy import fuzz
import requests
from bs4 import BeautifulSoup
import io
from streamlit import session_state as state
import base64


# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

RESUME_FOLDER = "dataset"
VECTORIZER_FILE = "tfidf_vectorizer.pkl"
FEATURES_FILE = "resume_features.pkl"
METADATA_FILE = "resume_metadata.pkl"

st.set_page_config(page_title="Enhanced Resume Screening RAG", layout="wide")

class WebResumeLoader:
    def __init__(self):
        self.resumes = []

    def load_from_url(self, url):
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            content_type = response.headers.get('Content-Type', '').lower()
            
            if 'application/pdf' in content_type:
                self._process_pdf(response.content, url)
            elif 'text/html' in content_type:
                self._process_html(response.text, url)
            else:
                print(f"Unsupported content type for URL: {url}")
        
        except requests.RequestException as e:
            print(f"Error fetching URL {url}: {e}")

    def _process_pdf(self, content, url):
        try:
            pdf_file = io.BytesIO(content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            
            self.resumes.append({
                'source_url': url,
                'content': text,
                'file_type': 'pdf'
            })
        except Exception as e:
            print(f"Error processing PDF from {url}: {e}")

    def _process_html(self, html_content, url):
        soup = BeautifulSoup(html_content, 'html.parser')
        resume_text = soup.get_text()
        
        self.resumes.append({
            'source_url': url,
            'content': resume_text,
            'file_type': 'html'
        })

    def get_resumes(self):
        return self.resumes

def extract_text_from_pdf(pdf_path):
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text
    except Exception as e:
        print(f"Failed to read {pdf_path}. Error: {str(e)}")
        return ""

def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    return ' '.join(tokens)

def get_section(line, threshold=80):
    sections = {
        'work_experience': ['experience', 'work history', 'employment'],
        'education': ['education', 'academic background', 'qualifications'],
        'skills': ['skills', 'competencies', 'proficiencies']
    }
    
    for section, keywords in sections.items():
        if any(fuzz.partial_ratio(keyword, line.lower()) >= threshold for keyword in keywords):
            return section
    return None

def split_resume_sections(text):
    sections = []
    current_section = {'title': 'summary', 'content': ''}
    
    for line in text.split('\n'):
        detected_section = get_section(line)
        if detected_section:
            if current_section['content'].strip():
                sections.append(current_section)
            current_section = {'title': detected_section, 'content': ''}
        else:
            current_section['content'] += line + '\n'
    
    if current_section['content'].strip():
        sections.append(current_section)
    return sections

def extract_keywords(text, n=10):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=n)
    tfidf_matrix = vectorizer.fit_transform([text])
    feature_names = vectorizer.get_feature_names_out()
    sorted_items = sorted(zip(tfidf_matrix.tocsr().data, feature_names))
    keywords = [item[1] for item in sorted_items[::-1]]
    return keywords[:n]

def extract_metadata(text):
    sections = split_resume_sections(text)
    skills_section = next((section for section in sections if section['title'] == 'skills'), None)
    skills = extract_keywords(skills_section['content'], n=10) if skills_section else extract_keywords(text, n=10)
    education_section = next((section for section in sections if section['title'] == 'education'), None)
    education = education_section['content'].split('\n')[0] if education_section else ''
    return {
        'skills': skills,
        'education': education
    }

def create_or_load_index():
    if os.path.exists(VECTORIZER_FILE) and os.path.exists(FEATURES_FILE) and os.path.exists(METADATA_FILE):
        with open(VECTORIZER_FILE, 'rb') as f:
            vectorizer = pickle.load(f)
        with open(FEATURES_FILE, 'rb') as f:
            features = pickle.load(f)
            if not isinstance(features, np.ndarray):
                features = features.toarray()
        with open(METADATA_FILE, 'rb') as f:
            metadata = pickle.load(f)
        return vectorizer, features, metadata


    texts = []
    metadata = []
    for filename in os.listdir(RESUME_FOLDER):
        if filename.endswith('.pdf'):
            filepath = os.path.join(RESUME_FOLDER, filename)
            text = extract_text_from_pdf(filepath)
            if text:
                sections = split_resume_sections(text)
                preprocessed_text = ' '.join([preprocess_text(section['content']) for section in sections])
                texts.append(preprocessed_text)
                metadata.append({
                    'filename': filename,
                    'text': text,
                    **extract_metadata(text)
                })

    vectorizer = TfidfVectorizer()
    features = vectorizer.fit_transform(texts)

    with open(VECTORIZER_FILE, 'wb') as f:
        pickle.dump(vectorizer, f)
    with open(FEATURES_FILE, 'wb') as f:
        pickle.dump(features, f)
    with open(METADATA_FILE, 'wb') as f:
        pickle.dump(metadata, f)

    return vectorizer, features, metadata

def add_new_resume(resume_content, filename, vectorizer, features, metadata):
    sections = split_resume_sections(resume_content)
    preprocessed_text = ' '.join([preprocess_text(section['content']) for section in sections])
    new_feature = vectorizer.transform([preprocessed_text])
    
    # Check if features is already a numpy array
    if isinstance(features, np.ndarray):
        features = np.vstack((features, new_feature.toarray()))
    else:
        features = np.vstack((features.toarray(), new_feature.toarray()))
    
    metadata.append({
        'filename': filename,
        'text': resume_content,
        **extract_metadata(resume_content)
    })

    with open(FEATURES_FILE, 'wb') as f:
        pickle.dump(features, f)
    with open(METADATA_FILE, 'wb') as f:
        pickle.dump(metadata, f)

    return features, metadata

def expand_query(query):
    expanded_terms = extract_keywords(query, n=5)
    return ' '.join([query] + expanded_terms)

def search_resumes(vectorizer, features, metadata, query, required_skills, min_match_percentage=10):
    expanded_query = expand_query(query)
    query_vector = vectorizer.transform([preprocess_text(expanded_query)])
    similarities = cosine_similarity(query_vector, features)[0]
    
    results = []
    for i, score in enumerate(similarities):
        if score * 100 >= min_match_percentage:
            resume_skills = set(metadata[i].get('skills', []))
            required_skills_set = set(required_skills)
            skill_match_score = len(resume_skills.intersection(required_skills_set)) / len(required_skills_set) if required_skills_set else 1
            combined_score = (score + skill_match_score) / 2
            results.append({'score': combined_score * 100, 'metadata': metadata[i]})
    
    results.sort(key=lambda x: x['score'], reverse=True)
    return results

def save_uploaded_file(uploaded_file):
    file_path = os.path.join(RESUME_FOLDER, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def save_pdf_from_url(url, content):
    filename = url.split("/")[-1]
    if not filename.endswith('.pdf'):
        filename += '.pdf'
    file_path = os.path.join(RESUME_FOLDER, filename)
    with open(file_path, "wb") as f:
        f.write(content)
    return file_path

def get_pdf_file(file_path):
    with open(file_path, "rb") as f:
        return f.read()
    
def get_pdf_display_string(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    return f'data:application/pdf;base64,{base64_pdf}'

# Streamlit UI
st.title("Enhanced Resume Screening RAG Application")

vectorizer, features, metadata = create_or_load_index()

upload_option = st.radio("Choose upload method:", ("File Upload", "URL"))

if upload_option == "File Upload":
    uploaded_file = st.file_uploader("Upload a new resume (PDF only)", type="pdf")
    if uploaded_file is not None:
        file_path = save_uploaded_file(uploaded_file)
        text = extract_text_from_pdf(file_path)
        features, metadata = add_new_resume(text, uploaded_file.name, vectorizer, features, metadata)
        st.success(f"Resume '{uploaded_file.name}' added successfully!")
else:
    url = st.text_input("Enter resume URL (PDF or HTML):")
    if url:
        loader = WebResumeLoader()
        loader.load_from_url(url)
        resumes = loader.get_resumes()
        if resumes:
            for resume in resumes:
                if resume['file_type'] == 'pdf':
                    file_path = save_pdf_from_url(url, requests.get(url).content)
                    features, metadata = add_new_resume(resume['content'], os.path.basename(file_path), vectorizer, features, metadata)
                else:
                    features, metadata = add_new_resume(resume['content'], resume['source_url'], vectorizer, features, metadata)
            st.success(f"Resume from '{url}' added successfully!")
        else:
            st.error("Failed to load resume from the provided URL.")

if 'job_description' not in state:
    state.job_description = ""
if 'required_skills' not in state:
    state.required_skills = ""

job_description = st.text_area("Enter the job description:", value=state.job_description)
required_skills = st.text_input("Enter required skills (comma-separated):", value=state.required_skills)

# Update the session state
state.job_description = job_description
state.required_skills = required_skills

min_match = st.slider("Minimum match percentage", 0, 100, 10)

if 'search_results' not in state:
    state.search_results = None

search_button = st.button("Search Resumes")

if search_button:
    if not job_description:
        st.error("Please enter a job description.")
    else:
        required_skills_list = [skill.strip() for skill in required_skills.split(',') if skill.strip()]
        state.search_results = search_resumes(vectorizer, features, metadata, job_description, required_skills_list, min_match)

if state.search_results:
    st.subheader("Top Matching Resumes:")
    for i, result in enumerate(state.search_results, 1):
        st.write(f"{i}. {result['metadata']['filename']} (Match Score: {result['score']:.2f}%)")
        with st.expander("View Resume Details"):
            # st.write(f"Skills: {', '.join(result['metadata'].get('skills', []))}")
            # st.write(f"Education: {result['metadata'].get('education', 'Not specified')}")
            
            if result['metadata']['filename'].endswith('.pdf'):
                file_path = os.path.join(RESUME_FOLDER, result['metadata']['filename'])
                if os.path.exists(file_path):
                    pdf_file = get_pdf_file(file_path)
                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button(
                            label="Download Resume",
                            data=pdf_file,
                            file_name=result['metadata']['filename'],
                            mime="application/pdf",
                            key=f"download_{i}"
                        )
                    with col2:
                        if st.button("View PDF", key=f"view_{i}"):
                            pdf_display = get_pdf_display_string(file_path)
                            st.markdown(f'<iframe src="{pdf_display}" width="700" height="1000" type="application/pdf"></iframe>', unsafe_allow_html=True)
                else:
                    st.write("Original PDF not available.")
            elif result['metadata']['filename'].startswith('http'):
                st.write(f"[View Original HTML]({result['metadata']['filename']})")
            else:
                st.text(result['metadata']['text'])
    
    # Display expanded query
    #expanded_query = expand_query(job_description)
    # st.subheader("Expanded Search Query:")
    # st.write(expanded_query)
else:
    if search_button:
        st.info("No matching resumes found. Try adjusting your criteria.")