import streamlit as st
import re
import nltk
import tensorflow as tf
from transformers import DistilBertTokenizerFast, TFDistilBertModel
import fitz  
import numpy as np

# Set TensorFlow logging verbosity to suppress warnings
tf.get_logger().setLevel('ERROR')


custom_css = """
<style>
    body {
        background-color: #f0f0f0; /* Set a background color */
        font-family: Arial, sans-serif; /* Change the font */
    }
    .stApp {
        max-width: 800px; /* Set a maximum width for the app content */
        margin: 0 auto; /* Center the app content */
    }
    .stTitle {
        font-size: 24px; /* Customize the title font size */
    }
    .stButton {
        background-color: #008CBA; /* Set a background color for buttons */
        color: #fff; /* Set text color for buttons */
    }
</style>
"""

# Inject custom CSS
st.markdown(custom_css, unsafe_allow_html=True)


nltk.download('punkt')
nltk.download('stopwords')

# Define a custom object scope for loading the model
custom_objects = {
    'TFDistilBertModel': TFDistilBertModel  
}


model = tf.keras.models.load_model('job.h5', custom_objects=custom_objects)

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

# extract text from a PDF file
def extract_text_from_pdf(pdf_file):
    text = ""
    pdf_document = fitz.open(pdf_file)
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        text += page.get_text("text")
    return text


def clean_resume(resume_text):
    clean_text = re.sub('http\S+\s*', ' ', resume_text)
    clean_text = re.sub('RT|cc', ' ', clean_text)
    clean_text = re.sub('#\S+', '', clean_text)
    clean_text = re.sub('@\S+', '  ', clean_text)
    clean_text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', clean_text)
    clean_text = re.sub(r'[^\x00-\x7f]', r' ', clean_text)
    clean_text = re.sub('\s+', ' ', clean_text)
    return clean_text


def predict_category(resume_text, tokenizer):
    cleaned_resume = clean_resume(resume_text)
    input_features = batch_encode(tokenizer, [cleaned_resume])
    prediction_id = model.predict(input_features)[0]
    return prediction_id

def batch_encode(tokenizer, texts, batch_size=256, max_length=128):
    input_ids = []
    attention_mask = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer.batch_encode_plus(batch,
                                             max_length=max_length,
                                             padding="max_length",
                                             truncation=True,
                                             return_attention_mask=True,
                                             return_token_type_ids=False
                                             )
        input_ids.extend(inputs['input_ids'])
        attention_mask.extend(inputs['attention_mask'])

    return tf.convert_to_tensor(input_ids), tf.convert_to_tensor(attention_mask)

def main():
    st.title("Resume Screening App")
    uploaded_file = st.file_uploader('Upload Resume', type=['txt', 'pdf'])

    if uploaded_file is not None:
        try:
            if uploaded_file.type == 'application/pdf':
                resume_text = extract_text_from_pdf(uploaded_file)
            else:
                resume_bytes = uploaded_file.read()
                resume_text = resume_bytes.decode('utf-8')
        except UnicodeDecodeError:
            # If UTF-8 decoding fails, try decoding with 'latin-1'
            resume_text = resume_bytes.decode('latin-1')

        predicted_class = predict_category(resume_text, tokenizer)

        # Define category mapping based on your category IDs
        category_mapping = {
            0: "Database Administrator",
            1: "Network Administrator",
            2: "Project Manager",
            3: "Security Analyst",
            4: "Software Engineer - Backend",
            5: "Systems Administrator",
            6: "Web Developer"
        }
        predicted_class = predicted_class[0]  # Assuming predicted_class is a NumPy array with a single element

        category_name = category_mapping.get(predicted_class, "Unknown")

        st.write("Predicted Category:", category_name)

if __name__ == "__main__":
    main()
