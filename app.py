import streamlit as st
from streamlit_option_menu import option_menu
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from sentence_transformers import SentenceTransformer, util
from transformers import MarianMTModel, MarianTokenizer

# Set the device for computation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to set custom CSS styles to enhance the UI/UX.
def set_custom_styles():
    st.markdown(
        """
        <style>
        .main { background-color: rgba(255, 255, 255, 0.85); padding: 2rem; border-radius: 15px; }
        .stButton>button { background-color: #4CAF50; color: white; border-radius: 8px; font-size: 16px; padding: 10px 20px; }
        .stTextInput, .stTextArea { border: 2px solid #4CAF50; border-radius: 8px; font-family: 'Segoe UI', sans-serif; font-size: 16px; }
        .header { font-family: 'Segoe UI', sans-serif; color: #4CAF50; text-align: center; }
        .container { display: flex; justify-content: space-between; }
        .text-box { width: 48%; }
        .icon { display: inline-block; margin-right: 10px; }
        .left-textbox { text-align: left; margin-right: 10px; }
        .right-textbox { text-align: right; }
        .hallucination { background-color: #f8d7da; color: #721c24; padding: 10px; border-radius: 10px; border: 2px solid #f5c6cb; }
        .hallucination-header { font-weight: bold; text-align: center; font-size: 18px; }
        .no-hallucination { background-color: #d4edda; color: #155724; padding: 10px; border-radius: 10px; border: 2px solid #c3e6cb; text-align: center; font-size: 16px; }
        </style>
        """,
        unsafe_allow_html=True
    )

# Function to set the background image from URL
def set_background_image(image_url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{image_url}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Function to load a translation model for a specific language pair.
def load_translation_model(language):
    model_name = f'Helsinki-NLP/opus-mt-{language}-en'  # Translate from source language to English
    return MarianMTModel.from_pretrained(model_name).to(device), MarianTokenizer.from_pretrained(model_name)

# Function to translate text from a source language to a target language.
def translate_text(text, src_lang, tgt_lang='en'):
    model, tokenizer = load_translation_model(src_lang)  # Load model for source language
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)  # Tokenize the input text
    inputs = {key: value.to(device) for key, value in inputs.items()}  # Move inputs to device
    translated = model.generate(**inputs)  # Perform translation
    return tokenizer.batch_decode(translated, skip_special_tokens=True)  # Decode translated text

def check_entailment(premise, hypothesis):
    """
    This function checks the entailment relationship between two texts (premise and hypothesis) 
    using a Natural Language Inference (NLI) model.

    Parameters:
    - premise (str): The source text (e.g., conversation or original statement).
    - hypothesis (str): The target text (e.g., the summary or claim to check against the premise).

    Returns:
    - label (str): One of 'contradiction', 'neutral', or 'entailment' based on the NLI model output.
    - entailment_prob (list): The softmax probabilities for each class [contradiction, neutral, entailment].
    """

    # Tokenize the input texts (premise and hypothesis)
    inputs = nli_tokenizer.encode_plus(premise, hypothesis, return_tensors="pt", truncation=True, max_length=512)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Perform inference
    with torch.no_grad():
        logits = nli_model(**inputs).logits

    # Get probabilities
    entailment_prob = torch.softmax(logits, dim=-1)[0].tolist()
    labels = ['contradiction', 'neutral', 'entailment']
    highest_score_index = torch.argmax(torch.tensor(entailment_prob))
    return labels[highest_score_index], entailment_prob

def check_factual_consistency(reference, summary):
    """
    This function checks the factual consistency between a reference text (conversation) and a summary
    using the FactCC model.

    Parameters:
    - reference (str): The original conversation or text to be used as a reference.
    - summary (str): The summary or generated text to check for factual consistency against the reference.

    Returns:
    - factual_prob (list): The softmax probabilities indicating factual consistency.
    """
    inputs = factcc_tokenizer(reference, summary, return_tensors="pt", padding=True, truncation=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        logits = factcc_model(**inputs).logits
    factual_prob = torch.softmax(logits, dim=-1)[0].tolist()
    return factual_prob

def extract_named_entities(text):
    """
    Extracts named entities from a given text using a pre-trained NER model.

    Parameters:
    - text (str): The text from which named entities need to be extracted.

    Returns:
    - entities (dict): A dictionary where the keys are the named entities and the values are their respective entity types.
    """
    entities = ner_model(text)
    return {entity['word']: entity['entity_group'] for entity in entities}

def check_named_entity_consistency(conversation, summary):
    """
    Checks the consistency of named entities between a conversation and its summary.

    Parameters:
    - conversation (list): A list of sentences from the conversation.
    - summary (str): The summary text for which entity consistency will be checked.

    Returns:
    - inconsistencies (dict): A dictionary containing any inconsistencies found.
    """
    summary_entities = extract_named_entities(summary)
    conversation_text = ' '.join(conversation)
    conversation_entities = extract_named_entities(conversation_text)
    inconsistencies = {}
    for entity in summary_entities:
        if entity not in conversation_entities:
            inconsistencies[entity] = {'summary': summary_entities[entity], 'conversation': None}
        else:
            if summary_entities[entity] != conversation_entities[entity]:
                inconsistencies[entity] = {'summary': summary_entities[entity], 'conversation': conversation_entities[entity]}
    return inconsistencies

def detect_hallucination(conversation, summary, src_lang='es', tgt_lang='en'):
    """
    This function detects hallucinations in a summary based on its corresponding conversation.

    Parameters:
    - conversation (list): The conversation from which the summary is derived.
    - summary (str): The summary to be checked for hallucinations.
    - src_lang (str): The source language of the summary (default is 'es' for Spanish).
    - tgt_lang (str): The target language for translation (default is 'en' for English).

    Returns:
    - hallucination_details (list | None): If hallucinations are detected, it returns a list of details about them;
                                           otherwise, it returns None.
    """
    summary_lines = summary.split("\n")
    translated_summaries = translate_text(summary_lines, src_lang, tgt_lang)
    embeddings = retriever_model.encode(conversation, convert_to_tensor=True, device=device)
    hallucination_flag = False
    hallucination_details = []

    for translated_summary in translated_summaries:
        summary_embedding = retriever_model.encode(translated_summary, convert_to_tensor=True, device=device)
        scores = util.pytorch_cos_sim(summary_embedding, embeddings)
        top_k_indices = scores.argsort(descending=True).squeeze().tolist()[:3]
        relevant_segments = [conversation[i] for i in top_k_indices]

        for segment in relevant_segments:
            result_label, entailment_prob = check_entailment(segment, translated_summary)
            factual_prob = check_factual_consistency(segment, translated_summary)

            if result_label == "contradiction" and factual_prob[1] < 0.35:
                hallucination_flag = True
                hallucination_details.append({
                    "segment": segment,
                    "nli_result": result_label,
                    "entailment_prob": entailment_prob[2],
                    "factual_prob": factual_prob[1]
                })

    entity_inconsistencies = check_named_entity_consistency(conversation, summary)
    if entity_inconsistencies:
        hallucination_flag = True
        hallucination_details.append({
            "type": "Named Entity Inconsistency",
            "details": entity_inconsistencies
        })

    return hallucination_details if hallucination_flag else None

# Initialize models and tokenizers at the global scope
nli_model_name = "joeddav/xlm-roberta-large-xnli"
factcc_model_name = "manueldeprada/FactCC"
retriever_model_name = "sentence-transformers/all-MiniLM-L6-v2"

nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
factcc_tokenizer = AutoTokenizer.from_pretrained(factcc_model_name)
nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model_name).to(device)
factcc_model = AutoModelForSequenceClassification.from_pretrained(factcc_model_name).to(device)
retriever_model = SentenceTransformer(retriever_model_name, device=device)

# Entity extraction using Named Entity Recognition (NER) model.
ner_model = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", aggregation_strategy="simple", device=0 if torch.cuda.is_available() else -1)

# Main function
def main():
    # Apply custom CSS styles
    set_custom_styles()

    # Set a background image
    set_background_image("https://wallpapercave.com/wp/wp3287711.jpg")

    # Display header
    st.markdown("<h1 class='header'>Hallucination Detection in Summaries</h1>", unsafe_allow_html=True)

    # Input area
    with st.container():
        col1, col2 = st.columns(2)

        with col1:
            selected = option_menu(
                menu_title=None,
                options=["Conversation"],
                icons=["chat-left-text-fill"],
                menu_icon="cast",
                orientation="horizontal"
            )
            conversation = st.text_area("Conversation", placeholder="Enter the conversation here...", height=300)

        with col2:
            selected = option_menu(
                menu_title=None,
                options=["Summary"],
                icons=["file-text-fill"],
                menu_icon="cast",
                orientation="horizontal"
            )
            summary = st.text_area("Summary", placeholder="Enter the AI-generated summary here...", height=300)

    # Detect hallucinations
    if st.button("Run Hallucination Detection"):
        if conversation and summary:
            hallucination_results = detect_hallucination(conversation.split("\n"), summary)
            if hallucination_results:
                st.markdown("<h2 class='hallucination-header'>Potential Hallucinations Detected:</h2>", unsafe_allow_html=True)
                for detail in hallucination_results:
                    if 'segment' in detail:
                        st.markdown(f"<div class='hallucination'>Segment: {detail['segment']}<br>NLI Result: {detail['nli_result']}<br>Entailment Probability: {detail['entailment_prob']:.2f}<br>Factual Probability: {detail['factual_prob']:.2f}</div>", unsafe_allow_html=True)
                    if 'type' in detail and detail['type'] == "Named Entity Inconsistency":
                        st.markdown("<div class='hallucination'><strong>Named Entity Inconsistencies:</strong><br>", unsafe_allow_html=True)
                        for entity, info in detail['details'].items():
                            st.markdown(f"- {entity}: Summary Entity: {info['summary']}, Conversation Entity: {info['conversation']}</br>", unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='no-hallucination'>No Hallucinations Detected!</div>", unsafe_allow_html=True)
        else:
            st.warning("Please enter both a conversation and summary.")

# Run the app
if __name__ == "__main__":
    main()
