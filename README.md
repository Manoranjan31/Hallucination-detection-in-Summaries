# Hallucination-detection-in-Summaries
 Develop a system that detects hallucinations in a summary when compared to the original text. Hallucinations in this context refer to information present in the summary that is not supported by the original text.

## Features

- **Conversation Input:** Enter the original conversation for which the summary was generated.
- **Summary Input:** Provide the AI-generated summary to be evaluated.
- **Hallucination Detection:** The application detects potential hallucinations by analyzing:
  - Entailment relationships between the conversation and the summary.
  - Factual consistency of the summary against the conversation.
  - Named entity recognition to identify inconsistencies in named entities.

## Technologies Used

- **Streamlit:** For building the web application interface.
- **PyTorch:** For utilizing deep learning models.
- **Transformers Library:** For Natural Language Processing (NLP) tasks with pre-trained models.
- **Sentence Transformers:** For semantic similarity and embeddings.
- **MarianMT:** For translation tasks.

## Installation

To run this application, you'll need to have Python and the required libraries installed. Follow the steps below:

1. **Clone the repository:**

   ```bash
   git clone 
   cd <repository-directory>

