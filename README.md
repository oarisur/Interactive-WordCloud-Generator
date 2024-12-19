# Interactive Word Cloud Generator

## Description
This project is an **Interactive Word Cloud Generator** built with **Streamlit** and several **Natural Language Processing (NLP)** techniques. Users can generate visually appealing word clouds from their text data. The app allows for various customizations such as word frequency analysis, stopword filtering, sentiment analysis, and more.

## Features
- **Language Detection**: Automatically detects the language of the input text and applies the relevant stopword list.
- **Word Cloud Customization**: Customize the word cloud with options like background color, font style, and maximum words.
- **Stopword Filtering**: Exclude common words or custom words from the word cloud.
- **Parts of Speech Filtering**: Filter words based on their parts of speech (e.g., nouns, adjectives, verbs).
- **Mask Options**: Create word clouds with a custom shape by uploading an image or drawing on the canvas.
- **Sentiment Analysis**: Analyzes the sentiment of the input text (positive, negative, neutral).
- **Keyword Highlighting**: Highlight specific keywords in the word cloud.

## Project Setup and Usage Guide

### 1. Clone the Repository
Clone the repository from GitHub:
```bash
git clone https://github.com/username/repo_name.git

### 2. Install Dependencies
Navigate into the project directory and install the required libraries using the following command:
pip install -r requirements.txt

### 3. Run the App
Once all dependencies are installed, you can run the Streamlit app:
streamlit run app.py

## Usage

### 1. Input Text
- Upload a `.txt` file or paste your text directly into the provided text area.
  - **Note**: You can only use one method (either upload a file or paste text).

### 2. Customize the Word Cloud
- **Background Color**: Use the color picker to change the word cloud's background.
- **Maximum Words**: Set the maximum number of words to be displayed.
- **Custom Stopwords**: Exclude unwanted words by entering them as a comma-separated list.

### 3. Filter by Parts of Speech
- Choose which parts of speech to include in your word cloud:
  - Nouns
  - Adjectives
  - Verbs
  - Adverbs

### 4. Highlight Keywords
- Enter specific keywords you want to highlight in the word cloud. These keywords will appear in red.

### 5. Choose a Font
- Pick a font style for your word cloud from the available options.

### 6. Add a Mask (Optional)
- Upload an image to use as a mask, or use the drawing tool to create your own custom shape.
  - **Note**: You can use either an uploaded mask image or a custom drawing, but not both at the same time.

### 7. Generate and View Results
- Click the **"Generate Word Cloud"** button to create the word cloud.
- View the generated word cloud along with a word frequency table, sentiment analysis of the text, and top phrases.

### 8. Download the Word Cloud Image
- If you are satisfied with the word cloud, click the **"Download Word Cloud Image"** button to download the image.

## Example Output
_Provide a sample word cloud image here (optional)_


## Technologies Used
- **Streamlit**: A Python library used for building the interactive web app.
- **NLTK**: Natural Language Toolkit used for text preprocessing, tokenization, and part-of-speech tagging.
- **TextBlob**: A library used for sentiment analysis.
- **WordCloud**: Python library for generating word clouds.
- **Pandas**: Used for displaying the word frequency table.
- **Matplotlib**: Used for visualizing the word cloud.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing
Feel free to fork the repository and submit pull requests. If you encounter any issues or have suggestions for improvement, open an issue on the GitHub repository.
