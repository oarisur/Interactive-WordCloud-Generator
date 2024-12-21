# Interactive Word Cloud Generator
Welcome to the Interactive Word Cloud Generator! This tool allows you to visualize your text data in a beautiful and customizable word cloud format. You can either upload a text file or paste your text directly to generate a word cloud. The word cloud is highly customizable, allowing you to modify the background color, select the parts of speech to include, highlight specific keywords, and much more. Additionally, you can choose a mask image to shape the word cloud or even draw your own custom mask.

## Features
- Text Input:

  - Upload a .txt file or paste your text directly.
- Customization Options:

  - Choose a background color for the word cloud.
  - Adjust the size of the canvas for the word cloud.
  - Specify the maximum number of words in the word cloud.
  - Add custom stopwords to exclude specific words.
- Parts of Speech Filter:

  - Filter words by Nouns, Adjectives, Verbs, and Adverbs to refine the word cloud.
- Keyword Highlighting:

  - Highlight specific keywords in red while keeping others in black.
- Font Options:

  - Choose a system-installed font to personalize the appearance of your word cloud.
- Mask Options:

  - Upload an image (e.g., a logo, shape, etc.) to shape the word cloud, or draw your own custom mask.
- Sentiment Analysis:

  - The sentiment of the input text will be analyzed and categorized as Positive, Negative, or Neutral.
- Word Frequency Table:

  - Display a table of word frequencies from the input text.
- Top Phrases (N-grams):

  - View the most common two-word phrases (bigrams) in the text.
## Installation
To run the Interactive Word Cloud Generator, follow these steps to set up the project on your local machine:

### 1. Clone the repository
Clone the repository to your local machine using Git:
[git clone <repository_url>
cd <repository_folder>](https://github.com/oarisur/Interactive-WordCloud-Generator.git)

### 2. Set up a Virtual Environment (optional but recommended)
It is recommended to use a virtual environment to avoid conflicts with other projects. To create and activate a virtual environment, run:

#### For Windows:
````
python -m venv venv
venv\Scripts\activate
````

#### For macOS/Linux:
````
python3 -m venv venv
source venv/bin/activate
````
### 3. Install dependencies
Install the required dependencies using pip and the provided requirements.txt file:
````
pip install -r requirements.txt
````
This will install all the necessary packages such as Streamlit, NLTK, WordCloud, and others.

### 4. Run the Application
Once the dependencies are installed, you can run the Interactive Word Cloud Generator application with the following command:
````
streamlit run wordcloud_generator.py
````
This will open the application in your default web browser. You can now start generating your word clouds!

## How to Use
### 1. Input Text
You can input text in two ways:

- Upload a Text File: Use the "Upload a text file" button to upload a .txt file.
- Paste Your Text: Paste your text into the "Or paste your text here" text area.
  - Note: Only one input method is allowed at a time. You cannot use both simultaneously.

### 2. Customize the Word Cloud
#### Background Color:
- Choose a background color for your word cloud.
#### Canvas Size:
- Adjust the canvas width and canvas height sliders to set the size of your word cloud.
#### Maximum Words:
- Set the maximum number of words to display in the word cloud.
#### Custom Stopwords:
- Enter additional words (comma-separated) to exclude from the word cloud (e.g., common words like "the", "and", etc.).
### 3. Filter by Parts of Speech
Select which parts of speech you want to include:

- Nouns
- Adjectives
- Verbs
- Adverbs
### 4. Highlight Keywords
Enter keywords (comma-separated) that you want to highlight in red.

### 5. Font Selection
Select a font for your word cloud. You can choose from a wide range of pre-installed fonts.

### 6. Mask Options
#### Upload Mask Image:
- Upload a PNG image to use as a mask. The word cloud will be shaped according to the mask.
#### Drawing Mode:
- Alternatively, use the Drawing Tool to create a custom mask. You can choose between different drawing modes (free draw, line, rectangle, etc.).
### 7. Generate the Word Cloud
After customizing the settings, click "Generate Word Cloud" to create the word cloud. The generated word cloud will be displayed in the middle column.

### 8. Results and Analysis
Once the word cloud is generated, the following will be displayed:

- Word Cloud: The visualized word cloud based on your settings.
- Download Image: A button to download the word cloud as a PNG image.
- Word Frequency Table: A table showing the frequency of the words used in the input text.
- Sentiment Analysis: The sentiment analysis results, categorized as Positive, Negative, or Neutral.
- Top Phrases: A list of the most common two-word phrases (bigrams).

## Screenshots
### Homepage
![home_page](https://github.com/user-attachments/assets/7e27be06-ac13-4947-aa5c-f6e4bf4a2263)
### Generated Output
![result1](https://github.com/user-attachments/assets/8d57990d-665d-4f94-99bb-3ccf9201cd03)
![result2](https://github.com/user-attachments/assets/3af3f0e3-db3c-4b57-95fb-ea4240d96f63)

## License
This tool is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Contributions
Contributions are welcome! If you encounter any issues or have ideas for improvements, feel free to submit an issue or a pull request.
