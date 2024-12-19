import io
import os
import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from matplotlib import font_manager
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams
from collections import Counter
from textblob import TextBlob
import numpy as np
import pandas as pd
from PIL import Image
from nltk import pos_tag
from langdetect import detect, LangDetectException
from streamlit_drawable_canvas import st_canvas

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Tokenizer and lemmatizer setup
lemmatizer = WordNetLemmatizer()

# Part-of-Speech (POS) tagging map for filtering words
pos_map = {
    "Nouns": ["NN", "NNS", "NNP", "NNPS"],
    "Adjectives": ["JJ", "JJR", "JJS"],
    "Verbs": ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"],
    "Adverbs": ["RB", "RBR", "RBS"]
}

# Function to detect the language of the input text
def detect_language(text):
    try:
        return detect(text)
    except LangDetectException:
        return "english"  # Default to English if detection fails

# Function to generate a word cloud from input text
def generate_word_cloud_from_text(
    text, detected_language, max_words, stopword_option, pos_filter, mask=None, bg_color="#ffffff"):
    # Stopwords setup
    stop_words = set(stopwords.words(detected_language))
    if stopword_option:
        custom_stopwords = stopword_option.split(",")
        stop_words.update([word.strip() for word in custom_stopwords if word.strip()])

    # Tokenize the text and apply POS tagging
    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens if word.isalpha()]
    tagged_tokens = pos_tag(tokens)

    # Apply POS filtering based on the selected options
    selected_pos = []
    for pos in pos_filter:
        selected_pos.extend(pos_map[pos])
    filtered_tokens = [
        lemmatizer.lemmatize(word)
        for word, tag in tagged_tokens
        if word not in stop_words and tag in selected_pos
    ]

    # Function to highlight specific keywords in the word cloud
    def keyword_color_func(word, **kwargs):
        return "red" if word in keywords_to_highlight else "black"

    # Determine color function based on toggle state
    color_function = keyword_color_func if keywords_to_highlight else None

    # Generate the word cloud with the filtered tokens
    wordcloud = WordCloud(
        width=canvas_width,
        height=canvas_height,
        background_color=bg_color,
        max_words=max_words,
        mask=mask,
        font_path=font_path,
        contour_color="black",
        contour_width=2,
        color_func=color_function
    ).generate(" ".join(filtered_tokens))

    return wordcloud, filtered_tokens

# Main function to handle user input and generate the word cloud
def handle_text_and_generate_cloud():
    # Check if both a file and user text are provided (invalid input scenario)
    if uploaded_file is not None and user_text:
        st.warning("Please choose only one input method: either upload a file or paste text.")
        return
    
    # Get the text from the uploaded file or user input
    if uploaded_file is not None:
        text = uploaded_file.read().decode("utf-8")
    else:
        text = user_text if user_text else ""

    # If text is provided, proceed with generating the word cloud
    if text:
        # Detect the language of the text
        detected_language = detect_language(text)

        # Create mask from uploaded image or canvas drawing if available
        mask = None
        if mask_image_file is not None:
            try:
                mask = np.array(Image.open(mask_image_file).convert("L"))
                mask = (mask > 128).astype(np.uint8) * 255
            except Exception as e:
                st.warning("Error processing uploaded mask image. Falling back to no mask.")
                mask = None
        elif canvas_result.image_data is not None:
            try:
                canvas_mask = (canvas_result.image_data[:, :, :3].sum(axis=2) < 700).astype(np.uint8) * 255
                mask = np.array(Image.fromarray(canvas_mask).convert("L"))
            except Exception as e:
                st.warning("Error processing canvas drawing. Falling back to no mask.")
                mask = None
        else:
            mask = None

        # Ensure the mask matches the word cloud dimensions
        if mask is not None:
            mask = np.array(Image.fromarray(mask).resize((canvas_width, canvas_height)))

        # Generate word cloud with the given settings
        wordcloud, filtered_tokens = generate_word_cloud_from_text(
            text, detected_language, max_words, stopword_option, pos_filter, mask=mask, bg_color=bg_color
        )

        # Display the generated word cloud
        st.subheader("🌥 Generated Word Cloud")
        fig_width = canvas_width / 100
        fig_height = canvas_height / 100
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)

        # Provide a download button for the word cloud image
        img_buffer = io.BytesIO()
        wordcloud.to_image().save(img_buffer, format='PNG')
        img_buffer.seek(0)
        st.download_button("🗅 Download Word Cloud Image", img_buffer, "wordcloud.png", "image/png")

        # Display word frequency table
        freq_dist = nltk.FreqDist(filtered_tokens)
        word_freq = list(freq_dist.items())
        word_freq.sort(key=lambda x: x[1], reverse=True)
        st.subheader("🕰 Word Frequency Table")
        df = pd.DataFrame(word_freq[:max_words], columns=['Word', 'Frequency'])
        df.insert(0, 'Serial', range(1, len(df) + 1))
        df.set_index('Serial', inplace=True)
        st.dataframe(df, use_container_width=True)

        # Perform sentiment analysis using TextBlob
        blob = TextBlob(text)
        sentiment = blob.sentiment.polarity
        sentiment_label = "Positive" if sentiment > 0 else "Negative" if sentiment < 0 else "Neutral"
        st.subheader(f"🕰 Sentiment Analysis: {sentiment_label}")
        st.write(f"**Polarity Score**: {sentiment:.2f}")

        # Display top phrases (n-grams)
        st.subheader("🔑 Top Phrases")
        n_grams = ngrams(filtered_tokens, 2)
        n_gram_freq = Counter(n_grams)
        for phrase, count in n_gram_freq.most_common(10):
            st.write(f"{' '.join(phrase)}: {count}")

# Layout setup with three columns for different sections

st.set_page_config(layout="wide")
# Title with stars and letters
st.markdown(
    """
    <h1 style="text-align: center; font-size: 3rem">
        <span style="color:rgb(6, 158, 163); font-family: times new roman;">INTERACTIVE WORD CLOUD GENERATOR</span>    
    </h1>
    <h3 style="text-align: center; font-style: italic; color: #FFC0CB;">
        Visualize your text beautifully with customized word clouds
    </h3>
    """,
    unsafe_allow_html=True
)
col1, col2, col3 = st.columns([1, 3, 1])

# Left Column: Configuration Panel
with col1:
    st.subheader("")
    st.subheader("🖋️ Input Text")
    uploaded_file = st.file_uploader("Upload a text file", type="txt")
    user_text = st.text_area("Or paste your text here:")

    st.subheader("🎨 WordCloud Customization")
    bg_color = st.color_picker("Background color", "#ffffff")
    # Add sliders to adjust canvas and word cloud size
    canvas_width = st.slider("Canvas Width", 200, 1200, 800)
    canvas_height = st.slider("Canvas Height", 200, 1200, 400)
    max_words = st.slider("Maximum words", 10, 500, 100)
    stopword_option = st.text_area("Custom stopwords (optional):", placeholder="E.g., stopword1, stopword2, stopword3")

    st.subheader("🔍 Filter by Parts of Speech")
    pos_filter = st.multiselect(
        "Include parts of speech:",
        options=["Nouns", "Adjectives", "Verbs", "Adverbs"],
        default=["Nouns", "Adjectives"]
    )

    st.subheader("✨ Highlight Keywords")
    highlighted_keywords = st.text_area("Keywords to highlight (optional):", placeholder="E.g., keyword1, keyword2, keyword3")
    keywords_to_highlight = set(map(str.strip, highlighted_keywords.split(','))) if highlighted_keywords else set()

    st.subheader("🖋 Font Options")
    available_fonts = font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
    font_names = [font_manager.FontProperties(fname=font).get_name() for font in available_fonts]
    selected_font_name = st.selectbox("Select font", font_names)
    font_path = next(
        (font for font in available_fonts if font_manager.FontProperties(fname=font).get_name() == selected_font_name),
        None
    )
    if font_path is None:
        st.warning("Selected font is unavailable. Using default font.")
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

# Right Column: Mask Options and Drawing Tool
with col3:
    st.subheader("")
    st.subheader("🖼 Mask Options")
    mask_image_file = st.file_uploader("Upload an image for masking (optional)", type=["png"])

    if mask_image_file is not None:
        # Display the uploaded mask image
        st.image(mask_image_file, caption="Uploaded Mask Image", use_container_width=True)

    # If a mask image is uploaded, disable the drawing mode toggle
    if mask_image_file is not None:
        drawing_mode_toggle = False
        st.error("Drawing mode is unavailable when a mask image is present. Please remove the uploaded image to enable drawing mode.")
    else:
        # Enable drawing mode toggle if no mask image is uploaded
        drawing_mode_toggle = st.checkbox("Enable Drawing Mode (optional)", value=False)

    # Set canvas background color based on drawing mode toggle
    canvas_bg_color = "#000000" if drawing_mode_toggle else "#ffffff"

    # Drawing tools
    st.subheader("✏️ Drawing Tool")
    drawing_mode = st.selectbox("Drawing mode", ["freedraw", "line", "rect", "circle", "transform"])
    stroke_width = st.slider("Stroke width", 1, 100, 3)
        
    # Initialize canvas and handle drawing
    canvas_result = st_canvas(
        fill_color="#ffffff", 
        stroke_color="#ffffff", 
        stroke_width=stroke_width,
        background_color=canvas_bg_color, 
        update_streamlit=True,
        width=canvas_width,
        height=canvas_height,
        drawing_mode=drawing_mode,
        key="canvas",
    )
   
# Middle Column: Display Results and Collapsible Instructions
with col2:

    # Collapsible instructions section
    with st.expander("📘 How to Use the Word Cloud Generator"):
        st.markdown(
            """
            Welcome to the **Interactive Word Cloud Generator**! Follow these simple steps to create and customize your word cloud from a text input or file. 
            
            ### **1. Input Text**
            - **Upload a Text File:** If you have a `.txt` file with content, you can upload it by clicking the "Upload a text file" button. Only `.txt` files are supported.
            - **Paste Your Text:** If you prefer, simply paste your text directly into the "Or paste your text here" text area.
            
            #### **Note:**
            You can only choose **one** input method (either upload a file or paste text). Please make sure not to use both simultaneously.
            
            ---
            
            ### **2. Customize Your Word Cloud**
            - **Background Color:** Use the color picker to select your preferred background color for the word cloud. The default is white.
            - **Maximum Words:** Adjust the slider to set how many words you want to appear in your word cloud. The range is between **10** and **500**.
            - **Custom Stopwords:** You can enter additional words you want to exclude from the word cloud (e.g., "the", "and", etc.). Enter them as a comma-separated list. These words will be removed from the analysis.
            
            ---
            
            ### **3. Filter by Parts of Speech**
            - Select which parts of speech you'd like to include in your word cloud by checking options for **Nouns**, **Adjectives**, **Verbs**, and **Adverbs**. 
            - The default is **Nouns** and **Adjectives**, but you can customize this based on your needs.
            
            ---
            
            ### **4. Highlight Keywords**
            - In the **Highlight Keywords** section, enter a list of keywords that you want to highlight in red in your word cloud. Simply type the keywords, separated by commas. The words will appear in **red**, while all other words will be in **black**.
            
            ---
            
            ### **5. Choose a Font**
            - Select a font for your word cloud from the available options. The font will influence the style of your word cloud, making it more personalized.
            - If you don't see your preferred font, the default font will be used. You can choose from system fonts installed on your device.
            
            ---
            
            ### **6. Mask Options**
            - **Upload a Mask Image (Optional):** You can upload an image (e.g., a logo, shape, etc.) to use as a mask. The word cloud will be shaped to fit the image you upload.
            - **Drawing Tool (Optional):** If you prefer, you can draw your own custom mask using the drawing tool. You can switch between different drawing modes (e.g., free draw, line, rectangle, etc.), change stroke width, and toggle the drawing background color (black or white).
            
            #### **Mask Notes:**
            - Ensure your uploaded image is in **PNG** format.
            - The system is designed to use either an uploaded mask image or a custom drawing, both can not be used at the same time. Presence of an uploaded mask image will block the drawing mode.
            
            ---
            
            ### **7. Generate Word Cloud**
            Once you have finished customizing your options, simply click the **"Generate Word Cloud"** button to create your word cloud.
            
            ---
            
            ### **8. Review Results**
            - **Word Cloud:** Your word cloud will be displayed in the middle column. It will reflect all your customizations (e.g., parts of speech, stopwords, keywords, font, mask, etc.).
            - **Download the Image:** If you're satisfied with the word cloud, you can download the image by clicking on the **"Download Word Cloud Image"** button.
            - **Word Frequency Table:** Below the word cloud, you'll see a table showing the frequency of each word used in your text.
            - **Sentiment Analysis:** The sentiment of your text will be analyzed and categorized as **Positive**, **Negative**, or **Neutral**.
            - **Top Phrases:** A list of the top **2-word phrases** (bigrams) in your text will also be shown.
            
            ---
            
            ### **Additional Features**
            - **Language Detection:** The tool automatically detects the language of your text and adjusts stopword filtering accordingly.
            - **Real-time Customization:** You can make changes to the configuration panel at any time, and the word cloud will be re-generated based on your new settings.
            
            ---
            
            ### **Important Notes**
            - Ensure your **input text** is in a **UTF-8** format to prevent any encoding issues.
            - If you encounter an error or the mask image isn't processing correctly, the word cloud will automatically fall back to a default square shape with no mask.
            
            ---
            
            ### **Happy Word Cloud Creation!**
            We hope this tool helps you explore and visualize your text data in creative ways. Feel free to experiment with different configurations and see how the word cloud changes!
            """
        )
    
    # Button to trigger word cloud generation
    st.subheader("")
    st.subheader("")
    if st.button("Generate Word Cloud"):
        handle_text_and_generate_cloud()