import gradio as gr
from transformers import pipeline

# Load the model using pipeline
# This will download the model the first time it is run
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

def analyze_sentiment(text):
    if not text or not text.strip():
        return "Please enter some text."
        
    try:
        # Get prediction
        result = sentiment_pipeline(text)[0]
        label = result['label']
        score = result['score']
        
        # Format the output as requested
        if label == 'POSITIVE':
            return f"😊 Positive — Confidence: {score * 100:.1f}%"
        elif label == 'NEGATIVE':
            return f"😞 Negative — Confidence: {score * 100:.1f}%"
        else:
            return f"Neutral/Other — Confidence: {score * 100:.1f}%"
            
    except Exception as e:
        return f"Error analyzing sentiment: {str(e)}"

# Create the Gradio interface using Blocks for full control over the UI
with gr.Blocks(title="Sentiment Analysis Tool", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Sentiment Analysis Tool")
    gr.Markdown("Enter any text and the AI will tell you whether the sentiment is Positive or Negative")
    
    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(
                label="Enter your text here", 
                placeholder="Type your sentence here...",
                lines=3
            )
            submit_btn = gr.Button("Analyze Sentiment", variant="primary")
            
        with gr.Column():
            text_output = gr.Textbox(
                label="Sentiment Result", 
                lines=3,
                interactive=False
            )
            
    # Link the button click to the analyze_sentiment function
    submit_btn.click(
        fn=analyze_sentiment, 
        inputs=text_input, 
        outputs=text_output
    )
    
    # Add the requested test examples
    gr.Examples(
        examples=[
            ["I love this product, it is amazing!"],
            ["This is the worst experience ever"]
        ],
        inputs=text_input
    )

if __name__ == "__main__":
    # Launch the app. Setting share=True will generate a public URL.
    demo.launch(share=True)
