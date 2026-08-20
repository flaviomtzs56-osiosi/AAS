from transformers import pipeline 

print("Loading the AI model... (This could take a moment the first time)")
summarizer = pipeline("summarization", model=("facebook/bart-large-cnn"))
article_text = """
The food manufacturing industry is undergoing a massive digital transformation in 2026. Automation and robotics are taking over repetitive tasks on the assembly line, such as sorting, cutting, and packaging. Meanwhile, artificial intelligence is being deployed to monitor food safety and quality control through high-speed camera tracking systems. Companies that fail to integrate smart sensors and data systems into their factories are quickly falling behind their competitors in production speed and sustainability metrics. """

print ("AI is processing the text...")
summary = summarizer(
    article_text, 
    max_length=40, 
    min_length=15, 
    do_sample=False,
    clean_up_tokenization_spaces=False
)
print("\0--- Ai AUTOMATION SUMMARY ---")
print(summary[0]['summary_text'])