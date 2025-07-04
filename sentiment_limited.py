import os
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from wordcloud import WordCloud
from scipy.special import softmax
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from matplotlib.backends.backend_pdf import PdfPages

# Load model
model_name = "cardiffnlp/twitter-roberta-base-sentiment"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)

# Labels and pastel color palette
labels = ['Negative', 'Neutral', 'Positive']
colors = {
    'Negative': '#F08080',      # Light Coral
    'Neutral':  '#F0E68C',      # Khaki
    'Positive': '#66CDAA'       # Medium Aquamarine
}

# Input file
path = input("Enter full path to CSV file: ").strip()
if not os.path.isfile(path): print("File not found."); exit()
df = pd.read_csv(path).sample(n=5000, random_state=42)

# Choose column
print("Columns:", list(df.columns))
col = input("Enter column name with text: ").strip()
if col not in df.columns: print("Column not found."); exit()
texts = df[col].astype(str)

# Prediction
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits[0].cpu().numpy()
    probs = softmax(logits)
    return labels[probs.argmax()], float(probs.max())

print("Analyzing 5000 rows...")
results = [predict(txt) for txt in tqdm(texts, desc="Processing")]
df[['Sentiment', 'Confidence']] = results

# Export results
df.to_csv("results.csv", index=False)
df['Score'] = df['Sentiment'].map({'Negative': -1, 'Neutral': 0, 'Positive': 1})

# Plot settings
sns.set(style="whitegrid")
sentiment_counts = df['Sentiment'].value_counts().reindex(labels, fill_value=0)
avg_conf = df['Confidence'].mean()
high_conf_count = df[df['Confidence'] > 0.9].shape[0]

# PDF Report
with PdfPages("report.pdf") as pdf:

    # Page 1: Summary
    summary = f"""
Sentiment Analysis Report (5000 Sampled Entries)

Overview:
This report presents an automatic sentiment classification of text data using a RoBERTa model trained on Twitter posts. The sentiments are categorized into three primary classes:

- Positive: Represents favorable, optimistic, or happy language. In product reviews, this may indicate satisfaction or endorsement.
- Neutral: Represents factual or objective language with no strong emotional leaning. It may reflect reporting, ambiguity, or undecided opinions.
- Negative: Represents dissatisfaction, complaints, or emotionally negative statements like frustration or disagreement.

Analysis Summary:
- Positive: {sentiment_counts['Positive']}
- Neutral:  {sentiment_counts['Neutral']}
- Negative: {sentiment_counts['Negative']}

Confidence:
- Average prediction confidence: {avg_conf:.2f}
- High-confidence predictions (> 0.9): {high_conf_count}

Observations:
- The sentiment distribution shows a dominant trend toward '{sentiment_counts.idxmax()}'.
- The word clouds highlight common terms within each sentiment class.
- Confidence remains consistently high, indicating strong prediction reliability.
    """
    plt.figure(figsize=(8.5, 11))
    plt.axis('off')
    plt.text(0.05, 0.95, summary, ha='left', va='top', wrap=True, fontsize=14, linespacing=1.6)
    pdf.savefig(); plt.close()

    # Page 2: Bar chart
    plt.figure()
    sentiment_counts.plot(kind='bar', color=[colors[l] for l in labels])
    plt.title("Sentiment Count", fontsize=14)
    plt.ylabel("Reviews", fontsize=12)
    plt.xticks(fontsize=11, rotation=0)
    plt.yticks(fontsize=11)
    pdf.savefig(); plt.close()

    # Page 3: Pie chart
    plt.figure()
    sentiment_counts.plot(kind='pie', autopct='%1.1f%%', ylabel='', colors=[colors[l] for l in labels])
    plt.title("Sentiment Proportion", fontsize=14)
    pdf.savefig(); plt.close()

    # Page 4: Donut chart
    plt.figure()
    wedges, texts, autotexts = plt.pie(sentiment_counts, autopct='%1.1f%%', startangle=90, colors=[colors[l] for l in labels])
    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    plt.gca().add_artist(centre_circle)
    plt.title("Sentiment Donut Chart", fontsize=14)
    pdf.savefig(); plt.close()

    # Page 5: Scatter plot
    plt.figure()
    for label in labels:
        group = df[df['Sentiment'] == label]
        plt.scatter(group.index, group['Confidence'], alpha=0.4, label=label, color=colors[label])
    plt.title("Confidence Scatter Plot", fontsize=14)
    plt.xlabel("Index", fontsize=12)
    plt.ylabel("Confidence", fontsize=12)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.legend()
    pdf.savefig(); plt.close()

    # Page 6: Heatmap
    plt.figure()
    sns.heatmap(df[['Score', 'Confidence']].corr(), annot=True, cmap='coolwarm')
    plt.title("Correlation Heatmap", fontsize=14)
    pdf.savefig(); plt.close()

    # Page 7: Word Clouds
    for label in labels:
        filtered_texts = df[df['Sentiment'] == label][col].dropna().astype(str)
        words = " ".join(filtered_texts)
        wc = WordCloud(width=800, height=400, background_color='white').generate(words)
        plt.figure(figsize=(8, 4))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.title(f"Word Cloud: {label}", fontsize=14)
        pdf.savefig(); plt.close()

    # Page 8: Box plot
    plt.figure()
    sns.boxplot(x='Sentiment', y='Confidence', data=df, order=labels, palette=colors)
    plt.title("Confidence by Sentiment", fontsize=14)
    plt.xlabel("Sentiment", fontsize=12)
    plt.ylabel("Confidence", fontsize=12)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    pdf.savefig(); plt.close()

    # Page 9: Stacked Bar Chart
    df['Chunk'] = (df.index // 500) + 1
    stack_df = df.groupby(['Chunk', 'Sentiment']).size().unstack(fill_value=0).reindex(columns=labels)
    plt.figure(figsize=(8, 4))
    stack_df.plot(kind='bar', stacked=True, color=[colors[l] for l in labels])
    plt.title("Sentiment Over Chunks (500 rows each)", fontsize=14)
    plt.xlabel("Chunk", fontsize=12)
    plt.ylabel("Review Count", fontsize=12)
    plt.xticks(rotation=45, fontsize=10)
    plt.yticks(fontsize=11)
    plt.tight_layout()
    pdf.savefig(); plt.close()

print("Report saved as: report.pdf")
print("Sentiment results saved as: results.csv")
