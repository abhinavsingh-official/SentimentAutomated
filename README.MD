# SentimentAutomated

**SentimentAutomated** is a high-performance sentiment analysis tool that processes text-based datasets such as tweets, reviews, or user feedback. It classifies content into **Positive**, **Neutral**, and **Negative** sentiments using the RoBERTa model trained on Twitter data. The tool is designed for both quick sampling and full-scale analysis, generating clear CSV outputs and a rich multi-page PDF report with insightful visualizations.

---

## Features

- Uses `cardiffnlp/twitter-roberta-base-sentiment` transformer model
- Generates a detailed PDF report including:
  - Bar Chart, Pie Chart, Donut Chart
  - Word Clouds by sentiment
  - Confidence Scatter Plot and Box Plot
  - Correlation Heatmap
  - Stacked Bar Chart by data chunks
- Outputs prediction confidence for each row
- GPU acceleration using CUDA (required for full performance)
- Two script options:
  - `sentiment_limited.py`: Processes 5000-row sample
  - `sentiment.py`: Processes the full dataset

---

## Folder Structure

sentiment_limited.py
→ Analyzes a 5000-row sample from the dataset

sentiment.py
→ Processes the entire dataset without sampling

results.csv
→ Contains sentiment predictions and confidence scores for each entry

report.pdf
→ Multi-page PDF report with charts, plots, and summary

README.md
→ Documentation and usage instructions for the project

.gitignore
→ Ensures virtual environments and unnecessary files are not tracked by Git

## Requirements

Install dependencies:

pip install torch transformers pandas matplotlib seaborn tqdm wordcloud

## Note
Ensure you have a CUDA-enabled GPU and the correct version of PyTorch installed with GPU support. This project will not run efficiently without CUDA.

## Run
python sentiment_limited.py   # For quick 5000-row sample analysis
python sentiment.py           # For full dataset analysis


## You will be prompted to:

Enter the path to your CSV file

Specify the column containing the text for analysis

## Outputs
results.csv: Predictions and confidence scores

report.pdf: Multi-page visual summary of sentiment analysis

## Results 

![Page_1_Analysis](results_images/Page1(Analysis).png)
![Page_2_Analysis](results_images/sentimentCount.png)
![Page_3_Analysis](results_images/sentimentProportion.png)
![Page_4_Analysis](results_images/sentimentDonut.png)
![Page_5_Analysis](results_images/confidenceScatter.png)
![Page_6_Analysis](results_images/corrHeat.png)
![Page_7_Analysis](results_images/negCloud.png)
![Page_8_Analysis](results_images/posCloud.png)
![Page_9_Analysis](results_images/sentChunk.png)


