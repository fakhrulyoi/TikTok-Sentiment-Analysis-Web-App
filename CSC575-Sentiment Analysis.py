{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 132,
   "id": "01c0b609-54e0-49d4-9149-46a56bf0ac36",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "\n",
    "plt.style.use('ggplot')\n",
    "\n",
    "import nltk"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 134,
   "id": "e4df83ed-3a1d-4d71-88ec-30eca2966840",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Reviews</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Alhamdulillah, the result is the best this soa...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>It's really effective in removing dirt..do it ...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Worth buying at an affordable price</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>The noodles are thick and the mushroom powder ...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Thanks seller, my item arrived safely in good ...</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                             Reviews\n",
       "0  Alhamdulillah, the result is the best this soa...\n",
       "1  It's really effective in removing dirt..do it ...\n",
       "2                Worth buying at an affordable price\n",
       "3  The noodles are thick and the mushroom powder ...\n",
       "4  Thanks seller, my item arrived safely in good ..."
      ]
     },
     "execution_count": 134,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df = pd.read_csv('Downloads/Degree/Sem 6/CSC575/Project/CSC575_sentiment_data.csv')\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "813c597c-30f3-445b-a792-b6c0c6a066a8",
   "metadata": {},
   "outputs": [],
   "source": [
    "# ax = df['star_rating'].value_counts().sort_index() \\\n",
    "#     .plot(kind='bar',\n",
    "#           title='Count of Reviews by Stars',\n",
    "#           figsize=(10, 5))\n",
    "# ax.set_xlabel('Review Stars')\n",
    "# plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ba8157f6-f1a4-497a-b584-325d9f92e6a3",
   "metadata": {},
   "source": [
    "# NLTK's SentimentIntensityAnalyzer\n",
    "### To get label Positive, Neutral and Negative"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 153,
   "id": "a966d9c0-08b2-4552-8a63-35c528cacbc0",
   "metadata": {},
   "outputs": [],
   "source": [
    "from transformers import AutoTokenizer\n",
    "from transformers import AutoModelForSequenceClassification, pipeline\n",
    "from scipy.special import softmax"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 152,
   "id": "5141f40a-a2be-4d67-a3a5-8e6e9d506791",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Step 1: Load the RoBERTa Model and Tokenizer\n",
    "MODEL = f\"cardiffnlp/twitter-roberta-base-sentiment\"\n",
    "tokenizer = AutoTokenizer.from_pretrained(MODEL)\n",
    "model = AutoModelForSequenceClassification.from_pretrained(MODEL)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 154,
   "id": "1163b70e-7113-4374-a533-9eb5ea407580",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[{'label': 'LABEL_2', 'score': 0.9852644801139832}]\n"
     ]
    }
   ],
   "source": [
    "sentiment_analyzer = pipeline(\"sentiment-analysis\", model=model, tokenizer=tokenizer)\n",
    "\n",
    "text = \"i love tiktok\"\n",
    "test = sentiment_analyzer(text)\n",
    "\n",
    "print(test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 148,
   "id": "15add16a-5b33-4060-a957-80e9d21465cb",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Reviews</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Alhamdulillah, the result is the best this soa...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>It's really effective in removing dirt..do it ...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Worth buying at an affordable price</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>The noodles are thick and the mushroom powder ...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Thanks seller, my item arrived safely in good ...</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                             Reviews\n",
       "0  Alhamdulillah, the result is the best this soa...\n",
       "1  It's really effective in removing dirt..do it ...\n",
       "2                Worth buying at an affordable price\n",
       "3  The noodles are thick and the mushroom powder ...\n",
       "4  Thanks seller, my item arrived safely in good ..."
      ]
     },
     "execution_count": 148,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Upload csv. file\n",
    "df = pd.read_csv('Downloads/Degree/Sem 6/CSC575/Project/CSC575_sentiment_data.csv')\n",
    "\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 158,
   "id": "c79119e7-e692-46ed-9b66-225703929718",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Define a function to convert the labels\n",
    "def convert_labels(result):\n",
    "    label_mapping = {\n",
    "        'LABEL_0': 'Negative',\n",
    "        'LABEL_1': 'Neutral',\n",
    "        'LABEL_2': 'Positive'\n",
    "    }\n",
    "    # Map the label and return the updated result\n",
    "    result['label'] = label_mapping.get(result['label'], result['label'])\n",
    "    return result"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 160,
   "id": "e7d0c320-c71a-4dab-a797-5c83698a07f4",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "                                               Reviews Sentiment_Label  \\\n",
      "0    Alhamdulillah, the result is the best this soa...        Positive   \n",
      "1    It's really effective in removing dirt..do it ...        Positive   \n",
      "2                  Worth buying at an affordable price        Positive   \n",
      "3    The noodles are thick and the mushroom powder ...        Positive   \n",
      "4    Thanks seller, my item arrived safely in good ...        Positive   \n",
      "..                                                 ...             ...   \n",
      "137  It's easy to browse through TikTok Shop, but I...         Neutral   \n",
      "138  While TikTok Shop has many sellers, the qualit...         Neutral   \n",
      "139  TikTok Shop offers a variety of products, but ...         Neutral   \n",
      "140  The shopping feature on TikTok Shop is conveni...        Positive   \n",
      "141  Sellers on TikTok Shop use engaging content to...         Neutral   \n",
      "\n",
      "     Sentiment_Score  \n",
      "0           0.958227  \n",
      "1           0.767554  \n",
      "2           0.660980  \n",
      "3           0.961044  \n",
      "4           0.981782  \n",
      "..               ...  \n",
      "137         0.583255  \n",
      "138         0.823654  \n",
      "139         0.580903  \n",
      "140         0.723374  \n",
      "141         0.723635  \n",
      "\n",
      "[142 rows x 3 columns]\n"
     ]
    }
   ],
   "source": [
    "all_sentiment = []\n",
    "\n",
    "# Nak guna column reviews\n",
    "for index, row in df.iterrows():\n",
    "    text = row['Reviews']\n",
    "    \n",
    "    # Run sentiment analysis on the review text\n",
    "    test_sentiment = sentiment_analyzer(text)\n",
    "    \n",
    "    # Convert the labels for each result\n",
    "    converted_result = convert_labels(test_sentiment[0])  # Take the first result only\n",
    "    all_sentiment.append(converted_result)\n",
    "\n",
    "# Create a new DataFrame with the original reviews and their sentiment labels\n",
    "df['Sentiment_Label'] = [result['label'] for result in all_sentiment]\n",
    "df['Sentiment_Score'] = [result['score'] for result in all_sentiment]\n",
    "\n",
    "# Display the updated DataFrame\n",
    "print(df[['Reviews', 'Sentiment_Label', 'Sentiment_Score']])\n",
    "\n",
    "df.to_csv('Downloads/Degree/Sem 6/CSC575/Project/CSC575_sentiment_data2.csv', index=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 168,
   "id": "2ca361ba-d90f-4141-8d64-c05f3ae36761",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(150, 3)\n",
      "\n",
      "Missing values in each column:\n",
      "Reviews            0\n",
      "Sentiment_Label    0\n",
      "Sentiment_Score    0\n",
      "dtype: int64\n",
      "\n",
      "Model Accuracy: 0.62\n",
      "\n",
      "Classification Report:\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "    Negative       0.48      0.77      0.59        13\n",
      "     Neutral       0.80      0.44      0.57        18\n",
      "    Positive       0.71      0.71      0.71        14\n",
      "\n",
      "    accuracy                           0.62        45\n",
      "   macro avg       0.66      0.64      0.62        45\n",
      "weighted avg       0.68      0.62      0.62        45\n",
      "\n",
      "\n",
      "Model and Vectorizer saved successfully!\n"
     ]
    }
   ],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.metrics import accuracy_score, classification_report\n",
    "import joblib\n",
    "\n",
    "# Step 3: Load the CSV data\n",
    "# Replace 'your_file.csv' with your actual file path\n",
    "data = pd.read_csv('Downloads/Degree/Sem 6/CSC575/Project/CSC575_sentiment_data2.csv')\n",
    "\n",
    "# Display the first few rows of the data\n",
    "print(data.shape)\n",
    "\n",
    "# Step 4: Preprocess the data\n",
    "# Checking for any missing values\n",
    "print(\"\\nMissing values in each column:\")\n",
    "print(data.isnull().sum())\n",
    "\n",
    "# Dropping any rows with missing values\n",
    "data.dropna(inplace=True)\n",
    "\n",
    "# Step 5: Split the data into features and labels\n",
    "X = data['Reviews']  # Text data\n",
    "y = data['Sentiment_Label']  # Labels (0 = negative, 1 = positive, etc.)\n",
    "\n",
    "# Step 6: Convert text data to numerical format using TF-IDF\n",
    "tfidf = TfidfVectorizer(max_features=5000, stop_words='english')\n",
    "X_transformed = tfidf.fit_transform(X)\n",
    "\n",
    "# Step 7: Split the data into training and testing sets\n",
    "X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.3, random_state=42)\n",
    "\n",
    "# Step 8: Train a Logistic Regression model\n",
    "model = LogisticRegression()\n",
    "model.fit(X_train, y_train)\n",
    "\n",
    "# Step 9: Make predictions and evaluate the model\n",
    "y_pred = model.predict(X_test)\n",
    "\n",
    "# Calculate accuracy\n",
    "accuracy = accuracy_score(y_test, y_pred)\n",
    "print(f\"\\nModel Accuracy: {accuracy:.2f}\")\n",
    "\n",
    "# Print classification report\n",
    "print(\"\\nClassification Report:\")\n",
    "print(classification_report(y_test, y_pred))\n",
    "\n",
    "# Step 10: Save the trained model and the TF-IDF vectorizer\n",
    "joblib.dump(model, 'Downloads/Degree/Sem 6/CSC575/Project/sentiment_model.pkl')\n",
    "joblib.dump(tfidf, 'Downloads/Degree/Sem 6/CSC575/Project/tfidf_vectorizer.pkl')\n",
    "\n",
    "print(\"\\nModel and Vectorizer saved successfully!\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
