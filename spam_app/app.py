from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)


df = pd.read_csv('messages.csv')
df['subject'] = df['subject'].fillna('')
df['message'] = df['message'].fillna('')
# Combine subject + full message body for richer features
df['full_email'] = df['subject'] + ' ' + df['message']

X_train, _, y_train, _ = train_test_split(
    df['full_email'], df['label'], test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
model = MultinomialNB()
model.fit(vectorizer.fit_transform(X_train), y_train)

print("✅ Model trained and ready!")


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    subject = data.get('subject', '')
    message = data.get('message', '')
    full_email = subject + ' ' + message

    if not full_email.strip():
        return jsonify({'error': 'Empty input'}), 400

    vec = vectorizer.transform([full_email])
    prediction = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0]
    confidence = round(float(prob[prediction]) * 100, 1)

    return jsonify({
        'label': 'SPAM' if prediction == 1 else 'HAM',
        'confidence': confidence
    })


if __name__ == '__main__':
    app.run(debug=True)
