import tensorflow as tf
from tensorflow import keras
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pickle

class EmotionClassifier:
    def __init__(self, model_path, tokenizer, max_length):
        self.model = keras.models.load_model(model_path)  
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.analyzer = SentimentIntensityAnalyzer() 

    def preprocess(self, text):
        sequences = self.tokenizer.texts_to_sequences([text])
        padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=self.max_length, padding='post')
        return padded_sequences

    def sentiment_classification(self, text):
        sentiment_mapping = {
        0: "anxiety",
        1: "bipolar",
        2: "Depression",
        3: "None",
        4: "Personality_Disorder",
        5: "Stress",
        6: "Suicidal"
        }

        sentiment_scores = self.analyzer.polarity_scores(text)
        sentiment = (
            'positive' if sentiment_scores['compound'] >= 0.05 
            else 'negative' if sentiment_scores['compound'] <= -0.05 
            else 'neutral'
        )

        padded_text = self.preprocess(text)
        predictions = self.model.predict(padded_text)
        prediction_dict = {}
        for prediction in predictions:
            current_prediction_dict = {}
            for i, prob in enumerate(prediction):
                emotion_name = sentiment_mapping[i]
                current_prediction_dict[emotion_name] = prob  

            prediction_dict = current_prediction_dict
        #predicted_class = tf.argmax(predictions, axis=1).numpy()

        return sentiment, prediction_dict
    
    def emotion_classification(self, text):
        
        padded_text = self.preprocess(text)
        emotion_mapping = {
            0: "anger",
            1: "fear",
            2: "joy",
            3: "love",
            4: "sadness",
            5: "surprised"
        }
        predictions = self.model.predict(padded_text)
        predicted_class = tf.argmax(predictions, axis=1).numpy()
        return emotion_mapping[predicted_class[0]]

if __name__ == "__main__":

    with open('E:\\Mental Health Chatbot\\Mental-Health-Chatbot\\tokenizer.pkl', 'rb') as handle:
        tokenizer_sentiment = pickle.load(handle)

    with open('E:\\Mental Health Chatbot\\Mental-Health-Chatbot\\tokenizer_five.pkl', 'rb') as handle:
        tokenizer_emotion = pickle.load(handle)    

    max_length = 40
    model_path_sentiment = 'E:\Mental Health Chatbot\Mental-Health-Chatbot\my_model.h5' 
    model_path_emotion = 'E:\Mental Health Chatbot\Mental-Health-Chatbot\my_model_five.h5' 

    classifier = EmotionClassifier(model_path_sentiment, tokenizer_sentiment, max_length)
    classifer_five = EmotionClassifier(model_path_emotion, tokenizer_emotion, max_length)

    custom_text = input("How do you feel today?: ")
    sentiment, prediction_dict = classifier.sentiment_classification(custom_text)
    predicted_class_emotion = classifer_five.emotion_classification(custom_text)


    print(f"Sentiment: {sentiment}")
    #for idx, prob in enumerate(probabilities):
        #print(f"Class {idx}: {prob:.4f}")
    print(prediction_dict)

    print(f"Predicted emotion: {predicted_class_emotion}")
