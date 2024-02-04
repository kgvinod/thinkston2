import os
import threading
import time
import requests
import numpy as np
import openai
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model


# Set your OpenAI API key
openai.api_key = "KeY"  # Replace with your actual OpenAI API key

class Environment:
    def __init__(self, location):
        self.location = location
        self.weather = "sunny"
        self.noise_level = "normal"
        self.social_interaction = "normal"

    def set_weather(self, weather):
        self.weather = weather

    def get_weather(self):
        return self.weather

    def set_noise_level(self, noise_level):
        self.noise_level = noise_level

    def get_noise_level(self):
        return self.noise_level

    def set_social_interaction(self, social_interaction):
        self.social_interaction = social_interaction

    def get_social_interaction(self):
        return self.social_interaction

class InternalEnvironment:
    def __init__(self, aches=0, discomfort=0):
        self.aches = aches
        self.discomfort = discomfort

    def get_aches(self):
        return self.aches

    def set_aches(self, aches):
        self.aches = aches

    def get_discomfort(self):
        return self.discomfort

    def set_discomfort(self, discomfort):
        self.discomfort = discomfort

class Person:
    def __init__(self, name, gender, age, location, environment, model_file="person_model.h5"):
        self.name = name
        self.gender = gender
        self.age = age
        self.location = location
        self.mood = "neutral"
        self.environment = environment
        self.internal_environment = InternalEnvironment()
        self.model_file = model_file

        if os.path.exists(model_file):
            self.model = load_model(model_file)
        else:
            self.model = self.build_model()
            self.train_model(self.get_dataset()[0], self.get_dataset()[1])


        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self.run, daemon=True)
        self._thread.start()

    def get_dataset(self):
        # Sample features (X)
        X_train = np.array([
            [1, 1, 1, 0],   # sunny, quiet, positive social interaction, no discomfort
            [1, 0, -1, -2], # sunny, normal, negative social interaction, discomfort -2
            [-1, -1, 0, -1], # rainy, loud, normal social interaction, discomfort -1
            [0, 1, 1, -3],  # cloudy, quiet, positive social interaction, discomfort -3
            [-0.5, 0, -1, 0], # snowy, normal, negative social interaction, no discomfort
        ])

        # Sample targets (y)
        y_train_raw = np.array([
            0, # happy
            2, # sad
            1, # neutral
            1, # neutral
            2, # sad
        ])

        # Convert y_train to one-hot encoding
        y_train = np.zeros((y_train_raw.size, y_train_raw.max() + 1))
        y_train[np.arange(y_train_raw.size), y_train_raw] = 1
        return (X_train, y_train)


    def build_model(self):
        model = Sequential()
        model.add(Dense(16, input_dim=4, activation='relu'))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(3, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])
        return model

    def train_model(self, X_train, y_train, epochs=10):
        self.model.fit(X_train, y_train, epochs=epochs, verbose=0)
        self.save_model()

    def save_model(self):
        self.model.save(self.model_file)        

    def predict_mood(self, environment_data):
        prediction = self.model.predict(np.array([environment_data]))[0]
        mood_idx = np.argmax(prediction)
        mood_map = {0: "happy", 1: "neutral", 2: "sad"}
        return mood_map[mood_idx]

    def run(self):
        while not self._stop_event.is_set():
            weather = self.environment.get_weather()
            noise_level = self.environment.get_noise_level()
            social_interaction = self.environment.get_social_interaction()

            aches = self.internal_environment.get_aches()
            discomfort = self.internal_environment.get_discomfort()

            # Encode environment conditions as numerical values
            weather_map = {"sunny": 1, "rainy": -1, "cloudy": 0, "snowy": -0.5}
            noise_map = {"quiet": 1, "normal": 0, "loud": -1}
            social_map = {"positive": 1, "normal": 0, "negative": -1}

            environment_data = [weather_map[weather], noise_map[noise_level], social_map[social_interaction]]

            mood_factors = []
            if aches:
                mood_factors.append(-1 * aches)

            if discomfort:
                mood_factors.append(-1 * discomfort)

            mood_score = sum(mood_factors)
            environment_data.append(mood_score)

            print(environment_data)

            self.mood = self.predict_mood(environment_data)
            time.sleep(1)

    def stop(self):
        self._stop_event.set()
        self._thread.join()

    def get_mood(self):
        return self.mood

    def ask(self, question):
        if question.lower() == "how are you":
            return self.get_mood_response()
        else:
            return self.gpt3_response(question)

    def get_mood_response(self):
        mood = self.get_mood()
        response_map = {
            "happy": "I'm feeling great, thank you!",
            "neutral": "I'm feeling okay.",
            "sad": "I'm feeling a bit down today."
        }
        return response_map[mood]

    def gpt3_response(self, question):
        #system_prompt = f"You are {self.name}, {self.age}-year-old {self.gender} living in {self.location}. You are currently feeling {self.mood}."
        user_prompt = f"Assume you are {self.name}, {self.age}-year-old {self.gender} living in {self.location}. Give a sample response to a question '{question}'."
        
      
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                        #{"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                        ],
        )

        response_message = response.choices[0].message.content.strip()
        #print(response_message)
        return response_message

# Example usage
location = "Calicut, Kerala"
environment = Environment(location)
person = Person("Alice", "female", 28, location, environment)

# Update environment and internal environment
environment.set_weather("sunny")
environment.set_noise_level("quiet")
environment.set_social_interaction("positive")
person.internal_environment.set_aches(1)
person.internal_environment.set_discomfort(1)

time.sleep(2)  # Wait for the thread to update mood
print(person.get_mood())  # Output: "neutral"

question = "How are you?"
response = person.ask(question)
print(response)  # Output: "I'm feeling okay."

question = "Tell me a bit about yourself and how you spend your time in your home town."
response = person.ask(question)
print(response)  # Output: GPT-3 generated response

person.stop()  # Stop the thread when done

 
