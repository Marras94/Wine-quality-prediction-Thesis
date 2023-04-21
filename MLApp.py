from kivymd.app import MDApp
from kivy.lang.builder import Builder
from kivymd.uix.screen import Screen
from kivymd.uix.textfield import MDTextField
from kivy.uix.screenmanager import ScreenManager
import numpy as np
import pickle

# Load the saved red wine model and score
with open('Red_model.pkl', 'rb') as file:
    Red_model = pickle.load(file)

with open('Red_score.pkl', 'rb') as file:
    Red_score = pickle.load(file)

# Load the saved white wine model and score
with open('White_model.pkl', 'rb') as file:
    White_model = pickle.load(file)

with open('White_score.pkl', 'rb') as file:
    White_score = pickle.load(file)



# Window number 1.
class StartWindow(Screen):
    #model = None

    def get_wine_type(self):
        # Set default model to None
        self.model = None

        # Get the wine type from the start screen buttons
        start_screen = self.manager.get_screen("Start")

        if start_screen.ids.red_button.state == "down":
            self.model = Red_model
            return self.model

        elif start_screen.ids.white_button.state == "down":
            self.model = White_model
            return self.model

        else:
            print("Can't get the model...")


# Window number 2.
class AttributeWindow(Screen):

    def get_inputs(self):
        model = self.manager.get_screen("Start").model

        if model == Red_model:
            fixed_acidity = float(self.ids.text_field1.text)
            volatile_acidity = float(self.ids.text_field2.text)
            citric_acid = float(self.ids.text_field3.text)
            residual_sugar = float(self.ids.text_field5.text)
            chlorides = float(self.ids.text_field6.text)
            free_sulfur_dioxide = float(self.ids.text_field4.text)
            total_sulfur_dioxide = float(self.ids.text_field7.text)
            density = float(self.ids.text_field8.text)
            pH = float(self.ids.text_field9.text)
            sulphates = float(self.ids.text_field10.text)
            alcohol = float(self.ids.text_field11.text)
            total_acidity = float(fixed_acidity * volatile_acidity)
            alcohol_acidity = float(alcohol / volatile_acidity)
            chlorides_sulphates = float(chlorides / sulphates)

            inputs = [fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                    chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density,
                    pH, sulphates, alcohol, total_acidity, alcohol_acidity, chlorides_sulphates]
            return inputs

        if model == White_model:
            fixed_acidity = float(self.ids.text_field1.text)
            volatile_acidity = float(self.ids.text_field2.text)
            citric_acid = float(self.ids.text_field3.text)
            residual_sugar = float(self.ids.text_field5.text)
            chlorides = float(self.ids.text_field6.text)
            free_sulfur_dioxide = float(self.ids.text_field4.text)
            total_sulfur_dioxide = float(self.ids.text_field7.text)
            #density = float(self.ids.text_field8.text)
            pH = float(self.ids.text_field9.text)
            sulphates = float(self.ids.text_field10.text)
            #alcohol = float(self.ids.text_field11.text)
            inputs = [fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                    chlorides, free_sulfur_dioxide, total_sulfur_dioxide,
                    pH, sulphates]
            return inputs
        
    def make_prediction(self):
        inputs = self.get_inputs()
        model = self.manager.get_screen("Start").model
        
        if model is None:
            self.manager.get_screen("Predict").update_labels(0, "Please select a wine type first.")

        elif not hasattr(model, 'predict'):
            self.manager.get_screen("Predict").update_labels(0, "Model is not fitted yet. Please wait a moment and try again.")

        else:
            inputs = np.array(inputs)
            prediction = model.predict(inputs.reshape(1, -1))
            predicted_quality = prediction.tolist()[0]  

            # Display the accuracy score based on the selected model
            if model == Red_model:
                print('RED')
                accuracy = Red_score
                
            elif model == White_model:
                print('White')
                accuracy = White_score

            else:
                print("Can't find model")
                accuracy = "N/A"

            # Update the labels in the PredictWindow
            self.manager.get_screen("Predict").update_labels(accuracy, predicted_quality)

    def on_click_predict(self):
        self.make_prediction()

    def go_back(self):
        # Reset the text fields
        for i in range(1, 12):
            input_field = self.ids[f"text_field{i}"]
            input_field.text = ""

        # Deselect the wine type buttons
        start_screen = self.manager.get_screen("Start")
        start_screen.ids.red_button.state = "normal"
        start_screen.ids.white_button.state = "normal"




# Window 3
class PredictWindow(Screen):

    def update_labels(self, accuracy, prediction):
        self.accuracy_label.text = f"Accuracy: {accuracy:.0%}"
        #self.accuracy_label.text = f"Accuracy: {accuracy:.2f}%"
        self.prediction_label.text = f"Prediction: {prediction}"

    def on_enter(self, *args):
        pass

    def reset_inputs(self):
        for widget in self.walk(restrict=True):
            if isinstance(widget, MDTextField):
                widget.text = ""



# Create the screen manager
# The screen manager, to define the layout for the app.
class WindowManager(ScreenManager):
    pass


class WineQualityApp(MDApp):
    def build(self):
        self.theme_cls.theme_style = "Dark"
        self.theme_cls.primary_palette = "Orange"
        Prediction = Builder.load_file('predict.kv')
        # Create the screen manager
        sm = ScreenManager()
        
        # Add screens to the screen manager
        sm.add_widget(StartWindow(name='Start'))
        sm.add_widget(AttributeWindow(name='Attribute'))
        sm.add_widget(PredictWindow(name='Predict'))

        return Prediction

    
if __name__ == '__main__':
    WineQualityApp().run()

