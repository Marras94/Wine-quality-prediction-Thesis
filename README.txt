User guide

Overview
The primary objective of this project is to analyze wine datasets to predict the quality for future
wines. The scope of this project is to analyze a dataset of wine characteristics and quality ratings using
machine learning algorithms, specifically a random forest classifier. The project aims to identify which
features have the greatest impact on wine quality and to develop a model that can accurately predict
wine quality based on those features.

	Problem statement
	Develop a predictive model that utilizes various wine attributes to accurately predict the quality of the
	Portuguese "Vinho Verde", both red and white, and use a random forest model to provide actionable
	insights to wine producers to improve the quality of their wines.


List of content
- Datasets:
	* winequality-red.csv
	* winequality-white.csv
- Machine learning models:
	* RF_redwine.py
	* RF_whitewine.py
- Main code: MLApp.py
- KivyMD: predict.kv


Installation instructions
All files in this project are needed to make the code work properly. Download the folder and if you change any names of the files, this must be changed in the code. 
Python 3.10 and VSCode were used for the project. If another type of Python is used, this must be taken into account in relation to different libraries. 
Other code editors can be used.


Configuration instructions
All the needed libraries are imported/listed in the code. But here is an overview. Make sure that all libraries are  
- pandas
- numpy
- seaborn
- matplotlib
- pickle
- MinMaxScaler
- SMOTE
- GridSearchCV
- RandomForestClassifier
- train_test_split, cross_val_score
- accuracy_score, classification_report, confusion_matrix

For using Kivy:
- MDApp, Builder, Screen, MDTextField, ScreenManager


Usage instructions
First step: Make sure that all files in the dataset is in the same folder and open them from the folder.
Second step: Run the files RF_redwine.py and RF_whitewine.py
	- Trains the Red_model and White_model, gets the performance score for both models. 
Third step:  Run the MLApp.py
	- Opens the Red and White model by using pickel. Runs the app and gives you the opportunity to test with new wine attributes.


Troubleshooting
- Missing libraries/ wrong libraries
	- Can occur if you are using a different version of python. Changes the version or addapt the libraries according to your version.

- Error that indicates that names/files don't excist
	- Double check that every file is in the same folder   

- "This RandomForestClassifier instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator."
	- Issues with the librarie Pickle

- Don't hesitate to get in touch if there are other errors that you find difficult to solve. 
	I have been through every possible error with this project.


Contact information
If having any questions or issues with the code, please contact me at:
linkedin.com/in/marthe-molde-rasmussen-3a3989240
