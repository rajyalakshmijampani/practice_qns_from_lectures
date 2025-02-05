Steps to execute:

1. Set up a virtual environment and install the requirements
2. Execute process_training_data.py. It will create and store FAISS indices of training data into a folder named "FAISS". This is a one-time action.
3. Execute mock.py. It will generate 3 mock questions.

Week number and number of questions to be generated are hardcoded in the code. Can be updated 
Training data has only 4 weeks of content. So allowed range for week number is [1,4].
