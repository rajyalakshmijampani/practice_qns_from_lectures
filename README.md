Steps to execute:

1. Set up a virtual environment and install the requirements
2. run process_training_data.py. It will create and store FAISS indices of training data into a folder named "FAISS". This is a one-time action.
3. run mock.py. It will generate 3 mock questions. Week number is hardcoded in the code. Can update it. Allowed range is [1,4], since the training data has only 4 weeks of content
