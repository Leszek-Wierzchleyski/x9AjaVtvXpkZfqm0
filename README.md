Introduction

We have a dataset consisting of the current job titles of potential candidates for an HR job in plain English, along with their number of LinkedIn connections, and their current location. We want to train a ranking algorithm to allow a recruiter to find the top N applicants automatically, for further inspection. A major limitation of this model is that it will only be trained using the current job titles of the potential candidates and does not include qualifications, years of experience, etc. This means that the usefulness of the ranking algorithm is limited.


Tokenising and Embedding

First we need to tokenise and embed the job titles, such that they can be fed into a model. We can do this using Hugging Face’s BERT model. We tokenise and embed the list of job titles as well as the phrase “Aspiring human resources - seeking human resources” as the target phrase. We use padding to ensure that the embedding vectors are all the same length. After this we compute the means of the embedding vectors and compare the mean vectors for each job title to that of the target phrase (calculating the cosine similarity). The mean vectors are the input features for our ranking model.

Ranking and Label Creation

Once we have the similarity scores we can append these to the data frame and sort the data frame by similarity score in descending order. We then append a column of numbers (ranks) to the sorted data frame and these numbers serve as labels for the ranking algorithm.

Principal Component Analysis

The mean vectors are 1024 elements long and we have 104 samples. This means that our features massively out number our samples and as such must be reduced. To do this we implement Principal Component Analysis and reduce the vectors to length 10. We implement this using SciKit-Learn Decomposition. Once we have the 10 values for each sample, we can put them into 10 separate columns in a data frame to create the input for the ranking model.

Training the Ranking Algorithm 

We implement a Lambda Rank algorithm using LightGBM. LightGBM is a good choice as it is quick, memory efficient, and accurate. The technical difference between LightGBM and XGBoost, is that LighGBM builds trees leaf-wise whereas XGBoost builds trees level-wise. We use a standard train-test split of 80:20. Once the model has been trained on the training data, we use it to predict fit scores on the test set. These scores can then be sorted and compared against the labels to study the performance of the model. 

Conclusion

This model has some limited use in ranking applicants based on their current job titles. As previously stated, the data upon which the model is trained is limited in scope and does not include most of the features which recruiters would want to look at when considering potential candidates (qualifications, years of experience, projects etc). Nonetheless the model provides some utility as a first pass filter for an application process.
