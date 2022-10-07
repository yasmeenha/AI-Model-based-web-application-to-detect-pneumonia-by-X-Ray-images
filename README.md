# xray
softwre development in AI project
Current Progress:
                         We are currently on Milestone 3 and our project is progressing as per our initial plans. We have finished coding for the back-end deployment of the project.We have deployed a deep learning model using tensorflow and keras and our pretrained model  can predict whether a given x-ray image is a normal x-ray or whether the given x-ray image is infected with pneumonia.Now we have to work on the front end deployment of our project.We  have to make a website for our project on aws cloud where the user can upload their x-ray image and the model can predict whether the x-ray of the user is a normal one or infected with pneumonia. For the website deployment we are planning to use flask and Heroku.
 Meetings: 
Based on our availability Wednesday or Tuesday
This week we will try and finish  the website deployment part of our project.



Executive summary(Motivation):

          Our goal of the project is to empower radiologists with a Convolutional Neural Networks based AI model to predict the severity of pneumonia based on the X-ray images. As pneumonia is a deadly disease and the key to saving the lives of people is by detecting pneumonia in the early stages. In the past 2 years with the rise in Covid pandemic too, there was a rise in pneumonia affecting the lungs and there was a shortage of experienced radiologists. In such events an AI model with good accuracy can assist  junior doctors in predicting pneumonia from x-ray images in the initial stages of the disease and thus saving lives. [Reference 9,10]


Abstract:

We are using three layered Convolutional neural networks as a baseline to train our model to detect between a good x-ray image and an x-ray image that is infected with pneumonia. The program uploads the image and sends it to the back-end TensorFlow model. The model then gave the score in percentage  of both the categories and based on it we figured out whether it is a healthy x-ray image or a pneumonia infected x-ray image. 

          We used supervised learning on our data. We used 80 percent  of our x-ray images as the training data to train our model to distinguish between healthy lung x-ray and the x-ray with infected pneumonia. We checked the accuracy of how well the model is trained based on supervised learning, and we have a 98 percent accurate model.. [1,2]

We will create a web-based application and host the trained model in the backend to communicate and process the image. We used the  dataset provided in  https://data.mendeley.com/datasets/rscbjbr9sj/2  to  train the model. The tensor flow library was  used to train the model and flask and Heroku will be used for web-based deployment. The cloud service that will be used is aws.




AI based web application to detect pneumonia by X Ray images
 

Workflow:

The flow of the project is as follows
1.	The user will upload a picture of chest X Ray to be examined on the front end of the website.
2.	The program uploads the image and sends it to backend tensor flow model.
3.	The tensor flow model gives the score and based on the score we are predicting if the x-ray image is a normal one or if the x-ray image is infected with pneumonia.
4.	The user receives a message on the screen if the x-ray is normal or infected with pneumonia  along with the accuracy score.

 
Dataset:
 

We are using the  dataset provided in  https://data.mendeley.com/datasets/rscbjbr9sj/2 to  train the model.
 The X-ray Images are split into a training set and a testing set of independent patients. X-ray Images are labeled as (disease)-(randomized patient ID)-(image number by this patient) and split into 2categories: NORMAL and pneumonia.
The image on the left hand side shows  a normal x-ray and the two images on the right hand side show that the x-ray image is infected with pneumonia.The white opaqueness in the x-ray image is the indication of pneumonia.

The dataset is organized into 3 folders (train, test, val) and contains subfolders for each image category (Pneumonia/Normal). There are 5,863 X-Ray images (JPEG) and 2 categories (Pneumonia/Normal).
Chest X-ray images (anterior-posterior) were selected from retrospective cohorts of pediatric patients of one to five years old from Guangzhou Women and Children’s Medical Center, Guangzhou. All chest X-ray imaging was performed as part of patients’ routine clinical care.
For the analysis of chest x-ray images, all chest radiographs were initially screened for quality control by removing all low quality or unreadable scans. The diagnoses for the images were then graded by two expert physicians before being cleared for training the AI system. In order to account for any grading errors, the evaluation set was also checked by a third expert.


Reference:https://data.mendeley.com/datasets/rscbjbr9sj/2

List of technologies used for our project design:
1.	Data from Kaggle: https://data.mendeley.com/datasets/rscbjbr9sj/2
2.	Cloud tools: we are using AWS cloud as a cloud platform,Aws Beanstalk
3.	Tensor flow 
4.	Flask
5.	Heroku
Exploratory and Extensible:
1.	Base neural network and dataset EDA
2.	Neural network training and model generation
3.	Creation of serving model and basic web application
4.	Web application creation and integration with serving
5.	Testing and project report presentation

Tensor Flow Library for training the model and Keras for fine tuning the model.
For our project we are using Deep Learning which is a branch of Machine Learning.The computer simulations called  neural networks are used in deep learning to learn the different patterns in the x-ray image data being used by us and it doesn’t need any explicit programming to train the model.In the course of our project we are classifying the x-ray image data into 2 categories- normal and pneumonia.To train the tensor flow model we then divided our data into 3 parts as testing data,training data and validation data.After the model has been trained , we checked the accuracy.And, by further  fine tuning our model by using the keras tuner we tried to achieve the best possible accuracy  around 98.5%.
 

Final Milestone:
The final milestone of the project is to deploy the website on the cloud.The  technologies that will be used for the  front end deployment of the model are flask ,Heroku, aws cloud and aws Beanstalk.

References:

1.tensorflow.org/tutorials/keras/keras_tuner
the tensor flow library helps in selecting the hyperparameters that govern the training process of an ML model, https://keras.io/guides/
2.the dataset: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
3. https://docs.aws.amazon.com/elasticbeanstalk/latest/dg/create-deploy-python-apps.html, aws beanstalk web deployment 
4.Flask and Heroku for website deployment, https://docs.google.com/document/d/1C0bXpXYdD-yVUzu-8crsu5uZGDxkJs92/edit
5.docs.python.org/3/tutorial/   for the python tutorials
6.scikit.learn.org(to understand different ML models)
7.deep learning tutorials  geekforgeeks.org/machine-learning/
8. aws cloud tutorials-www.tutorialspoint.com/amazon_web_services/amazon_web_services_elastic_compute_cloud.htm
9. https://www.brookings.edu/techstream/building-better-diagnostic-standards-for-medical-ai/
10. https://med.stanford.edu/news/all-news/2017/11/algorithm-can-diagnose-pneumonia-better-than-radiologists.html






