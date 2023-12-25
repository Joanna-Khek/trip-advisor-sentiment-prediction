# Trip Advisor Sentiment Prediction
![trip_advisor_logo](https://github.com/Joanna-Khek/trip-advisor-sentiment-prediction/assets/53141849/87b6f32c-ef61-495b-8e52-6010b673bfa2)

## Project Description
The objective of this project is to develop a robust sentiment analysis system capable of classifying customer reviews into positive or negative labels on the popular travel platform, TripAdvisor. This project is implemented using **PyTorch** and uses the state-of-the-art DistilBERT model, which is known for their efficiency and performance in natural language processing tasks. 

Source | Link 
--- | ---
Data | https://www.kaggle.com/datasets/andrewmvd/trip-advisor-hotel-reviews
Framework | PyTorch
Deployment | FastAPI, Docker

## Deployment
The model is deployed using FastAPI and containerised using Docker. The webpage leverages **Jinja** and **Bootstrap** components

![web_ss](https://github.com/Joanna-Khek/trip-advisor-sentiment-prediction/assets/53141849/cfbbc37b-6fc9-49b6-8699-7edbf41238fc)


## Getting Started
To get started with the project, follow these steps:            

1. Clone the repository to your local machine ``git clone https://github.com/Joanna-Khek/trip-advisor-sentiment-prediction``

2. Navigate to the Project Directory ``cd trip-advisor-sentiment-prediction``

3. Build the Docker Image ``docker build -t trip-advisor-sentiment . ``

4. Run the Docker Container ``docker run -p 8002:8002 trip-advisor-sentiment``

5. View the API page ``http://localhost:8002/``

## Usage

After running the Docker Container, users can obtain the predictions via the FastAPI UI or using the API endpoints.

![api_ss](https://github.com/Joanna-Khek/trip-advisor-sentiment-prediction/assets/53141849/a1d4efd9-defe-48b9-9744-405c03a6f7f5)








