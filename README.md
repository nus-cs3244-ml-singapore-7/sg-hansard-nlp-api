# sg-hansard-nlp-api

## About
API for the Singapore Hansard NLP Model to perform NER and Sentiment Analysis. The model can be downloaded from the [releases](https://github.com/nus-cs3244-ml-singapore-7/singapore-hansard-nlp/releases/tag/sentiment-ner) page. 
The backend was built using [FastAPI](https://fastapi.tiangolo.com/) and deployed with [Docker](https://www.docker.com/) on Google Cloud Platform cloud run.  

## Set up Locally
1. Clone Repo `git clone https://github.com/nus-cs3244-ml-singapore-7/sg-hansard-nlp-api.git`
2. Change directory `cd sg-hansard-nlp-api`
3. Download the models from [releases](https://github.com/nus-cs3244-ml-singapore-7/singapore-hansard-nlp/releases/tag/sentiment-ner) and add it to the `finetuned_models` directory
4. Install virtual environment `virtualenv env`
5. Activate environment  `env\Scripts\activate`
6. Install requirements `pip install -r requirements.txt`
7. Install pytorch cpu `pip install torch==1.7.1+cpu -f https://download.pytorch.org/whl/torch_stable.html`
8. Start the server `uvicorn main:app --host 0.0.0.0 --port 8080`

## Deploy to Google Cloud Platform (GCP)
1. Create docker image `docker build -t <image_name>:<tag_name> .`
1. Login using [gcloud CLI](https://cloud.google.com/sdk/docs/install) `gcloud auth login`
2. Tag the image in the correct format for deployment `docker tag <image_name>:<tag_name> gcr.io/<project_name>/<image_name>:<tag_name>`
3. Push to GCP container registry `docker push gcr.io/<project_name>/<image_name>:<tag_name>`
4. Go to GCP container registry and deploy using cloud run

