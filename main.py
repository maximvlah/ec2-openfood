import argparse
import base64
import io
import os
import logging
import sys

import yaml
import boto3
import base64
import pickle
import tarfile
import urllib.request

import base64
from io import BytesIO

from annoy import AnnoyIndex

from tqdm import tqdm

import tensorflow as tf
import tensorflow_hub as hub

from urllib.parse import urlparse

from aiohttp.client import ClientSession
from asyncio import wait_for, gather, Semaphore

from typing import Optional, List

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from pydantic import BaseModel

import numpy as np

from PIL import Image

from mangum import Mangum


logger = logging.getLogger(__name__)

# Get AWS credentials
AWS_SERVER_PUBLIC_KEY = os.environ.get('AWS_SERVER_PUBLIC_KEY', None)
AWS_SERVER_ACCESS_KEY = os.environ.get('AWS_SERVER_ACCESS_KEY', None)

class HealthCheck(BaseModel):
    """
    Health check.
    """
    message: Optional[str] = 'OK'

class ImageInput(BaseModel):
    """
    Represents an image to be predicted.
    """
    url: Optional[str] = None
    data: Optional[str] = None


class ProductSearchRequest(BaseModel):
    """
    Represents a request to process
    """
    images: List[ImageInput] = []

class ProductSearchResponse(BaseModel):
    """
    Represents the result of a product search request
    """
    search_results: List = None

class ImageNotDownloadedException(Exception):
    pass

def download_files_from_s3(bucket_name,
                           s3_client,
                           files):
    """
    Downloads files from S3 bucket
    
    :param bucket_name (str): name of S3 bucket
    :param s3_client (boto3.client): S3 client
    :param files (list[str]): list of files to download
    """
    # Get list of files in S3 bucket
    for f in tqdm(files, desc="Downloading files from S3", total=len(files)):

        if not os.path.exists(f):
            s3_client.download_file(bucket_name, f, f)
        else:
            print(f"{f} already exists!")
    
def retrieve_data_from_aws(idxs,s3_client,table,partition_key):
    """
    Retrieves data for product from AWS
    
    :param idxs (list[int]): best match idxs
    :param s3_client (boto3.client): S3 client
    :param table (boto3.dynamodb.Table): DynamoDB table
    :param partition_key (str): partition key of DynamoDB table
    
    :return: data (list[dict]): data for best matches
    """

    search_results = []
    for idx in range(len(idxs)):

        # Get data from DynamoDB
        ean = IDX_TO_EAN_MAP[idxs[idx]]
        response = table.get_item(
            Key={
                partition_key: str(ean) #NOTE: converting to string to match DynamoDB partition key type
            }
        )
        best_matches_metadata = response['Item']

        # Load image from S3 bucket and convert to Base64
        #NOTE: using crops of original images instead of original images
        s3_image = s3_client.get_object(Bucket=CONFIG["bucket_name"], Key=f"crops/{ean}_.jpg") 
        s3_image = Image.open(io.BytesIO(s3_image['Body'].read())).convert('RGB')
        best_matches_metadata['b64'] = img2b64(s3_image)

        # Append to search_results
        search_results.append(best_matches_metadata)

    return search_results

def load_search_index(path,
               num_dimensions=2048,
               metric="angular"):
    """
    Loads the search index from disk

    :param path (str): path to the search index
    :param num_dimensions (int): number of dimensions of the embedding vector
    :param metric (str): similarity metric whith which the index was built

    :return: search_index (AnnoyIndex): Annoy Index

    """

    search_index = AnnoyIndex(num_dimensions,metric)
    search_index.load(path)

    print(f"Loaded search index from {path}")
    return search_index

def img2b64(image):
    """
    Converts PIL image to Base64 encoded string
    
    :param image (PIL.image)
    :return: b64 (str): Base64 encoded string
    """

    buff = BytesIO()
    image.save(buff, format="JPEG")
    img_str = base64.b64encode(buff.getvalue())
    return img_str.decode("utf-8")

def b642bytes(b64_string):
    """
    Converts Base64 string to bytes
    :param b64_string: string to convert
    :return: bytes
    """
    bytes = str.encode(b64_string)
    bytes = base64.b64decode(bytes)
    return bytes


class ImageEmbedder:
    """
    Calculates image embedding.
    """
    def __init__(self,
                 model_url:str,
                 model_save_dir:str,
                 input_width:int=224,
                 input_height:int=224,
                 input_channels:int=3):
        """
        Prepares the model used by the application for use.

        :param model_url
        :param input_width
        :param input_height
        :param input_channels
        """

        self.save_dir = model_save_dir
        
        self._download_model(model_url,self.save_dir)
        if "tfhub" in model_url:
            # Load tfhub module
            self.model = tf.keras.Sequential([hub.KerasLayer(self.save_dir,trainable=False)])
        else:
            # Load keras model
            self.model = tf.keras.models.load_model(self.save_dir)

        self.input_width = input_width
        self.input_height = input_height
        self.input_channels = input_channels
        self.model.build([None, self.input_width, self.input_height, self.input_channels])

        # Warm up
        self.model(tf.zeros((1, self.input_width, self.input_height, self.input_channels)))
        logger.info("Warmed up the model.")
        
    def _download_model(self,url:str,save_dir:str="models/latest"):
        """
        Downloads latest embedder model.
        :param url:string
        :param save_dir:string
        """

        os.makedirs(save_dir,exist_ok=True)

        if not os.path.exists(save_dir +"/saved_model.pb"):

            if "tfhub" in url:
                logger.info("Downloading latest embedder model..")
                archive_name = "model.tar.gz"
                urllib.request.urlretrieve(
                    url,
                    os.path.join(save_dir,archive_name)
                )

                with tarfile.open(os.path.join(save_dir,archive_name), "r:gz") as f:
                    f.extractall(save_dir)
                    
                os.remove(os.path.join(save_dir,archive_name))
            else:
                raise ValueError("Currently only tfhub models are supported!")
        else:
            logger.info("Model already exists. Skipping download.")

    def _preprocess_img(self,bytes):
        """
        Processes image bytes in format suitable for tf inference

        :param bytes (bytes)
        :return: img (np.array): preprocessed image
        """
        img = tf.io.decode_image(bytes,channels=self.input_channels)

        img = tf.image.resize(img, 
                              method="bilinear", 
                              size=(self.input_width,self.input_height))
        img = tf.keras.preprocessing.image.img_to_array(img)

        img = img / 255.0
        return img

    def _prepare_images(self, images):
        """
        Prepares the images for prediction.

        :param images: The list of images to prepare for prediction in bytes format.

        :return: A list of processed images.
        """
        batch = np.zeros((len(images), self.input_height, self.input_width, self.input_channels), dtype=np.float32)
        for i, image_bytes in enumerate(images):
            batch[i, :] = np.array(self._preprocess_img(image_bytes), dtype=np.float32)
        return batch

    def predict(self, images, batch_size:int):
        """
        Calculates embeddings.

        :param images: A list of images to use.
        :param batch_size: The number of images to process at once.

        :return: A tensor containing the embeddings for each image.
        """
        batch = self._prepare_images(images)
        embeddings = self.model.predict(batch, batch_size)
        return embeddings

# Load yaml config file
with open("config.yaml", "r") as stream:
    try:
        CONFIG = (yaml.safe_load(stream))
    except yaml.YAMLError as exc:
        print(exc)
        CONFIG = None
assert CONFIG is not None, "Could not load config file."

 # FastAPI app
app = FastAPI()

# Login into AWS
session = boto3.Session(
    aws_access_key_id=AWS_SERVER_PUBLIC_KEY,
    aws_secret_access_key=AWS_SERVER_ACCESS_KEY,
)

# Connect to DynamoDB
dynamodb = session.resource('dynamodb',region_name=CONFIG["region_name"])
table = dynamodb.Table(CONFIG["table_name"])

# Connect to S3
s3_client = session.client('s3',region_name=CONFIG["region_name"])

# Download files needed for vector search from S3
download_files_from_s3(CONFIG["bucket_name"],
                       s3_client,
                       CONFIG["s3_files"])

# Load mapping from idx to EAN.
with open("idx_to_ean_map.pickle", "rb") as f: 
    IDX_TO_EAN_MAP = pickle.load(f)

# Load Annoy Search Index
annoy = load_search_index("search_index.ann", 
                          CONFIG["embedding_size"], 
                          CONFIG["metric"])

@app.exception_handler(Exception)
async def unknown_exception_handler(request: Request, exc: Exception):
    """
    Catch-all for all other errors.
    """
    return JSONResponse(status_code=500, content={'message': 'Internal error.'})

@app.exception_handler(ImageNotDownloadedException)
async def client_exception_handler(request: Request, exc: ImageNotDownloadedException):
    """
    Called when the image could not be downloaded.
    """
    return JSONResponse(status_code=400, content={'message': 'One or more images could not be downloaded.'})

def configure_logging(logging_level=logging.INFO):
    """
    Configures logging for the application.
    """
    root = logging.getLogger()
    root.handlers.clear()
    stream_handler = logging.StreamHandler(stream=sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stream_handler.setFormatter(formatter)
    root.setLevel(logging_level)
    root.addHandler(stream_handler)

@app.on_event('startup')
def load_model():
    """
    Loads the model prior to the first request.
    """
    if not hasattr(app.state, 'model'):
        configure_logging()
        logger.info('Loading models...')
        app.state.model = ImageEmbedder(CONFIG["model_url"],
                                        CONFIG["model_save_dir"],
                                        CONFIG["input_width"],
                                        CONFIG["input_height"],
                                        CONFIG["input_channels"])

def get_url_scheme(url, default_scheme='unknown'):
    """
    Returns the scheme of the specified URL or 'unknown' if it could not be determined.
    """
    result = urlparse(url, scheme=default_scheme)
    return result.scheme


async def retrieve_content(entry, sess, sem):
    """
    Retrieves the image content for the specified entry.
    """
    image_bytes = None
    if entry.data is not None:
        image_bytes = b642bytes(entry.data)
        return image_bytes

    elif entry.url is not None:
        source_uri = entry.url
        scheme = get_url_scheme(source_uri)
        if scheme in ('http', 'https'):
            image_bytes = await download(source_uri, sess, sem)
            return image_bytes
        else:
            raise ValueError('Invalid scheme: %s' % scheme)
    return None


async def retrieve_images(entries):
    """
    Retrieves the images for processing.

    :param entries: The entries to process.

    :return: The retrieved data.
    """
    tasks = list()
    sem = Semaphore(CONFIG["thread_count"])
    async with ClientSession() as sess:
        for entry in entries:
            tasks.append(
                wait_for(
                    retrieve_content(entry, sess, sem),
                    timeout=CONFIG["timeout"],
                )
            )
        return await gather(*tasks)


async def download(url, sess, sem):
    """
    Downloads an image from the specified URL.

    :param url: The URL to download the image from.
    :param sess: The session to use to retrieve the data.
    :param sem: Used to limit concurrency.

    :return: The file's data.
    """
    async with sem, sess.get(url) as res:
        logger.info('Downloading %s' % url)
        content = await res.read()
        logger.info('Finished downloading %s' % url)
    if res.status != 200:
        raise ImageNotDownloadedException('Could not download image.')
    return content


def predict_images(images):
    """
    Predicts the image's category and transforms the results into the output format.

    :param images: The Pillow Images to predict.

    :return: The prediction results.
    """
    results = app.state.model.predict(images, CONFIG["batch_size"])
    return results

@app.post('/v1/predict', response_model=ProductSearchResponse)
async def process(req: ProductSearchRequest):
    """
    Calculates feature vectors of the images contained in the request.

    :param req: The request object containing the image data to predict.

    :return: Search results.
    """
    logger.info('Processing request...')
    logger.debug(req.json())
    logger.info(f'Downloading & processing {len(req.images)} images...')
    images = await retrieve_images(req.images)
    logger.info('Performing prediction...')
    predictions = predict_images(images)

    search_results = []
    for i in range(predictions.shape[0]):
        logger.info(f'Searching for {i}nth image...')
        best_matches_idxs = annoy.get_nns_by_vector(predictions[i],
                                                    CONFIG["k"])

        logger.info('Retrieving data from AWS...')
        search_results.append(retrieve_data_from_aws(best_matches_idxs,
                                                     s3_client,
                                                     table,
                                                     CONFIG["table_partition_key"]))
    logger.info('Transaction complete.')

    return ProductSearchResponse(search_results=search_results)

@app.get('/health')
def test():
    """
    Can be called by load balancers as a health check.
    """
    return HealthCheck()

# Need this in case AWS Lambda is used
handler = Mangum(app)

if __name__ == '__main__':

    import uvicorn

    parser = argparse.ArgumentParser(description='Runs the API locally.')
    parser.add_argument('--port',
                        help='The port to listen for requests on.',
                        type=int,
                        default=8080)
    args = parser.parse_args()
    configure_logging()
    uvicorn.run(app, host='0.0.0.0', port=args.port)