import os

FLASK_BIND_HOST = '0.0.0.0'
FLASK_BIND_PORT = 5000
FLASK_DEBUG = False

#PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
PROJECT_ROOT = '/home/miracle/flask/myflask'
STATIC_ROOT = os.path.join(PROJECT_ROOT, '/static')
MATCHING_MODEL_PATH = os.path.join(PROJECT_ROOT, 'data/knn6.2016.09.24.pkl')
MATCHING_MODEL_DRESS_DETAILS_PATH = os.path.join(PROJECT_ROOT, 'data/dress_details.csv')

STATIC_LABELS = [
    "background",
    "mask",
    ]

AWS_PROFILE = 'findmydress-web'
DB_CONNECTION_STRING = 'postgresql+psycopg2://findmydress:Ooheum7hai2v@postgres.dressme.internal/findmydress'
IMAGES_S3_BUCKET = 'findmydress'
