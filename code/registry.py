from code.params import *
from google.cloud import storage
from tensorflow import keras


def load_model():
    if MODEL_TARGET == "gcs":
        print(f'load latest model from GCS bucket {BUCKET_NAME}')

        client = storage.Client()
        bucket = client.get_bucket(BUCKET_NAME)
        print(f'*'*7)
        print(bucket)
        blob = bucket.get_blob(blob_name ="model_vect_SVC_BRD4_10k.pkl")
        print(f'#'*7)
        print(blob.name)

        try:
            latest_model_path_to_save = os.path.join(LOCAL_MODEL_PATH, blob.name)
            print(f'#'*7)
            print(latest_model_path_to_save)
            blob.download_to_filename(latest_model_path_to_save)
            print(f'#'*7)
            print("here")

            #model = keras.models.load_model(latest_model_path_to_save)
            print("✅ model downloaded from cloud storage")

        except:
            print(f"\n❌ No model found in GCS bucket {BUCKET_NAME}")

            return None
