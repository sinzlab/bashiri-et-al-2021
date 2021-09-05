import os
import datajoint as dj

dj.config["database.host"] = os.environ["DJ_HOST"]
dj.config["database.user"] = os.environ["DJ_USERNAME"]
dj.config["database.password"] = os.environ["DJ_PASSWORD"]
dj.config["enable_python_native_blobs"] = True

# set external store based on env vars
dj.config["stores"] = {
    "minio": {  # store in s3
        "protocol": "s3",
        "endpoint": os.environ.get("MINIO_ENDPOINT", "DUMMY_ENDPOINT"),
        "bucket": "nnfabrik",
        "location": "dj-store",
        "access_key": os.environ.get("MINIO_ACCESS_KEY", "FAKEKEY"),
        "secret_key": os.environ.get("MINIO_SECRET_KEY", "FAKEKEY"),
    }
}