import uuid
from pathlib import Path

from dotenv import load_dotenv
from pymongo import MongoClient
from gridfs import GridFS
from io import BytesIO
from datetime import datetime
import os
from bson.objectid import ObjectId
import logging
import tempfile
import hashlib
from tqdm import tqdm

from src.repo.model_schema import ModelSchema, ModelEntry, ModelTrainStatus

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Cloud MongoDB credentials
MONGO_CLOUD_PASS = os.getenv("MONGO_CLOUD_PASS", None)
MONGO_CLOUD_URI = os.getenv("MONGO_CLOUD_URI").replace("<db_password>", MONGO_CLOUD_PASS) if MONGO_CLOUD_PASS else os.getenv("MONGO_CLOUD_URI")
MONGO_CLOUD_DB = os.getenv("MONGO_CLOUD_DB", None)

# Local MongoDB credentials (default to localhost:27017)
MONGO_LOCAL_URI = os.getenv("MONGO_LOCAL_URI", "mongodb://localhost:27017/")
MONGO_LOCAL_DB = os.getenv("MONGO_LOCAL_DB", "local_model_db")


class ModelRepository:
    """
    Repository class to manage model entries with separate local (localhost:27017) and cloud MongoDB instances.
    Saves instantly to a temporary directory and local MongoDB, and queues cloud saves for processing on close().
    """

    def __init__(self, cloud_uri: str = MONGO_CLOUD_URI, cloud_db: str = MONGO_CLOUD_DB,
                 local_uri: str = MONGO_LOCAL_URI, local_db: str = MONGO_LOCAL_DB):
        # Local MongoDB setup
        try:
            self.local_client = MongoClient(local_uri)
            self.local_db = self.local_client[local_db]
            self.local_models = self.local_db["models"]
            self.local_fs = GridFS(self.local_db)
        except Exception as e:
            logger.error(f"Error connecting to local MongoDB: {str(e)}")
            raise e

        # Cloud MongoDB setup
        try:
            self.cloud_client = MongoClient(cloud_uri)
            self.cloud_db = self.cloud_client[cloud_db]
            self.cloud_models = self.cloud_db["models"]
            self.cloud_fs = GridFS(self.cloud_db)
        except Exception as e:
            logger.error(f"Error connecting to cloud MongoDB: {str(e)}")
            raise e

        temp_dir = Path("temp_checkpoints")
        temp_dir.mkdir(parents=True, exist_ok=True)
        self.local_dir = temp_dir.name
        logger.info(f"Temporary directory for local models: {self.local_dir}")

        # Queue for pending cloud saves
        self.cloud_save_queue = []

    def _compute_sha256(self, data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    def _get_local_path(self, model_id: str) -> str:
        return os.path.join(self.local_dir, f"temp_checkpoint_{model_id}.pth")

    def _save_to_cloud(self, model_id: str, checkpoint_data: bytes, checkpoint_hash: str,
                       model_schema: ModelSchema, train_status: ModelTrainStatus):
        """Save to cloud MongoDB with progress logging."""
        logger.info(f"Starting cloud save for model {model_id}")
        now = datetime.utcnow()
        entry = ModelEntry(
            _id=model_id,
            model_schema=model_schema,
            train_status=train_status,
            checkpoint_id=model_id,
            checkpoint_hash=checkpoint_hash,
            created_at=now,
            updated_at=now
        )
        logger.info(f"Cloud save progress for {model_id}: 33% (Prepared entry)")

        # Upload to GridFS (progress cannot be tracked for put, so we log the simulated progress)
        checkpoint_id = self.cloud_fs.put(checkpoint_data, filename="checkpoint.pth")
        entry.checkpoint_id = str(checkpoint_id)
        logger.info(f"Cloud save progress for {model_id}: 66% (Uploaded to GridFS)")

        document = self.cloud_models.find_one({"_id": model_id})
        if document:
            self.cloud_models.update_one({"_id": model_id}, {"$set": entry.dict(by_alias=True)})
        else:
            document = entry.dict(by_alias=True)
            document["_id"] = model_id
            self.cloud_models.insert_one(document)
        logger.info(f"Cloud save completed for {model_id}: 100% (MongoDB updated)")

    def save(self, model_id: str, model_schema: ModelSchema, train_status: ModelTrainStatus,
             checkpoint_file: str) -> str:
        """
        Save instantly to a temporary directory and local MongoDB, queue cloud save for processing on close().
        """
        if not os.path.exists(checkpoint_file):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file}")

        # Read checkpoint file with progress bar
        file_size = os.path.getsize(checkpoint_file)
        raw_data = bytearray()
        with open(checkpoint_file, 'rb') as f, tqdm(total=file_size, unit='B', unit_scale=True,
                                                     desc='Reading checkpoint file') as pbar:
            while True:
                chunk = f.read(4096)
                if not chunk:
                    break
                raw_data.extend(chunk)
                pbar.update(len(chunk))
        raw_data = bytes(raw_data)

        checkpoint_hash = self._compute_sha256(raw_data)
        checkpoint_data = raw_data

        # Save to temporary directory
        local_path = self._get_local_path(model_id)
        with open(local_path, 'wb') as f, tqdm(total=len(checkpoint_data), unit='B', unit_scale=True,
                                               desc='Saving model locally') as pbar:
            for i in range(0, len(checkpoint_data), 4096):
                chunk = checkpoint_data[i:i + 4096]
                f.write(chunk)
                pbar.update(len(chunk))
        logger.info(f"Saved model {model_id} to temporary directory at {local_path}")

        # Save to local MongoDB
        now = datetime.utcnow()
        entry = ModelEntry(
            _id=model_id,
            model_schema=model_schema,
            train_status=train_status,
            checkpoint_id=model_id,
            checkpoint_hash=checkpoint_hash,
            created_at=now,
            updated_at=now
        )
        document = self.local_models.find_one({"_id": model_id})
        if document:
            self.local_models.update_one({"_id": model_id}, {"$set": entry.dict(by_alias=True)})
        else:
            document = entry.dict(by_alias=True)
            document["_id"] = model_id
            self.local_models.insert_one(document)
        logger.info(f"Saved model {model_id} to local MongoDB")

        # Queue cloud save for later processing
        self.cloud_save_queue.append((model_id, checkpoint_data, checkpoint_hash, model_schema, train_status))
        logger.info(f"Queued cloud save for model {model_id} to be processed on close")

        return model_id

    def load(self, model_id: str) -> tuple[ModelSchema, ModelTrainStatus, BytesIO]:
        """
        Load a model, checking the local temporary file first. If it exists and passes integrity,
        return its content. Otherwise, load from local MongoDB (or cloud if needed), then cache locally.
        """
        local_path = self._get_local_path(model_id)

        # Attempt to load from the local temporary file
        if os.path.exists(local_path):
            logger.info(f"Loading model {model_id} from local file: {local_path}")
            file_size = os.path.getsize(local_path)
            raw_data = bytearray()
            with open(local_path, 'rb') as f, tqdm(total=file_size, unit='B', unit_scale=True,
                                                   desc='Loading local model') as pbar:
                while True:
                    chunk = f.read(4096)
                    if not chunk:
                        break
                    raw_data.extend(chunk)
                    pbar.update(len(chunk))
            raw_data = bytes(raw_data)
            document = self._get_document(self.local_models, model_id)
            entry = ModelEntry.from_mongo(document)
            if self._compute_sha256(raw_data) == entry.checkpoint_hash:
                logger.info(f"Local file integrity check passed for model {model_id}")
                return entry.model_schema, entry.train_status, BytesIO(raw_data)
            else:
                logger.warning(f"Local file integrity check failed for model {model_id}; loading from DB...")

        # Try loading from local MongoDB next
        try:
            document = self._get_document(self.local_models, model_id)
            entry = ModelEntry.from_mongo(document)
            checkpoint_id = ObjectId(entry.checkpoint_id)
            fs_file = self.local_fs.get(checkpoint_id)
            file_size = fs_file.length
            raw_data = bytearray()
            with tqdm(total=file_size, unit='B', unit_scale=True,
                      desc='Downloading model from local MongoDB') as pbar:
                while True:
                    chunk = fs_file.read(4096)
                    if not chunk:
                        break
                    raw_data.extend(chunk)
                    pbar.update(len(chunk))
            raw_data = bytes(raw_data)
            if self._compute_sha256(raw_data) == entry.checkpoint_hash:
                # Cache the file locally with progress
                with open(local_path, 'wb') as f, tqdm(total=len(raw_data), unit='B', unit_scale=True,
                                                       desc='Caching model locally') as pbar:
                    for i in range(0, len(raw_data), 4096):
                        chunk = raw_data[i:i + 4096]
                        f.write(chunk)
                        pbar.update(len(chunk))
                logger.info(f"Cached model {model_id} locally from local MongoDB at {local_path}")
                return entry.model_schema, entry.train_status, BytesIO(raw_data)
            else:
                logger.warning(f"Local MongoDB model integrity check failed for model {model_id}")
        except Exception as e:
            logger.info(f"Model {model_id} not found in local MongoDB: {str(e)}")

        # Finally, load from cloud MongoDB
        logger.info(f"Loading model {model_id} from cloud MongoDB")
        document = self._get_document(self.cloud_models, model_id)
        entry = ModelEntry.from_mongo(document)
        checkpoint_id = ObjectId(entry.checkpoint_id)
        fs_file = self.cloud_fs.get(checkpoint_id)
        file_size = fs_file.length
        raw_data = bytearray()
        with tqdm(total=file_size, unit='B', unit_scale=True,
                  desc='Downloading model from cloud MongoDB') as pbar:
            while True:
                chunk = fs_file.read(4096)
                if not chunk:
                    break
                raw_data.extend(chunk)
                pbar.update(len(chunk))
        raw_data = bytes(raw_data)
        if self._compute_sha256(raw_data) != entry.checkpoint_hash:
            raise ValueError(f"Integrity check failed for checkpoint of model ID: {model_id}")

        self.save(model_id, entry.model_schema, entry.train_status, local_path)

        # Cache the downloaded file locally using progress bars
        with open(local_path, 'wb') as f, tqdm(total=len(raw_data), unit='B', unit_scale=True,
                                               desc='Caching model locally') as pbar:
            for i in range(0, len(raw_data), 4096):
                chunk = raw_data[i:i + 4096]
                f.write(chunk)
                pbar.update(len(chunk))

        logger.info(f"Cached model {model_id} locally from cloud MongoDB at {local_path}")

        # Return the loaded checkpoint from cloud (now saved locally)
        return entry.model_schema, entry.train_status, BytesIO(raw_data)

    def _get_document(self, collection, model_id: str) -> dict:
        document = None
        try:
            if ObjectId.is_valid(model_id):
                document = collection.find_one({"_id": ObjectId(model_id)})
            if not document:
                document = collection.find_one({"_id": model_id})
            if not document:
                raise ValueError("Model not found with ID: " + model_id)
        except Exception as e:
            raise ValueError(f"Invalid model ID or database error: {str(e)}")
        return document

    def list_all(self) -> list[ModelEntry]:
        documents = self.local_models.find()
        return [ModelEntry.from_mongo(doc) for doc in documents]

    def get_one(self, model_id: str) -> ModelEntry:
        document = self._get_document(self.local_models, model_id)
        return ModelEntry.from_mongo(document)

    def delete(self, model_id: str):
        try:
            document = self._get_document(self.local_models, model_id)
            checkpoint_id = ObjectId(document["checkpoint_id"])
            self.local_fs.delete(checkpoint_id)
            self.local_models.delete_one({"_id": model_id})
        except ValueError:
            logger.info(f"Model {model_id} not found in local MongoDB, skipping deletion")

        try:
            document = self._get_document(self.cloud_models, model_id)
            checkpoint_id = ObjectId(document["checkpoint_id"])
            self.cloud_fs.delete(checkpoint_id)
            self.cloud_models.delete_one({"_id": model_id})
        except ValueError:
            logger.info(f"Model {model_id} not found in cloud MongoDB, skipping deletion")

        local_path = self._get_local_path(model_id)
        if os.path.exists(local_path):
            os.remove(local_path)
            logger.info(f"Deleted temporary copy of model {model_id} at {local_path}")

    def close(self):
        """
        Process any queued cloud saves and, additionally, update cloud storage with the current local version for each model.
        Then close database connections.
        """
        logger.info(f"Closing repository, processing {len(self.cloud_save_queue)} queued cloud saves")
        # Process any explicitly queued cloud saves
        for model_id, checkpoint_data, checkpoint_hash, model_schema, train_status in self.cloud_save_queue:
            try:
                self._save_to_cloud(model_id, checkpoint_data, checkpoint_hash, model_schema, train_status)
            except Exception as e:
                logger.error(f"Failed to save model {model_id} to cloud: {str(e)}")
        self.cloud_save_queue.clear()

        # For each model stored in local MongoDB, update cloud storage with the current local version.
        # (Assuming you want to update every model. Adjust this as needed.)
        for document in self.local_models.find():
            model_id = document["_id"]
            local_path = self._get_local_path(model_id)
            if os.path.exists(local_path):
                with open(local_path, 'rb') as f:
                    checkpoint_data = f.read()
                checkpoint_hash = self._compute_sha256(checkpoint_data)
                entry = ModelEntry.from_mongo(document)
                try:
                    self._save_to_cloud(model_id, checkpoint_data, checkpoint_hash, entry.model_schema,
                                        entry.train_status)
                    logger.info(f"Updated cloud version for model {model_id} from local file {local_path}")
                except Exception as e:
                    logger.error(f"Failed to update cloud for model {model_id}: {str(e)}")

        self.local_client.close()
        self.cloud_client.close()
        logger.info("Closed local and cloud MongoDB connections")
