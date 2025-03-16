import uuid

from dotenv import load_dotenv
from pymongo import MongoClient
from gridfs import GridFS
from bson.objectid import ObjectId
from io import BytesIO
from datetime import datetime
import os

from src.repo.model_schema import ModelSchema, ModelEntry, ModelTrainStatus

load_dotenv()


class ModelRepository:
    """
    Repository class to manage model entries in MongoDB with Pydantic validation.
    Uses GridFS for storing checkpoint.pth files.
    """
    def __init__(self, uri: str = os.getenv("MONGO_URI"), database_name: str = os.getenv("MONGO_DB")):
        """
        Initialize the repository with a MongoDB connection.

        Args:
            uri (str): MongoDB connection URI (e.g., "mongodb://localhost:27017/").
            database_name (str): Name of the database (e.g., "model_registry").
        """
        self.client = MongoClient(uri)
        self.db = self.client[database_name]
        self.models = self.db["models"]  # Collection for model schemas
        self.fs = GridFS(self.db)        # GridFS for checkpoint files

    def save(self, model_schema: ModelSchema, train_status: ModelTrainStatus ,checkpoint_file: str) -> str:
        """
        Save a model schema and its checkpoint file.

        Args:
            model_schema (ModelSchema): Validated model configuration.
            checkpoint_file (str): Path to the checkpoint.pth file.

        Returns:
            str: The unique ID of the saved model entry.

        Raises:
            FileNotFoundError: If the checkpoint file does not exist.
            ValueError: If schema validation fails (handled by Pydantic).
            :param checkpoint_file:
            :param model_schema:
            :param train_status:
        """
        if not os.path.exists(checkpoint_file):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file}")

        # Upload checkpoint to GridFS
        with open(checkpoint_file, 'rb') as f:
            checkpoint_id = self.fs.put(f, filename="checkpoint.pth")

        # Create ModelEntry instance
        now = datetime.utcnow()
        entry = ModelEntry(
            _id = uuid.uuid4().__str__(),
            model_schema = model_schema,
            train_status = train_status,
            checkpoint_id = str(checkpoint_id),
            created_at = now,
            updated_at = now
        )

        # Insert into MongoDB
        document = entry.dict(by_alias=True, exclude={"id"})  # Exclude 'id' as MongoDB generates it
        result = self.models.insert_one(document)
        return str(result.inserted_id)

    def load(self, model_id: str) -> tuple[ModelSchema, ModelTrainStatus ,BytesIO]:
        """
        Load a model schema and its checkpoint file by ID.

        Args:
            model_id (str): The unique ID of the model entry.

        Returns:
            tuple: (schema: ModelSchema, checkpoint_io: BytesIO) containing the schema and checkpoint data.

        Raises:
            ValueError: If the model ID is invalid or not found.
        """
        try:
            obj_id = ObjectId(model_id)
        except Exception:
            raise ValueError("Invalid model ID")

        document = self.models.find_one({"_id": obj_id})
        if document is None:
            raise ValueError("Model not found")

        # Convert to Pydantic model
        entry = ModelEntry.from_mongo(document)

        # Retrieve checkpoint from GridFS
        checkpoint_id = ObjectId(entry.checkpoint_id)
        checkpoint_data = self.fs.get(checkpoint_id).read()
        checkpoint_io = BytesIO(checkpoint_data)

        return entry.model_schema, entry.train_status ,checkpoint_io

    def list_all(self) -> list[ModelEntry]:
        """
        List all model entries.

        Returns:
            list: List of ModelEntry objects.
        """
        documents = self.models.find()
        return [ModelEntry.from_mongo(doc) for doc in documents]

    def get_one(self, model_id: str) -> ModelEntry:
        """
        Get a single model entry by ID (without checkpoint).

        Args:
            model_id (str): The unique ID of the model entry.

        Returns:
            ModelEntry: The model entry object.

        Raises:
            ValueError: If the model ID is invalid or not found.
        """
        try:
            obj_id = ObjectId(model_id)
        except Exception:
            raise ValueError("Invalid model ID")

        document = self.models.find_one({"_id": obj_id})
        if document is None:
            raise ValueError("Model not found")
        return ModelEntry.from_mongo(document)

    def delete(self, model_id: str):
        """
        Delete a model entry and its associated checkpoint file.

        Args:
            model_id (str): The unique ID of the model entry.

        Raises:
            ValueError: If the model ID is invalid or not found.
        """
        try:
            obj_id = ObjectId(model_id)
        except Exception:
            raise ValueError("Invalid model ID")

        document = self.models.find_one({"_id": obj_id})
        if document is None:
            raise ValueError("Model not found")

        # Delete checkpoint from GridFS and entry from collection
        checkpoint_id = ObjectId(document["checkpoint_id"])
        self.fs.delete(checkpoint_id)
        self.models.delete_one({"_id": obj_id})

    def close(self):
        """Close the MongoDB client connection."""
        self.client.close()