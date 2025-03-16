import uuid
from dotenv import load_dotenv
from pymongo import MongoClient
from gridfs import GridFS
from io import BytesIO
from datetime import datetime
import os
from bson.objectid import ObjectId

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
        self.fs = GridFS(self.db)  # GridFS for checkpoint files

    def save(self, model_id: str ,model_schema: ModelSchema, train_status: ModelTrainStatus, checkpoint_file: str) -> str:
        """
        Save a model schema and its checkpoint file.

        Args:
            model_schema (ModelSchema): Validated model configuration.
            train_status (ModelTrainStatus): Model training status.
            checkpoint_file (str): Path to the checkpoint.pth file.
            model_id (str): Unique identifier for the model.

        Returns:
            str: The unique ID of the saved model entry.

        Raises:
            FileNotFoundError: If the checkpoint file does not exist.
            ValueError: If schema validation fails (handled by Pydantic).
        """
        if not os.path.exists(checkpoint_file):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file}")

        # Upload checkpoint to GridFS
        with open(checkpoint_file, 'rb') as f:
            checkpoint_id = self.fs.put(f, filename="checkpoint.pth")

        # Create ModelEntry instance
        now = datetime.utcnow()
        entry = ModelEntry(
            _id=model_id,
            model_schema=model_schema,
            train_status=train_status,
            checkpoint_id=str(checkpoint_id),
            created_at=now,
            updated_at=now
        )

        # Insert into MongoDB
        document = self.models.find_one({"_id": model_id})
        if document:
            # Update existing document
            result = self.models.update_one(
                {"_id": model_id},
                {"$set": entry.dict(by_alias=True)}
            )
            return str(result.upserted_id)
        else:
            # Insert new document
            document = entry.dict(by_alias=True)
            document["_id"] = model_id
            result = self.models.insert_one(document)
            return str(result.inserted_id)

    def load(self, model_id: str) -> tuple[ModelSchema, ModelTrainStatus, BytesIO]:
        """
        Load a model schema and its checkpoint file by ID.

        Args:
            model_id (str): The unique ID of the model entry.

        Returns:
            tuple: (schema: ModelSchema, train_status: ModelTrainStatus, checkpoint_io: BytesIO)
                   containing the schema, training status, and checkpoint data.

        Raises:
            ValueError: If the model ID is invalid or not found.
        """
        # Check if model_id is a valid ObjectId or a UUID string
        document = None
        try:
            # First try as ObjectId
            if ObjectId.is_valid(model_id):
                document = self.models.find_one({"_id": ObjectId(model_id)})

            # If not found, try as string (UUID)
            if not document:
                document = self.models.find_one({"_id": model_id})

            if not document:
                raise ValueError("Model not found with ID: " + model_id)

        except Exception as e:
            raise ValueError(f"Invalid model ID or database error: {str(e)}")

        # Convert to Pydantic model
        entry = ModelEntry.from_mongo(document)

        # Retrieve checkpoint from GridFS
        checkpoint_id = ObjectId(entry.checkpoint_id)
        checkpoint_data = self.fs.get(checkpoint_id).read()
        checkpoint_io = BytesIO(checkpoint_data)

        return entry.model_schema, entry.train_status, checkpoint_io

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
        document = None
        try:
            # First try as ObjectId
            if ObjectId.is_valid(model_id):
                document = self.models.find_one({"_id": ObjectId(model_id)})

            # If not found, try as string (UUID)
            if not document:
                document = self.models.find_one({"_id": model_id})

            if not document:
                raise ValueError("Model not found with ID: " + model_id)

        except Exception as e:
            raise ValueError(f"Invalid model ID or database error: {str(e)}")

        return ModelEntry.from_mongo(document)

    def delete(self, model_id: str):
        """
        Delete a model entry and its associated checkpoint file.

        Args:
            model_id (str): The unique ID of the model entry.

        Raises:
            ValueError: If the model ID is invalid or not found.
        """
        document = None
        try:
            # First try as ObjectId
            if ObjectId.is_valid(model_id):
                document = self.models.find_one({"_id": ObjectId(model_id)})

            # If not found, try as string (UUID)
            if not document:
                document = self.models.find_one({"_id": model_id})

            if not document:
                raise ValueError("Model not found with ID: " + model_id)

        except Exception as e:
            raise ValueError(f"Invalid model ID or database error: {str(e)}")

        # Delete checkpoint from GridFS and entry from collection
        checkpoint_id = ObjectId(document["checkpoint_id"])
        self.fs.delete(checkpoint_id)

        # Delete using the appropriate ID type
        if ObjectId.is_valid(model_id):
            self.models.delete_one({"_id": ObjectId(model_id)})
        else:
            self.models.delete_one({"_id": model_id})

    def close(self):
        """Close the MongoDB client connection."""
        self.client.close()