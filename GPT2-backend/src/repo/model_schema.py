from pydantic import BaseModel, Field
from datetime import datetime
from bson.objectid import ObjectId

class ModelSchema(BaseModel):
    """Pydantic model for the model configuration (from config.yaml)."""
    type: str = Field(..., description="Model class name (e.g., 'GPT2')")
    d_model: int = Field(..., ge=1, description="Embedding dimension")
    block_size: int = Field(..., ge=1, description="Block size")
    n_heads: int = Field(..., ge=1, description="Number of attention heads")
    n_layers: int = Field(..., ge=1, description="Number of transformer blocks")
    dropout: float = Field(..., ge=0.0, le=1.0, description="Dropout rate")

class ModelTrainStatus(BaseModel):
    """Pydantic model for the model training status."""
    current_epoch: int = Field(..., ge=0, description="Current epoch")
    val_loss: float = Field(..., ge=0.0, description="validation loss")
    train_loss: float = Field(..., ge=0.0, description="training loss")
    accuracy: float = Field(..., ge=0.0, description="accuracy")

class ModelEntry(BaseModel):
    """Pydantic model for a complete model entry in MongoDB."""
    id: str = Field(None, alias="_id", description="Unique ID")
    model_schema: ModelSchema = Field(..., description="Model configuration")
    train_status: ModelTrainStatus = Field(..., description="Model training status")
    checkpoint_id: str = Field(..., description="GridFS ID of the checkpoint file")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")

    class Config:
        """Pydantic configuration for MongoDB compatibility."""
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {
            ObjectId: str,  # Convert ObjectId to string for JSON serialization
            datetime: lambda dt: dt.isoformat()  # Convert datetime to ISO string
        }

    @classmethod
    def from_mongo(cls, data: dict) -> "ModelEntry":
        """Convert a MongoDB document to a Pydantic ModelEntry."""
        if "_id" in data:
            data["_id"] = str(data["_id"])
        if "checkpoint_id" in data:
            data["checkpoint_id"] = str(data["checkpoint_id"])
        return cls(**data)