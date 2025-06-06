import torch
import uuid
import time
import logging
from typing import List, Dict, Tuple, Optional, Callable, Any
from collections import defaultdict

logger = logging.getLogger(__name__)

class EmbeddedTensorStorage:
    """
    A simplified in-memory simulation of a Tensor Database.
    It stores tensors and their associated metadata, organized into datasets.
    This class is designed to be a basic substitute for a real Tensorus instance for demonstration purposes.
    """
    def __init__(self) -> None:
        """
        Initializes the in-memory storage for datasets.
        Each dataset can store multiple tensors and their corresponding metadata.
        """
        self.datasets: Dict[str, Dict[str, List[Any]]] = defaultdict(lambda: {"tensors": [], "metadata": []})
        logger.info("EmbeddedTensorStorage initialized (In-Memory Simulation).")

    def create_dataset(self, name: str) -> None:
        """
        Creates a new dataset (akin to a table or collection) to store tensors and metadata.
        If the dataset already exists, this method does nothing and logs a warning.

        Args:
            name: The unique name for the dataset.

        Raises:
            TypeError: If the dataset name is not a string.
            ValueError: If the dataset name is empty.
        """
        if not isinstance(name, str):
            raise TypeError("Dataset name must be a string.")
        if not name:
            raise ValueError("Dataset name cannot be empty.")

        if name in self.datasets:
            logger.warning(f"Dataset '{name}' already exists. No action taken.")
        else:
            # defaultdict already handles creation, this is more for explicit logging
            self.datasets[name] # Access to ensure it's in the dict if it wasn't already
            logger.info(f"Dataset '{name}' created successfully in the simulated storage.")

    def insert(self, dataset_name: str, tensor: torch.Tensor, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Inserts a tensor and its associated metadata into a specified dataset.

        Args:
            dataset_name: The name of the dataset to insert into.
            tensor: The torch.Tensor object to be stored.
            metadata: An optional dictionary of metadata associated with the tensor.
                      A unique 'record_id' and other default attributes will be generated
                      and added if not provided or if they are missing.

        Returns:
            str: The unique record ID for the inserted tensor (either provided or generated).

        Raises:
            TypeError: If 'dataset_name' is not a string, or if the 'tensor' argument is not a torch.Tensor.
            ValueError: If 'dataset_name' is empty.
        """
        if not isinstance(dataset_name, str):
            raise TypeError(f"Dataset name must be a string. Got: {type(dataset_name)}")
        if not dataset_name:
            raise ValueError("Dataset name for insertion cannot be empty.")
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Data to be inserted must be a torch.Tensor. Got: {type(tensor)}")

        # Ensure dataset exists (it will be auto-created by defaultdict if new)
        # This line also makes it explicit for clarity, though defaultdict handles it.
        _ = self.datasets[dataset_name]

        current_metadata = metadata.copy() if metadata is not None else {}

        # Ensure essential metadata fields are present
        record_id = current_metadata.get("record_id", str(uuid.uuid4()))
        current_metadata["record_id"] = record_id
        current_metadata.setdefault("timestamp_utc", time.time())
        current_metadata.setdefault("shape", list(tensor.shape))
        current_metadata.setdefault("dtype", str(tensor.dtype).replace('torch.', ''))

        self.datasets[dataset_name]["tensors"].append(tensor.clone()) # Store a clone
        self.datasets[dataset_name]["metadata"].append(current_metadata)
        logger.debug(f"Inserted tensor with ID {record_id} into dataset '{dataset_name}'.")
        return record_id

    def get_records_by_metadata_filter(self, dataset_name: str, filter_fn: Callable[[Dict[str, Any]], bool]) -> List[Dict[str, Any]]:
        """
        Retrieves records (tensor and its metadata) from a dataset that match a given filter function.
        The filter function is applied to the metadata of each record.

        Args:
            dataset_name: The name of the dataset to query.
            filter_fn: A callable that accepts a metadata dictionary and returns True if the record should be included.

        Returns:
            A list of dictionaries, where each dictionary contains a 'tensor' and its 'metadata' for matching records.
            Returns an empty list if the dataset does not exist or no records match.

        Raises:
            TypeError: If 'dataset_name' is not a string or 'filter_fn' is not callable.
            ValueError: If 'dataset_name' is empty.
        """
        if not isinstance(dataset_name, str):
            raise TypeError(f"Dataset name must be a string. Got: {type(dataset_name)}")
        if not dataset_name:
            raise ValueError("Dataset name for query cannot be empty.")
        if not callable(filter_fn):
            raise TypeError(f"filter_fn must be a callable function. Got: {type(filter_fn)}")

        if dataset_name not in self.datasets:
            logger.warning(f"Attempted to query non-existent dataset: '{dataset_name}'")
            return []

        results: List[Dict[str, Any]] = []
        for i, meta_item in enumerate(self.datasets[dataset_name]["metadata"]):
            try:
                if filter_fn(meta_item):
                    results.append({"tensor": self.datasets[dataset_name]["tensors"][i], "metadata": meta_item})
            except Exception as e:
                logger.error(f"Error applying filter function to metadata item {meta_item.get('record_id', 'N/A')} in dataset '{dataset_name}': {e}")
                # Optionally, re-raise or handle more gracefully depending on desired strictness

        logger.debug(f"Retrieved {len(results)} records from '{dataset_name}' using filter.")
        return results

    def get_all_records(self, dataset_name: str) -> List[Dict[str, Any]]:
        """
        Retrieves all records (tensor and metadata) from a specified dataset.

        Args:
            dataset_name: The name of the dataset.

        Returns:
            A list of all records in the dataset. Each record is a dictionary with 'tensor' and 'metadata'.
            Returns an empty list if the dataset does not exist.

        Raises:
            TypeError: If 'dataset_name' is not a string.
            ValueError: If 'dataset_name' is empty.
        """
        if not isinstance(dataset_name, str):
            raise TypeError(f"Dataset name must be a string. Got: {type(dataset_name)}")
        if not dataset_name:
            raise ValueError("Dataset name for get_all_records cannot be empty.")

        if dataset_name not in self.datasets:
            logger.warning(f"Attempted to get all records from non-existent dataset: '{dataset_name}'")
            return []

        return [
            {"tensor": t, "metadata": m}
            for t, m in zip(self.datasets[dataset_name]["tensors"], self.datasets[dataset_name]["metadata"])
        ]

    def delete_dataset(self, dataset_name: str) -> bool:
        """
        Deletes an entire dataset and all its contents.

        Args:
            dataset_name: The name of the dataset to delete.

        Returns:
            True if the dataset was found and deleted, False otherwise.

        Raises:
            TypeError: If 'dataset_name' is not a string.
            ValueError: If 'dataset_name' is empty.
        """
        if not isinstance(dataset_name, str):
            raise TypeError(f"Dataset name must be a string. Got: {type(dataset_name)}")
        if not dataset_name:
            raise ValueError("Dataset name for deletion cannot be empty.")

        if dataset_name in self.datasets:
            del self.datasets[dataset_name]
            logger.info(f"Dataset '{dataset_name}' and all its records have been deleted.")
            return True
        else:
            logger.warning(f"Attempted to delete non-existent dataset: '{dataset_name}'. No action taken.")
            return False

    def list_datasets(self) -> List[str]:
        """
        Lists the names of all existing datasets.

        Returns:
            A list of strings, where each string is a dataset name.
        """
        return list(self.datasets.keys())

    def get_dataset_size(self, dataset_name: str) -> Optional[int]:
        """
        Returns the number of records (tensor-metadata pairs) in a given dataset.

        Args:
            dataset_name: The name of the dataset.

        Returns:
            The number of records in the dataset, or None if the dataset does not exist.

        Raises:
            TypeError: If 'dataset_name' is not a string.
            ValueError: If 'dataset_name' is empty.
        """
        if not isinstance(dataset_name, str):
            raise TypeError(f"Dataset name must be a string. Got: {type(dataset_name)}")
        if not dataset_name:
            raise ValueError("Dataset name for get_dataset_size cannot be empty.")

        if dataset_name in self.datasets:
            return len(self.datasets[dataset_name]["metadata"])
        else:
            logger.warning(f"Attempted to get size of non-existent dataset: '{dataset_name}'")
            return None

# Example Usage (typically you would run this in a separate script that imports this class)
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG) # Enable DEBUG for detailed output from the class

    # --- Basic Setup and Dataset Creation ---
    storage = EmbeddedTensorStorage()
    storage.create_dataset("my_image_embeddings")
    storage.create_dataset("my_text_features")

    # Attempting to create an existing dataset (should show a warning)
    storage.create_dataset("my_image_embeddings")

    print(f"Available datasets: {storage.list_datasets()}") # Expected: ['my_image_embeddings', 'my_text_features']

    # --- Error Handling for Dataset Creation ---
    try:
        storage.create_dataset(123) # type: ignore
    except TypeError as e:
        print(f"Caught expected error: {e}")
    try:
        storage.create_dataset("")
    except ValueError as e:
        print(f"Caught expected error: {e}")

    # --- Tensor Insertion ---
    # Example 1: Image embedding (e.g., from a CNN)
    img_tensor = torch.rand(1, 512) # Example: 1 embedding of 512 dimensions
    img_meta = {"source_image_id": "img_001.jpg", "model_name": "ResNet50"}
    record_id1 = storage.insert("my_image_embeddings", img_tensor, img_meta)
    print(f"Inserted image tensor, record ID: {record_id1}")

    # Example 2: Text features (e.g., from a Transformer)
    text_tensor = torch.rand(1, 10, 768) # Example: 1 text, 10 tokens, 768 dims per token
    text_meta = {"document_id": "doc_abc", "text_layer": -1}
    record_id2 = storage.insert("my_text_features", text_tensor, text_meta)
    print(f"Inserted text tensor, record ID: {record_id2}")

    # Example 3: Another image embedding, auto-generated metadata fields
    img_tensor2 = torch.rand(1, 512)
    record_id3 = storage.insert("my_image_embeddings", img_tensor2) # No metadata provided
    print(f"Inserted another image tensor with auto-metadata, record ID: {record_id3}")

    # --- Error Handling for Insertion ---
    try:
        storage.insert("my_image_embeddings", [1,2,3], img_meta) # type: ignore
    except TypeError as e:
        print(f"Caught expected error: {e}")
    try:
        storage.insert(123, img_tensor, img_meta) # type: ignore
    except TypeError as e:
        print(f"Caught expected error: {e}")
    try:
        storage.insert("", img_tensor, img_meta)
    except ValueError as e:
        print(f"Caught expected error: {e}")


    # --- Data Retrieval ---
    print(f"Size of 'my_image_embeddings': {storage.get_dataset_size('my_image_embeddings')}") # Expected: 2
    print(f"Size of 'my_text_features': {storage.get_dataset_size('my_text_features')}")   # Expected: 1
    print(f"Size of 'non_existent_dataset': {storage.get_dataset_size('non_existent_dataset')}") # Expected: None

    # Get all records from 'my_image_embeddings'
    all_image_records = storage.get_all_records("my_image_embeddings")
    print(f"Found {len(all_image_records)} records in 'my_image_embeddings'. First record ID: {all_image_records[0]['metadata']['record_id']}")

    # Filter records: find image embeddings from 'ResNet50' model
    resnet_records = storage.get_records_by_metadata_filter(
        "my_image_embeddings",
        lambda meta: meta.get("model_name") == "ResNet50"
    )
    print(f"Found {len(resnet_records)} ResNet50 records. Record ID: {resnet_records[0]['metadata']['record_id']}")

    # Filter with problematic function
    storage.get_records_by_metadata_filter(
        "my_image_embeddings",
        lambda meta: meta["non_existent_key"] == "some_value" # This will cause an error during filter_fn execution
    )


    # --- Error Handling for Retrieval ---
    try:
        storage.get_all_records(123) # type: ignore
    except TypeError as e:
        print(f"Caught expected error: {e}")
    try:
        storage.get_records_by_metadata_filter("my_image_embeddings", "not_a_function") # type: ignore
    except TypeError as e:
        print(f"Caught expected error: {e}")

    # --- Deletion ---
    storage.delete_dataset("my_text_features") # Expected: True, with log message
    print(f"Available datasets after deletion: {storage.list_datasets()}") # Expected: ['my_image_embeddings']
    storage.delete_dataset("non_existent_dataset") # Expected: False, with log message

    # --- Error Handling for Deletion ---
    try:
        storage.delete_dataset(123) # type: ignore
    except TypeError as e:
        print(f"Caught expected error: {e}")

    print("EmbeddedTensorStorage demonstration complete.")
