from __future__ import annotations

import logging
import uuid
from typing import Any

import lance
from lance.dataset import Index, LanceDataset

import daft
from daft import DataType, from_pylist
from daft.dependencies import pa

logger = logging.getLogger(__name__)


@daft.func(  # type: ignore[operator]
    return_dtype=DataType.struct(
        {
            "status": DataType.string(),
            "fragment_ids": DataType.list(DataType.int32()),
            "fields": DataType.list(DataType.int32()),
            "uuid": DataType.string(),
            "error": DataType.string(),
        }
    ),
)
def handle_fragment_index(
    dataset_uri: str,
    column: str,
    index_type: str,
    name: str,
    fragment_uuid: str,
    replace: bool,
    train: bool,
    fragment_ids: list[int],
    storage_options: dict[str, str] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    try:
        # Basic input validation
        if not fragment_ids:
            raise ValueError("fragment_ids cannot be empty")

        # Validate fragment_id ranges
        for fragment_id in fragment_ids:
            if fragment_id < 0 or fragment_id > 0xFFFFFFFF:
                raise ValueError(f"Invalid fragment_id: {fragment_id}")

        # Load dataset
        dataset = LanceDataset(dataset_uri, storage_options=storage_options)

        # Validate fragments exist
        available_fragments = {f.fragment_id for f in dataset.get_fragments()}
        invalid_fragments = set(fragment_ids) - available_fragments
        if invalid_fragments:
            raise ValueError(f"Fragment IDs {invalid_fragments} do not exist")

        # Use the distributed index building API - Phase 1: Fragment index creation
        logger.info("Building distributed index for fragments %s using create_scalar_index", fragment_ids)

        # Call create_scalar_index directly - no return value expected
        # After execution, fragment-level indices are automatically built
        dataset.create_scalar_index(
            column=column,
            index_type=index_type,
            name=name,
            replace=replace,
            train=train,
            fragment_uuid=fragment_uuid,
            fragment_ids=fragment_ids,
            **kwargs,
        )

        # Get field ID for the indexed column
        field_id = dataset.schema.get_field_index(column)
        logger.info("Fragment index created successfully for fragments %s", fragment_ids)
        return {
            "status": "success",
            "fragment_ids": fragment_ids,
            "fields": [field_id],
            "uuid": fragment_uuid,
        }
    except Exception as e:
        logger.error("Fragment index task failed for fragments %s: %s", fragment_ids, str(e))
        return {
            "status": "error",
            "fragment_ids": fragment_ids,
            "error": str(e),
        }


def _generate_default_index_name(dataset: lance.LanceDataset, column: str, index_type: str) -> str:
    """Generate a default index name based on column name and index type."""
    base_name = f"{column}_{index_type.lower()}_idx"

    if dataset is None:
        return base_name

    # Check for existing indices with similar names
    existing_indices = dataset.list_indices()
    existing_names = {idx.name for idx in existing_indices}

    if base_name not in existing_names:
        return base_name

    # Generate a unique name by appending a counter
    counter = 1
    while f"{base_name}_{counter}" in existing_names:
        counter += 1

    return f"{base_name}_{counter}"


def _distribute_fragments_balanced(fragments: list[Any], num_workers: int) -> list[dict[str, list[Any]]]:
    """Distribute fragments across workers using a balanced algorithm that considers fragment sizes.

    This function implements a greedy algorithm that assigns fragments to the worker
    with the currently smallest total workload, helping to balance the processing
    time across workers.

    Args:
        fragments: List of Lance fragment objects
        num_workers: Number of workers to distribute fragments across
        logger: Logger instance for debugging information

    Returns:
        List of lists, where each inner list contains fragment IDs for one worker
    """
    if not fragments:
        return [{"fragment_ids": []} for _ in range(num_workers)]

    # Get fragment information (ID and size)
    fragment_info = []
    for fragment in fragments:
        try:
            # Try to get fragment size information
            # fragment.count_rows() gives us the number of rows in the fragment
            row_count = fragment.count_rows()
            fragment_info.append(
                {
                    "id": fragment.fragment_id,
                    "size": row_count,
                }
            )
        except Exception as e:
            # If we can't get size info, use fragment_id as a fallback
            logger.warning(
                "Could not get size for fragment %s: %s. " "Using fragment_id as size estimate.",
                fragment.fragment_id,
                e,
            )
            fragment_info.append(
                {
                    "id": fragment.fragment_id,
                    "size": fragment.fragment_id,  # Fallback to fragment_id
                }
            )

    # Sort fragments by size in descending order (largest first)
    # This helps with better load balancing using the greedy algorithm
    fragment_info.sort(key=lambda x: x["size"], reverse=True)

    # Initialize worker batches and their current workloads
    worker_batches: list[list[int]] = [[] for _ in range(num_workers)]
    worker_workloads = [0] * num_workers

    # Greedy assignment: assign each fragment to the worker with minimum workload
    for frag_info in fragment_info:
        # Find the worker with the minimum current workload
        min_workload_idx = min(range(num_workers), key=lambda i: worker_workloads[i])

        # Assign fragment to this worker
        worker_batches[min_workload_idx].append(frag_info["id"])
        worker_workloads[min_workload_idx] += frag_info["size"]

    # Log distribution statistics for debugging
    total_size = sum(frag_info["size"] for frag_info in fragment_info)
    logger.info("Fragment distribution statistics:")
    logger.info("  Total fragments: %d", len(fragment_info))
    logger.info("  Total size: %d", total_size)
    logger.info("  Workers: %d", num_workers)

    for i, (batch, workload) in enumerate(zip(worker_batches, worker_workloads)):
        percentage = (workload / total_size * 100) if total_size > 0 else 0
        logger.info("  Worker %d: %d fragments, " "workload: %d (%d%%)", i, len(batch), workload, percentage)

    # Filter out empty batches (shouldn't happen with proper input validation)
    non_empty_batches = [
        {
            "fragment_ids": batch,
        }
        for batch in worker_batches
        if batch
    ]

    return non_empty_batches


def create_fts_index_internal(
    lance_ds: lance.LanceDataset,
    uri: str,
    *,
    column: str,
    index_type: str = "INVERTED",
    name: str | None = None,
    replace: bool = True,
    train: bool = True,
    fragment_ids: list[int] | None = None,
    fragment_uuid: str | None = None,
    storage_options: dict[str, str] | None = None,
    daft_remote_args: dict[str, Any] | None = None,
    concurrency: int | None = None,
    **kwargs: Any,
) -> lance.LanceDataset:
    """Internal implementation of distributed FTS index creation using Daft UDFs.

    This function implements the 3-phase distributed indexing workflow:
    Phase 1: Fragment parallel processing using Daft UDFs
    Phase 2: Index metadata merging
    Phase 3: Atomic index creation and commit
    """
    # Input validation
    if not column:
        raise ValueError("Column name cannot be empty")

    if index_type not in ["INVERTED", "FTS"]:
        raise ValueError(f"Index type must be 'INVERTED' or 'FTS', not '{index_type}'")

    # Validate column exists and has correct type
    try:
        field = lance_ds.schema.field(column)
    except KeyError as e:
        available_columns = [field.name for field in lance_ds.schema]
        raise ValueError(f"Column '{column}' not found. Available: {available_columns}") from e

    # Check column type
    value_type = field.type
    if pa.types.is_list(field.type) or pa.types.is_large_list(field.type):
        value_type = field.type.value_type

    if not pa.types.is_string(value_type) and not pa.types.is_large_string(value_type):
        raise TypeError(f"Column {column} must be string type, got {value_type}")

    # Generate index name if not provided
    if name is None:
        name = f"{column}_idx"

    # Get fragments and validate fragment IDs
    fragments = lance_ds.get_fragments()
    if not fragments:
        raise ValueError("Dataset contains no fragments")

    # Handle fragment_ids parameter - if provided, filter fragments
    if fragment_ids is not None:
        available_fragment_ids = {f.fragment_id for f in fragments}
        invalid_fragments = set(fragment_ids) - available_fragment_ids
        if invalid_fragments:
            raise ValueError(f"Fragment IDs {invalid_fragments} do not exist in dataset")
        # Filter fragments to only include requested ones
        fragments = [f for f in fragments if f.fragment_id in fragment_ids]
        fragment_ids_to_use = fragment_ids
    else:
        fragment_ids_to_use = [fragment.fragment_id for fragment in fragments]

    # Adjust num_workers based on fragment count
    if concurrency is None:
        concurrency = len(fragment_ids_to_use)

    # Adjust concurrency if needed
    if concurrency > len(fragment_ids_to_use):
        concurrency = len(fragment_ids_to_use)
        logger.info("Adjusted concurrency to %d to match fragment count", concurrency)

    # Generate unique index ID
    index_id = str(uuid.uuid4())

    logger.info(
        "Starting distributed FTS index creation: column=%s, type=%s, name=%s, workers=%s",
        column,
        index_type,
        name,
        concurrency,
    )

    # Phase 1: Fragment parallel processing using Daft UDFs
    logger.info("Phase 1: Starting fragment parallel processing")

    # Create DataFrame with fragment batches
    fragment_data = _distribute_fragments_balanced(fragments, concurrency)
    df = from_pylist(fragment_data)

    # Configure Daft remote args
    daft_remote_args = daft_remote_args or {}

    df = df.select(
        handle_fragment_index(
            dataset_uri=uri,
            column=column,
            index_type=index_type,
            name=name,
            fragment_uuid=index_id,
            replace=replace,
            train=train,
            fragment_ids=df["fragment_ids"],
            storage_options=storage_options,
            **kwargs,
        ).alias("index_result")
    )
    df.show()  # TODO delete
    results = df.to_pandas()["index_result"]

    # Check for failures
    failed_results = [r for r in results if r["status"] == "error"]
    if failed_results:
        error_messages = [r["error"] for r in failed_results]
        raise RuntimeError(
            f"Index building failed on {len(failed_results)} fragment batches: {'; '.join(error_messages)}"
        )

    successful_results = [r for r in results if r["status"] == "success"]
    if not successful_results:
        raise RuntimeError("No successful index building results")

    logger.info("Phase 1 completed successfully: %d fragment batches processed", len(successful_results))

    # Phase 2: Index metadata merging
    logger.info("Phase 2: Starting index metadata merging")

    # Reload dataset to get latest state
    lance_ds = lance.LanceDataset(uri, storage_options=storage_options)
    try:
        lance_ds.merge_index_metadata(index_id, index_type)
    except TypeError:
        lance_ds.merge_index_metadata(index_id)

    logger.info("Phase 2 completed: Index metadata merged")

    # Phase 3: Atomic index creation and commit
    logger.info("Phase 3: Starting atomic index creation and commit")

    # Get field information from successful results
    fields = successful_results[0]["fields"]

    # Create index object
    index = Index(
        uuid=index_id,
        name=name,
        fields=fields,
        dataset_version=lance_ds.version,
        fragment_ids=set(fragment_ids_to_use),
        index_version=0,
    )

    # Create and commit the index operation
    create_index_op = lance.LanceOperation.CreateIndex(
        new_indices=[index],
        removed_indices=[],
    )

    # Commit the index operation atomically
    updated_dataset = lance.LanceDataset.commit(
        uri,
        create_index_op,
        read_version=lance_ds.version,
        storage_options=storage_options,
    )

    logger.info("Phase 3 completed: Index '%s' created successfully with ID %s", name, index_id)
    return updated_dataset
