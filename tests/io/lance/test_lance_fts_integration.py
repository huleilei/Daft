"""Integration tests for distributed Lance FTS index creation."""

from __future__ import annotations

import lance
import pyarrow as pa
import pytest

from daft.io.lance import create_fts_index

# Skip all tests if lance is not available
pytestmark = pytest.mark.skipif(not hasattr(lance, "__version__"), reason="Lance package not available")


@pytest.fixture
def sample_text_data():
    """Create sample text data for testing."""
    return {
        "id": list(range(20)),
        "text": [
            "The quick brown fox jumps over the lazy dog",
            "Python is a powerful programming language",
            "Machine learning algorithms are fascinating",
            "Data science requires statistical knowledge",
            "Natural language processing uses text analysis",
            "Distributed computing scales horizontally",
            "Apache Arrow provides columnar data format",
            "Lance format enables efficient vector storage",
            "Full-text search improves data discovery",
            "Indexing accelerates query performance",
            "Ray framework enables parallel processing",
            "Daft provides distributed DataFrame operations",
            "Vector databases support similarity search",
            "Embeddings capture semantic relationships",
            "Neural networks learn complex patterns",
            "Transformers revolutionized NLP tasks",
            "GPU acceleration speeds up computations",
            "Cloud storage provides scalable solutions",
            "Microservices enable modular architectures",
            "APIs facilitate system integration",
        ],
        "category": [
            "animals",
            "tech",
            "ml",
            "data",
            "nlp",
            "distributed",
            "arrow",
            "lance",
            "search",
            "index",
            "ray",
            "daft",
            "vector",
            "embedding",
            "neural",
            "transformer",
            "gpu",
            "cloud",
            "microservice",
            "api",
        ],
        "priority": [i % 3 for i in range(20)],  # 0, 1, 2 priority levels
    }


@pytest.fixture
def large_text_dataset(tmp_path, sample_text_data):
    """Create a larger Lance dataset with multiple fragments for testing."""
    dataset_path = tmp_path / "large_text_dataset.lance"
    table = pa.Table.from_pydict(sample_text_data)

    # Create dataset with multiple fragments (4 rows per fragment = 5 fragments)
    lance.write_dataset(table, str(dataset_path), max_rows_per_file=4)

    return str(dataset_path)


@pytest.fixture
def small_text_dataset(tmp_path):
    """Create a small Lance dataset for basic testing."""
    dataset_path = tmp_path / "small_text_dataset.lance"

    small_data = {
        "id": [1, 2, 3, 4],
        "text": [
            "First document content",
            "Second document content",
            "Third document content",
            "Fourth document content",
        ],
        "metadata": ["meta1", "meta2", "meta3", "meta4"],
    }

    table = pa.Table.from_pydict(small_data)
    lance.write_dataset(table, str(dataset_path), max_rows_per_file=2)

    return str(dataset_path)


class TestCreateFtsIndexBasic:
    """Basic functionality tests for create_fts_index."""

    def test_create_fts_index_basic_inverted(self, large_text_dataset):
        """Test basic distributed INVERTED index creation."""
        updated_dataset = create_fts_index(
            dataset=large_text_dataset,
            column="text",
            index_type="INVERTED",
            num_workers=2,
        )

        # Verify the index was created
        indices = updated_dataset.list_indices()
        assert len(indices) > 0, "No indices found after building"

        # Find our index
        text_indices = [idx for idx in indices if "text" in idx.name]
        assert len(text_indices) > 0, "Text index not found"

        text_index = text_indices[0]
        assert text_index.index_type == "Inverted", f"Expected Inverted index, got {text_index.index_type}"

    def test_create_fts_index_basic_fts(self, large_text_dataset):
        """Test basic distributed FTS index creation."""
        updated_dataset = create_fts_index(
            dataset=large_text_dataset,
            column="text",
            index_type="FTS",
            num_workers=2,
        )

        # Verify the index was created
        indices = updated_dataset.list_indices()
        assert len(indices) > 0, "No indices found after building"

        # Find our index
        text_indices = [idx for idx in indices if "text" in idx.name]
        assert len(text_indices) > 0, "Text index not found"

    def test_create_fts_index_with_custom_name(self, large_text_dataset):
        """Test creating index with custom name."""
        custom_name = "my_custom_text_index"

        updated_dataset = create_fts_index(
            dataset=large_text_dataset,
            column="text",
            index_type="INVERTED",
            name=custom_name,
            num_workers=2,
        )

        # Verify the index was created with correct name
        indices = updated_dataset.list_indices()
        index_names = [idx.name for idx in indices]
        assert custom_name in index_names, f"Custom index name '{custom_name}' not found in {index_names}"

    def test_create_fts_index_search_functionality(self, large_text_dataset):
        """Test that the created index actually works for searching."""
        updated_dataset = create_fts_index(
            dataset=large_text_dataset,
            column="text",
            index_type="INVERTED",
            num_workers=2,
        )

        # Test full-text search functionality
        search_term = "Python"
        results = updated_dataset.scanner(
            full_text_query=search_term,
            columns=["id", "text"],
        ).to_table()

        # Should find at least one result containing "Python"
        assert results.num_rows > 0, f"No results found for search term '{search_term}'"

        # Verify results contain the search term
        text_results = results.column("text").to_pylist()
        assert any(search_term in text for text in text_results), "Search results don't contain the search term"

    def test_create_fts_index_multiple_search_terms(self, large_text_dataset):
        """Test searching with multiple terms after index creation."""
        updated_dataset = create_fts_index(
            dataset=large_text_dataset,
            column="text",
            index_type="INVERTED",
            num_workers=2,
        )

        # Test multiple search terms
        search_terms = ["machine", "learning", "data", "distributed"]

        for term in search_terms:
            results = updated_dataset.scanner(
                full_text_query=term,
                columns=["id", "text", "category"],
            ).to_table()

            if results.num_rows > 0:  # Some terms might not match
                text_results = results.column("text").to_pylist()
                # At least one result should contain the term (case-insensitive)
                assert any(
                    term.lower() in text.lower() for text in text_results
                ), f"Search results for '{term}' don't contain the search term"


class TestCreateFtsIndexWorkerConfiguration:
    """Tests for different worker configurations."""

    def test_create_fts_index_single_worker(self, large_text_dataset):
        """Test distributed index creation with single worker."""
        updated_dataset = create_fts_index(
            dataset=large_text_dataset,
            column="text",
            index_type="INVERTED",
            num_workers=1,
        )

        indices = updated_dataset.list_indices()
        assert len(indices) > 0, "No indices found after building"

    def test_create_fts_index_multiple_workers(self, large_text_dataset):
        """Test distributed index creation with multiple workers."""
        updated_dataset = create_fts_index(
            dataset=large_text_dataset,
            column="text",
            index_type="INVERTED",
            num_workers=4,
        )

        indices = updated_dataset.list_indices()
        assert len(indices) > 0, "No indices found after building"

    def test_create_fts_index_auto_adjust_workers(self, small_text_dataset):
        """Test that num_workers is automatically adjusted if it exceeds fragment count."""
        # Small dataset has only 2 fragments, request more workers
        updated_dataset = create_fts_index(
            dataset=small_text_dataset,
            column="text",
            index_type="INVERTED",
            num_workers=10,  # More than the 2 fragments
        )

        # Should still work and create the index
        indices = updated_dataset.list_indices()
        assert len(indices) > 0, "No indices found after building"

    def test_create_fts_index_with_daft_remote_args(self, large_text_dataset):
        """Test index creation with custom Daft remote arguments."""
        daft_remote_args = {
            "num_cpus": 1,
            "memory_bytes": 1024 * 1024 * 1024,  # 1GB
        }

        updated_dataset = create_fts_index(
            dataset=large_text_dataset,
            column="text",
            index_type="INVERTED",
            num_workers=2,
            daft_remote_args=daft_remote_args,
        )

        indices = updated_dataset.list_indices()
        assert len(indices) > 0, "No indices found after building"

    def test_create_fts_index_with_storage_options(self, large_text_dataset):
        """Test index creation with storage options."""
        storage_options = {}  # Empty storage options should work

        updated_dataset = create_fts_index(
            dataset=large_text_dataset,
            column="text",
            index_type="INVERTED",
            num_workers=2,
            storage_options=storage_options,
        )

        indices = updated_dataset.list_indices()
        assert len(indices) > 0, "No indices found after building"


class TestCreateFtsIndexErrorHandling:
    """Error handling tests for create_fts_index."""

    def test_create_fts_index_invalid_column(self, large_text_dataset):
        """Test error handling for non-existent column."""
        with pytest.raises(ValueError, match="Column 'nonexistent' not found"):
            create_fts_index(
                dataset=large_text_dataset,
                column="nonexistent",
                index_type="INVERTED",
                num_workers=2,
            )

    def test_create_fts_index_invalid_index_type(self, large_text_dataset):
        """Test error handling for invalid index type."""
        with pytest.raises(ValueError, match="Index type must be 'INVERTED' or 'FTS'"):
            create_fts_index(
                dataset=large_text_dataset,
                column="text",
                index_type="INVALID_TYPE",
                num_workers=2,
            )

    def test_create_fts_index_invalid_num_workers(self, large_text_dataset):
        """Test error handling for invalid num_workers."""
        with pytest.raises(ValueError, match="num_workers must be positive"):
            create_fts_index(
                dataset=large_text_dataset,
                column="text",
                index_type="INVERTED",
                num_workers=0,
            )

        with pytest.raises(ValueError, match="num_workers must be positive"):
            create_fts_index(
                dataset=large_text_dataset,
                column="text",
                index_type="INVERTED",
                num_workers=-1,
            )

    def test_create_fts_index_empty_column_name(self, large_text_dataset):
        """Test error handling for empty column name."""
        with pytest.raises(ValueError, match="Column name cannot be empty"):
            create_fts_index(
                dataset=large_text_dataset,
                column="",
                index_type="INVERTED",
                num_workers=2,
            )

    def test_create_fts_index_non_string_column(self, tmp_path):
        """Test error handling for non-string column."""
        # Create dataset with non-string column
        data = {
            "id": [1, 2, 3, 4],
            "numeric_col": [10, 20, 30, 40],
            "text": ["text1", "text2", "text3", "text4"],
        }

        dataset_path = tmp_path / "non_string_test.lance"
        table = pa.Table.from_pydict(data)
        lance.write_dataset(table, str(dataset_path), max_rows_per_file=2)

        with pytest.raises(TypeError, match="must be string type"):
            create_fts_index(
                dataset=str(dataset_path),
                column="numeric_col",
                index_type="INVERTED",
                num_workers=2,
            )

    def test_create_fts_index_invalid_dataset_uri(self):
        """Test error handling for invalid dataset URI."""
        with pytest.raises((ValueError, RuntimeError, OSError)):
            create_fts_index(
                dataset="invalid://path/to/dataset",
                column="text",
                index_type="INVERTED",
                num_workers=2,
            )

    def test_create_fts_index_empty_dataset(self, tmp_path):
        """Test handling of empty dataset."""
        # Create empty dataset
        empty_data = {"text": [], "id": []}
        dataset_path = tmp_path / "empty_dataset.lance"
        table = pa.Table.from_pydict(empty_data)
        lance.write_dataset(table, str(dataset_path))

        with pytest.raises(ValueError, match="Dataset has no fragments"):
            create_fts_index(
                dataset=str(dataset_path),
                column="text",
                index_type="INVERTED",
                num_workers=2,
            )


class TestCreateFtsIndexDatasetInputTypes:
    """Tests for different dataset input types."""

    def test_create_fts_index_with_dataset_uri_string(self, large_text_dataset):
        """Test index creation with dataset URI as string."""
        updated_dataset = create_fts_index(
            dataset=large_text_dataset,  # String URI
            column="text",
            index_type="INVERTED",
            num_workers=2,
        )

        indices = updated_dataset.list_indices()
        assert len(indices) > 0, "No indices found after building"

    def test_create_fts_index_with_lance_dataset_object(self, large_text_dataset):
        """Test index creation with Lance dataset object."""
        dataset_obj = lance.LanceDataset(large_text_dataset)

        updated_dataset = create_fts_index(
            dataset=dataset_obj,  # LanceDataset object
            column="text",
            index_type="INVERTED",
            num_workers=2,
        )

        indices = updated_dataset.list_indices()
        assert len(indices) > 0, "No indices found after building"


class TestCreateFtsIndexAdvancedFeatures:
    """Tests for advanced features and edge cases."""

    def test_create_fts_index_with_kwargs(self, large_text_dataset):
        """Test index creation with additional kwargs."""
        updated_dataset = create_fts_index(
            dataset=large_text_dataset,
            column="text",
            index_type="INVERTED",
            num_workers=2,
            remove_stop_words=False,  # Additional kwarg
        )

        indices = updated_dataset.list_indices()
        assert len(indices) > 0, "No indices found after building"

    def test_create_fts_index_multiple_columns_different_indices(self, tmp_path):
        """Test creating indices on multiple columns."""
        # Create dataset with multiple text columns
        multi_column_data = {
            "id": [1, 2, 3, 4],
            "title": ["Title 1", "Title 2", "Title 3", "Title 4"],
            "content": ["Content 1", "Content 2", "Content 3", "Content 4"],
            "description": ["Desc 1", "Desc 2", "Desc 3", "Desc 4"],
        }

        dataset_path = tmp_path / "multi_column_dataset.lance"
        table = pa.Table.from_pydict(multi_column_data)
        lance.write_dataset(table, str(dataset_path), max_rows_per_file=2)

        # Create indices on different columns
        columns_to_index = ["title", "content", "description"]

        current_dataset = lance.LanceDataset(str(dataset_path))

        for column in columns_to_index:
            updated_dataset = create_fts_index(
                dataset=current_dataset,
                column=column,
                index_type="INVERTED",
                name=f"{column}_index",
                num_workers=1,
            )
            current_dataset = updated_dataset

        # Verify all indices were created
        final_indices = current_dataset.list_indices()
        index_names = [idx.name for idx in final_indices]

        for column in columns_to_index:
            expected_name = f"{column}_index"
            assert expected_name in index_names, f"Index for column '{column}' not found"

    def test_create_fts_index_large_fragment_count(self, tmp_path):
        """Test index creation with many small fragments."""
        # Create dataset with many small fragments
        large_data = {
            "id": list(range(100)),
            "text": [f"Document {i} with unique content for testing" for i in range(100)],
        }

        dataset_path = tmp_path / "many_fragments_dataset.lance"
        table = pa.Table.from_pydict(large_data)
        # Create many small fragments (1 row per fragment = 100 fragments)
        lance.write_dataset(table, str(dataset_path), max_rows_per_file=1)

        updated_dataset = create_fts_index(
            dataset=str(dataset_path),
            column="text",
            index_type="INVERTED",
            num_workers=8,  # Fewer workers than fragments
        )

        indices = updated_dataset.list_indices()
        assert len(indices) > 0, "No indices found after building"

        # Test search functionality
        results = updated_dataset.scanner(
            full_text_query="Document",
            columns=["id", "text"],
        ).to_table()

        # Should find many results
        assert results.num_rows > 50, f"Expected many results, got {results.num_rows}"

    def test_create_fts_index_unicode_content(self, tmp_path):
        """Test index creation with Unicode text content."""
        unicode_data = {
            "id": [1, 2, 3, 4],
            "text": [
                "Hello world in English",
                "Bonjour le monde en français",
                "你好世界用中文",
                "こんにちは世界日本語で",
            ],
            "language": ["en", "fr", "zh", "ja"],
        }

        dataset_path = tmp_path / "unicode_dataset.lance"
        table = pa.Table.from_pydict(unicode_data)
        lance.write_dataset(table, str(dataset_path), max_rows_per_file=2)

        updated_dataset = create_fts_index(
            dataset=str(dataset_path),
            column="text",
            index_type="INVERTED",
            num_workers=2,
        )

        indices = updated_dataset.list_indices()
        assert len(indices) > 0, "No indices found after building"

        # Test search with English term
        results = updated_dataset.scanner(
            full_text_query="Hello",
            columns=["id", "text", "language"],
        ).to_table()

        assert results.num_rows > 0, "No results found for Unicode search"

    def test_create_fts_index_very_long_text(self, tmp_path):
        """Test index creation with very long text documents."""
        # Create documents with varying lengths
        long_texts = [
            "Short text",
            "Medium length text with more words and content",
            "A" * 1000,  # 1KB text
            "Very long document with lots of repeated content. " * 100,  # ~5KB text
        ]

        long_text_data = {
            "id": list(range(len(long_texts))),
            "text": long_texts,
            "length": [len(text) for text in long_texts],
        }

        dataset_path = tmp_path / "long_text_dataset.lance"
        table = pa.Table.from_pydict(long_text_data)
        lance.write_dataset(table, str(dataset_path), max_rows_per_file=2)

        updated_dataset = create_fts_index(
            dataset=str(dataset_path),
            column="text",
            index_type="INVERTED",
            num_workers=2,
        )

        indices = updated_dataset.list_indices()
        assert len(indices) > 0, "No indices found after building"


class TestCreateFtsIndexPerformance:
    """Performance and scalability tests."""

    def test_create_fts_index_performance_comparison(self, tmp_path):
        """Compare performance with different worker counts."""
        # Create a moderately sized dataset
        perf_data = {
            "id": list(range(50)),
            "text": [f"Performance test document {i} with content for benchmarking" for i in range(50)],
        }

        dataset_path = tmp_path / "perf_dataset.lance"
        table = pa.Table.from_pydict(perf_data)
        lance.write_dataset(table, str(dataset_path), max_rows_per_file=5)

        # Test with different worker counts
        worker_counts = [1, 2, 4]

        for num_workers in worker_counts:
            updated_dataset = create_fts_index(
                dataset=str(dataset_path),
                column="text",
                index_type="INVERTED",
                name=f"perf_index_{num_workers}",
                num_workers=num_workers,
            )

            indices = updated_dataset.list_indices()
            assert len(indices) >= num_workers - 1, f"Expected indices for {num_workers} workers"

    def test_create_fts_index_memory_efficiency(self, tmp_path):
        """Test memory efficiency with constrained resources."""
        # Create dataset
        memory_data = {
            "id": list(range(30)),
            "text": [f"Memory efficiency test document {i}" for i in range(30)],
        }

        dataset_path = tmp_path / "memory_dataset.lance"
        table = pa.Table.from_pydict(memory_data)
        lance.write_dataset(table, str(dataset_path), max_rows_per_file=3)

        # Test with memory constraints
        daft_remote_args = {
            "num_cpus": 1,
            "memory_bytes": 512 * 1024 * 1024,  # 512MB limit
        }

        updated_dataset = create_fts_index(
            dataset=str(dataset_path),
            column="text",
            index_type="INVERTED",
            num_workers=2,
            daft_remote_args=daft_remote_args,
        )

        indices = updated_dataset.list_indices()
        assert len(indices) > 0, "No indices found after building"
