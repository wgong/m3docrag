# Copyright 2024 Bloomberg Finance L.P.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Split Utilities Module for M3DocVQA.

This module provides utilities for organizing PDF files into split directories (e.g., train, dev)
and compressing these directories using functions from compression_utils.

Functions:
    - create_split_dirs: Copies specified PDF files into separate split directories (e.g., train, dev).
    - compress_split_directory: Compresses the split directory into a `.tar.gz` archive.
"""

from pathlib import Path
import shutil
import json
import jsonlines
from loguru import logger

def create_split_dirs(
    all_pdf_dir: str | Path,
    target_dir_base: str | Path,
    split_metadata_file: str | Path,
    split: str
) -> None:
    """Copies specified PDF files into a target directory based on a given split.

    Args:
        all_pdf_dir (Union[str, Path]): Path to the directory containing all downloaded PDF files.
        target_dir_base (Union[str, Path]): Base directory where the split-specific directory will be created.
        split_metadata_file (Union[str, Path]): Path to the metadata JSONL file for the split.
        split (str): Split type ('train' or 'dev').

    Raises:
        FileNotFoundError: If the JSONL metadata file does not exist.
        ValueError: If the split is not 'train' or 'dev'.
    """
    # Validate split type
    if split not in {"train", "dev"}:
        raise ValueError(f"Invalid split: {split}. Expected 'train' or 'dev'.")

    all_pdf_dir = Path(all_pdf_dir)
    target_dir = Path(target_dir_base) / f'pdfs_{split}'
    target_dir.mkdir(parents=True, exist_ok=True)

    # Validate metadata file
    split_metadata_file = Path(split_metadata_file)
    if not split_metadata_file.exists():
        raise FileNotFoundError(f"Metadata file for split '{split}' not found: {split_metadata_file}")

    # Load all doc IDs for the split
    split_doc_ids = []
    with jsonlines.open(split_metadata_file) as reader:
        for obj in reader:
            split_doc_ids.extend(doc['doc_id'] for doc in obj['supporting_context'])

    # Remove duplicates and log the count
    split_doc_ids = sorted(set(split_doc_ids))
    logger.info(f"Split {split} -> # supporting context: {len(split_doc_ids)}")

    # Save the split-specific IDs to a JSON file
    split_doc_ids_output_path = Path(f'./{split}_doc_ids.json')
    with open(split_doc_ids_output_path, 'w') as f:
        json.dump(split_doc_ids, f, indent=4)
    logger.info(f"Split {split} -> saved doc IDs at {split_doc_ids_output_path}")

    # Copy PDF files to the target directory
    missing_files = []
    for doc_id in split_doc_ids:
        pdf_file = all_pdf_dir / f"{doc_id}.pdf"
        if pdf_file.exists():
            shutil.copy(pdf_file, target_dir / pdf_file.name)
        else:
            missing_files.append(pdf_file)

    if missing_files:
        logger.warning(f"Warning: {len(missing_files)} files are missing and will be skipped.")
        for missing_file in missing_files:
            logger.warning(f"  Missing: {missing_file}")

