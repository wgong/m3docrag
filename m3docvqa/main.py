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

"""Main Script for M3DocVQA Dataset Creation Pipeline.

This script orchestrates downloading PDFs or PNGs, checking for corrupted PDFs, extracting images,
organizing them into directories, downloading/decompressing MMQA data, and creating wiki links mapping.

Usage:
    python main.py <action> [other options]

Actions:
    - download_pdfs: Download PDFs from URLs provided in metadata.
    - check_pdfs: Verify if the downloaded PDFs are valid.
    - extract_images: Extract images from the pages of downloaded PDFs.
    - organize_files: Organize downloaded PDFs into specified directory splits.
    - download_mmqa: Download and decompress the MMQA dataset.
    - generate_wiki_mapping: Generate a mapping of 'id' to 'url' from multiple JSONL files.

Example:
    python main.py generate_wiki_mapping --text=MMQA_texts.jsonl --image=MMQA_images.jsonl --table=MMQA_tables.jsonl --output=id_url_mapping.jsonl
"""

import fire
import json
import jsonlines
from pathlib import Path
from m3docvqa.downloader import download_wiki_page
from m3docvqa.pdf_utils import is_pdf_downloaded, is_pdf_clean, get_images_from_pdf
from m3docvqa.split_utils import create_split_dirs
from m3docvqa.mmqa_downloader import download_and_decompress_mmqa
from m3docvqa.wiki_mapper import generate_wiki_links_mapping
from loguru import logger
from tqdm.auto import tqdm


def _prepare_download(
    metadata_path: Path | str, 
    output_dir: Path | str, 
    first_n: int,
    doc_ids: set,
    ) -> tuple[list[str], list[Path]]:
    """Prepare URLs and save paths for downloading.

    Args:
        metadata_path (Path): Path to the metadata JSONL file.
        output_dir (str): Directory where files will be saved.
        first_n (int): Maximum number of entries to process.
        doc_ids (set): Set of doc ids to filter for downloading.

    Returns:
        tuple[list[str], list[Path]]: URLs and save paths for downloading.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    urls, save_paths = [], []
    with jsonlines.open(metadata_path) as reader:
        for line in reader:
            if len(urls) == first_n:
                break
            
            doc_id = line.get("id")
            if doc_ids and doc_id not in doc_ids:
                continue
            
            url = line.get("url")
            save_path = output_dir / f"{doc_id}.pdf"
            if not is_pdf_downloaded(save_path):
                urls.append(url)
                save_paths.append(save_path)

    return urls, save_paths


def download_pdfs(
    metadata_path: Path | str, 
    pdf_dir: Path | str, 
    result_log_path: Path | str, 
    supporting_doc_ids_per_split: Path | str,
    first_n: int = -1, 
    proc_id: int = 0, 
    n_proc: int = 1,
    split: str = 'dev',
    ):
    """Download Wikipedia pages as PDFs."""
    # Load document ids for the specified split
    if supporting_doc_ids_per_split:
        with open(supporting_doc_ids_per_split, "r") as f:
            doc_ids_per_split = json.load(f)
        split_doc_ids = {
            "train": set(doc_ids_per_split.get("train", [])),
            "dev": set(doc_ids_per_split.get("dev", [])),
            "all": set(doc_ids_per_split.get("train", []) + doc_ids_per_split.get("dev", []))
        }
        if split not in split_doc_ids:
            raise ValueError(f"Invalid or missing split. Expected one of {split_doc_ids.keys()}")
        doc_ids = split_doc_ids.get(split, split_doc_ids.get("all"))
        logger.info(f"Downloading documents for split: {split} with {len(doc_ids)} document IDs.")

    urls, save_paths = _prepare_download(metadata_path, pdf_dir, first_n, doc_ids)
    logger.info(f"Starting download of {len(urls)} PDFs to {pdf_dir}")
    download_results = download_wiki_page(urls, save_paths, "pdf", result_log_path, proc_id, n_proc)
    logger.info(f"Download completed with {sum(download_results)} successful downloads out of {len(urls)}")


def check_pdfs(pdf_dir: str, proc_id: int = 0, n_proc: int = 1):
    """Verifies the integrity of downloaded PDFs."""
    corrupted_paths = []
    total_checked, corrupted_count = 0, 0

    pdf_files = list(Path(pdf_dir).glob("*.pdf"))
    for pdf_path in tqdm(pdf_files, disable=(proc_id != 0), desc="Checking PDFs"):
        total_checked += 1
        if not is_pdf_downloaded(pdf_path) or not is_pdf_clean(pdf_path):
            corrupted_paths.append(pdf_path)
            corrupted_count += 1

    logger.info(f"Checked {total_checked} PDFs: {corrupted_count} corrupted files.")
    if corrupted_paths:
        logger.warning(f"Corrupted PDFs: {corrupted_paths}")


def extract_images(pdf_dir: str, image_dir: str, save_type='png'):
    """Extracts images from downloaded PDFs."""
    Path(image_dir).mkdir(parents=True, exist_ok=True)

    pdf_files = list(Path(pdf_dir).glob("*.pdf"))
    if not pdf_files:
        logger.warning(f"No PDFs found in {pdf_dir} for image extraction.")
        return 

    logger.info(f"Starting image extraction from {len(pdf_files)} PDFs in {pdf_dir}.")

    for pdf_path in tqdm(pdf_files, desc="Extracting images", unit="PDF"):
        get_images_from_pdf(pdf_path, save_dir=image_dir, save_type=save_type)
    logger.info(f"Images extracted from PDFs in {pdf_dir}")


def organize_files(all_pdf_dir: Path | str, target_dir_base: Path | str, split_metadata_file: str | Path, split: str):
    """Organizes PDFs into directory splits based on split information file."""
    create_split_dirs(
        all_pdf_dir=all_pdf_dir, 
        target_dir_base=target_dir_base,
        split_metadata_file=split_metadata_file,
        split=split,
    )
    logger.info(f"Files organized for {split} split: in {target_dir_base}")


def download_mmqa(output_dir: str):
    """Downloads and decompresses the MMQA dataset.

    Args:
        output_dir (str): Directory where the MMQA files will be downloaded and decompressed.
    """
    logger.info(f"Starting MMQA dataset download to {output_dir}")
    download_and_decompress_mmqa(output_dir)
    logger.info(f"MMQA dataset downloaded and decompressed successfully in {output_dir}")


def generate_wiki_mapping(text: str, image: str, table: str, output: str = "id_url_mapping.jsonl"):
    """Generates a mapping of 'id' to 'url' from multiple JSONL files.

    Args:
        text (str): Path to the JSONL file containing text data from multimodalqa dataset with 'id' and 'url' fields.
        image (str): Path to the JSONL file containing image data from multimodalqa dataset with 'id' and 'url' fields.
        table (str): Path to the JSONL file containing table data from multimodalqa dataset with 'id' and 'url' fields.
        output (str): Path to save the output JSONL file. Defaults to 'id_url_mapping.jsonl'.
    """
    logger.info("Starting wiki mapping generation...")
    generate_wiki_links_mapping(text_file=text, image_file=image, table_file=table, output_file=output)
    logger.info(f"Wiki mapping successfully saved to {output}")


def main():
    fire.Fire({
        "download_mmqa": download_mmqa,
        "generate_wiki_mapping": generate_wiki_mapping,
        "download_pdfs": download_pdfs,
        "check_pdfs": check_pdfs,
        "extract_images": extract_images,
        "organize_files": organize_files,
    })


if __name__ == "__main__":
    main()
