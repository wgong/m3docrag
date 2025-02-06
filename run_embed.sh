export LOCAL_ROOT="/opt/m3doc"    # custom folder
export LOCAL_DATA_DIR="$LOCAL_ROOT/job/datasets"
export LOCAL_EMBEDDINGS_DIR="$LOCAL_ROOT/job/embeddings"
export LOCAL_MODEL_DIR="$LOCAL_ROOT/job/model"
export LOCAL_OUTPUT_DIR="$LOCAL_ROOT/job/output"


DATASET_NAME="m3-docvqa"

RETRIEVAL_MODEL_TYPE="colpali"
RETRIEVAL_MODEL_NAME="colpaligemma-3b-pt-448-base"
RETRIEVAL_ADAPTER_MODEL_NAME="colpali-v1.2"
SPLIT="dev"
EMBEDDING_NAME=$RETRIEVAL_ADAPTER_MODEL_NAME"_"$DATASET_NAME"_"$SPLIT  # where to save embeddings
accelerate launch --num_processes=1 --mixed_precision=bf16 examples/run_page_embedding.py \
    --use_retrieval \
    --retrieval_model_type=$RETRIEVAL_MODEL_TYPE \
    --data_name=$DATASET_NAME \
    --split=$SPLIT \
    --loop_unique_doc_ids=True \
    --output_dir=$LOCAL_EMBEDDINGS_DIR/$EMBEDDING_NAME \
    --retrieval_model_name_or_path=$RETRIEVAL_MODEL_NAME \
    --retrieval_adapter_model_name_or_path=$RETRIEVAL_ADAPTER_MODEL_NAME    