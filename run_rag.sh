export LOCAL_ROOT="/opt/m3doc"    # custom folder
export LOCAL_DATA_DIR="$LOCAL_ROOT/job/datasets"
export LOCAL_EMBEDDINGS_DIR="$LOCAL_ROOT/job/embeddings"
export LOCAL_MODEL_DIR="$LOCAL_ROOT/job/model"
export LOCAL_OUTPUT_DIR="$LOCAL_ROOT/job/output"


BACKBONE_MODEL_NAME="Qwen2-VL-7B-Instruct"
RETRIEVAL_MODEL_TYPE="colpali"
RETRIEVAL_MODEL_NAME="colpaligemma-3b-pt-448-base"
RETRIEVAL_ADAPTER_MODEL_NAME="colpali-v1.2"
EMBEDDING_NAME="colpali-v1.2_m3-docvqa_dev" # from Step 1 Embedding
SPLIT="dev"
DATASET_NAME="m3-docvqa"
FAISS_INDEX_TYPE='ivfflat'
N_RETRIEVAL_PAGES=1
INDEX_NAME="${EMBEDDING_NAME}_pageindex_$FAISS_INDEX_TYPE" # from Step 2 Indexing
OUTPUT_SAVE_NAME="${RETRIEVAL_ADAPTER_MODEL_NAME}_${BACKBONE_MODEL_NAME}_${DATASET_NAME}" # where to save RAG results
BITS=16 # BITS=4 for 4-bit qunaitzation in low memory GPUs
python examples/run_rag_m3docvqa.py \
    --use_retrieval \
    --retrieval_model_type=$RETRIEVAL_MODEL_TYPE \
    --load_embedding=True \
    --split=$SPLIT \
    --bits=$BITS \
    --n_retrieval_pages=$N_RETRIEVAL_PAGES \
    --data_name=$DATASET_NAME \
    --model_name_or_path=$BACKBONE_MODEL_NAME \
    --embedding_name=$EMBEDDING_NAME \
    --retrieval_model_name_or_path=$RETRIEVAL_MODEL_NAME \
    --retrieval_adapter_model_name_or_path=$RETRIEVAL_ADAPTER_MODEL_NAME \
    --output_dir=$LOCAL_OUTPUT_DIR/$OUTPUT_SAVE_NAME