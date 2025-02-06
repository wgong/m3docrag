export LOCAL_ROOT="/opt/m3doc"    # custom folder
export LOCAL_DATA_DIR="$LOCAL_ROOT/job/datasets"
export LOCAL_EMBEDDINGS_DIR="$LOCAL_ROOT/job/embeddings"
export LOCAL_MODEL_DIR="$LOCAL_ROOT/job/model"
export LOCAL_OUTPUT_DIR="$LOCAL_ROOT/job/output"


DATASET_NAME="m3-docvqa"
RETRIEVAL_MODEL_TYPE="colpali"
RETRIEVAL_ADAPTER_MODEL_NAME="colpali-v1.2"
SPLIT="dev"
FAISS_INDEX_TYPE='ivfflat'
EMBEDDING_NAME=$RETRIEVAL_ADAPTER_MODEL_NAME"_"$DATASET_NAME"_"$SPLIT
INDEX_NAME=$EMBEDDING_NAME"_pageindex_"$FAISS_INDEX_TYPE # where to save resulting index
echo $EMBEDDING_NAME
echo $FAISS_INDEX_TYPE
python examples/run_indexing_m3docvqa.py \
    --use_retrieval \
    --retrieval_model_type=$RETRIEVAL_MODEL_TYPE \
    --data_name=$DATASET_NAME \
    --split=$SPLIT \
    --loop_unique_doc_ids=False \
    --embedding_name=$EMBEDDING_NAME \
    --faiss_index_type=$FAISS_INDEX_TYPE \
    --output_dir=$LOCAL_EMBEDDINGS_DIR/$INDEX_NAME