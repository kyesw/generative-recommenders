"""
SageMaker inference entry point for the HSTU generative recommender.

Request (JSON):
  {
    "user_id": 42,          # required unless "sequence" is provided
    "top_k": 10,            # optional, default 10
    "sequence": [1, 2, 3]  # optional — bypasses Feature Store lookup
  }

Response (JSON):
  {
    "user_id": 42,
    "item_ids": [101, 55, 203, ...]
  }

Environment variables (set at deploy time):
  GIN_CONFIG_FILE         path to gin config relative to /opt/ml/code/
  DATASET_NAME            dataset identifier (e.g. ml-1m)
  FEATURE_STORE_REGION    AWS region of the Feature Store
  FEATURE_CACHE_SIZE      Max cached users (default: 10000)
  FEATURE_CACHE_TTL       Cache TTL in seconds (default: 300)
"""

import glob
import json
import logging
import os
import sys
import time
from functools import lru_cache

import torch

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# model_fn — called once when the endpoint starts
# ---------------------------------------------------------------------------

def model_fn(model_dir: str) -> dict:
    """Load model, pre-compute item embeddings, initialise Feature Store client."""
    import gin
    from generative_recommenders.research.data.preprocessor import get_common_preprocessors
    from generative_recommenders.research.modeling.sequential.embedding_modules import (
        LocalEmbeddingModule,
    )
    from generative_recommenders.research.modeling.sequential.encoder_utils import (
        get_sequential_encoder,
    )
    from generative_recommenders.research.modeling.sequential.input_features_preprocessors import (
        LearnablePositionalEmbeddingInputFeaturesPreprocessor,
    )
    from generative_recommenders.research.modeling.sequential.output_postprocessors import (
        L2NormEmbeddingPostprocessor,
        LayerNormEmbeddingPostprocessor,
    )
    from generative_recommenders.research.modeling.similarity_utils import (
        get_similarity_function,
    )

    # -----------------------------------------------------------------------
    # 1. Parse gin config
    # -----------------------------------------------------------------------
    gin_config_file = os.environ.get(
        "GIN_CONFIG_FILE",
        "configs/ml-1m/hstu-sampled-softmax-n128-large-final.gin",
    )
    # gin resolves relative paths via frame inspection (__file__ of the caller),
    # which returns None when inference.py is loaded dynamically via importlib.
    # Convert to an absolute path so gin can find the file unconditionally.
    if not os.path.isabs(gin_config_file):
        gin_config_file = os.path.join("/opt/ml/code", gin_config_file)
    dataset_name = os.environ.get("DATASET_NAME", "ml-1m")
    region = os.environ.get("FEATURE_STORE_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))

    gin.parse_config_file(gin_config_file)
    logger.info(f"Gin config loaded: {gin_config_file}")

    # Read training hyperparameters from gin bindings.
    max_sequence_length = gin.query_parameter("train_fn.max_sequence_length")
    item_embedding_dim = gin.query_parameter("train_fn.item_embedding_dim")
    main_module = gin.query_parameter("train_fn.main_module")
    interaction_module_type = gin.query_parameter("train_fn.interaction_module_type")
    user_embedding_norm = gin.query_parameter("train_fn.user_embedding_norm")
    item_l2_norm = gin.query_parameter("train_fn.item_l2_norm")
    l2_norm_eps = gin.query_parameter("train_fn.l2_norm_eps")
    try:
        gr_output_length = gin.query_parameter("train_fn.gr_output_length")
    except ValueError:
        gr_output_length = 10  # default

    # -----------------------------------------------------------------------
    # 2. Resolve max_item_id from the dataset preprocessor
    # -----------------------------------------------------------------------
    dp = get_common_preprocessors()[dataset_name]
    max_item_id = dp.expected_max_item_id()
    logger.info(f"Dataset: {dataset_name}, max_item_id: {max_item_id}")

    # -----------------------------------------------------------------------
    # 3. Build model (gin configures HSTU sub-components automatically)
    # -----------------------------------------------------------------------
    embedding_module = LocalEmbeddingModule(
        num_items=max_item_id,
        item_embedding_dim=item_embedding_dim,
    )
    interaction_module, _ = get_similarity_function(
        module_type=interaction_module_type,
        query_embedding_dim=item_embedding_dim,
        item_embedding_dim=item_embedding_dim,
    )
    output_postproc_module = (
        L2NormEmbeddingPostprocessor(embedding_dim=item_embedding_dim, eps=l2_norm_eps)
        if user_embedding_norm == "l2_norm"
        else LayerNormEmbeddingPostprocessor(embedding_dim=item_embedding_dim, eps=l2_norm_eps)
    )
    input_preproc_module = LearnablePositionalEmbeddingInputFeaturesPreprocessor(
        max_sequence_len=max_sequence_length + gr_output_length + 1,
        embedding_dim=item_embedding_dim,
        dropout_rate=0.0,  # no dropout at inference
    )
    model = get_sequential_encoder(
        module_type=main_module,
        max_sequence_length=max_sequence_length,
        max_output_length=gr_output_length + 1,
        embedding_module=embedding_module,
        interaction_module=interaction_module,
        input_preproc_module=input_preproc_module,
        output_postproc_module=output_postproc_module,
        verbose=False,
    )

    # -----------------------------------------------------------------------
    # 4. Load checkpoint weights
    # -----------------------------------------------------------------------
    ckpt_files = [
        f for f in glob.glob(os.path.join(model_dir, "ckpts", "**"), recursive=True)
        if os.path.isfile(f)
    ]
    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoint found under {model_dir}/ckpts/")
    latest_ckpt = max(ckpt_files, key=os.path.getmtime)
    logger.info(f"Loading checkpoint: {latest_ckpt}")

    checkpoint = torch.load(latest_ckpt, map_location="cpu")
    # Strip the DDP "module." prefix added during distributed training.
    state_dict = {
        k.replace("module.", "", 1): v
        for k, v in checkpoint["model_state_dict"].items()
    }
    model.load_state_dict(state_dict)
    logger.info(f"Checkpoint loaded (epoch {checkpoint.get('epoch', '?')})")

    # -----------------------------------------------------------------------
    # 5. Move model to device and pre-compute normalised item embeddings
    # -----------------------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    with torch.no_grad():
        all_item_ids = torch.arange(1, max_item_id + 1, dtype=torch.long, device=device)
        item_embeddings = model.get_item_embeddings(all_item_ids)  # (N, D)
        # Apply the same L2 normalisation used during training eval.
        if item_l2_norm:
            item_embeddings = item_embeddings / torch.clamp(
                torch.linalg.norm(item_embeddings, dim=-1, keepdim=True),
                min=l2_norm_eps,
            )

    logger.info(f"Item embeddings pre-computed: shape={tuple(item_embeddings.shape)}, device={device}")

    # -----------------------------------------------------------------------
    # 6. Initialise Feature Store client with cache
    # -----------------------------------------------------------------------
    import boto3
    featurestore_client = boto3.client(
        "sagemaker-featurestore-runtime", region_name=region
    )
    
    cache_size = int(os.environ.get("FEATURE_CACHE_SIZE", "10000"))
    cache_ttl = int(os.environ.get("FEATURE_CACHE_TTL", "300"))
    
    cache = {}
    cache_stats = {"hits": 0, "misses": 0}

    return {
        "model": model,
        "item_embeddings": item_embeddings,  # (N, D) normalised, on device
        "featurestore_client": featurestore_client,
        "dataset_name": dataset_name,
        "max_sequence_length": max_sequence_length,
        "max_item_id": max_item_id,
        "item_l2_norm": item_l2_norm,
        "l2_norm_eps": l2_norm_eps,
        "device": device,
        "cache": cache,
        "cache_ttl": cache_ttl,
        "cache_size": cache_size,
        "cache_stats": cache_stats,
    }


# ---------------------------------------------------------------------------
# input_fn
# ---------------------------------------------------------------------------

def input_fn(request_body: str, content_type: str = "application/json") -> dict:
    if content_type != "application/json":
        raise ValueError(f"Unsupported content type: {content_type}")
    return json.loads(request_body)


# ---------------------------------------------------------------------------
# predict_fn
# ---------------------------------------------------------------------------

def predict_fn(data: dict, model_ctx: dict) -> dict:
    from generative_recommenders.research.modeling.sequential.features import SequentialFeatures

    model = model_ctx["model"]
    item_embeddings = model_ctx["item_embeddings"]   # (N, D) normalised
    featurestore_client = model_ctx["featurestore_client"]
    dataset_name = model_ctx["dataset_name"]
    max_sequence_length = model_ctx["max_sequence_length"]
    item_l2_norm = model_ctx["item_l2_norm"]
    l2_norm_eps = model_ctx["l2_norm_eps"]
    device = model_ctx["device"]
    cache = model_ctx["cache"]
    cache_ttl = model_ctx["cache_ttl"]
    cache_size = model_ctx["cache_size"]
    cache_stats = model_ctx["cache_stats"]

    user_id = data.get("user_id")
    top_k = int(data.get("top_k", 10))
    explicit_sequence = data.get("sequence")  # optional override

    # -------------------------------------------------------------------
    # 1. Resolve interaction sequence with cache
    # -------------------------------------------------------------------
    if explicit_sequence is not None:
        sequence_item_ids = [int(x) for x in explicit_sequence]
        sequence_timestamps = list(range(len(sequence_item_ids)))
    else:
        if user_id is None:
            raise ValueError("Request must provide 'user_id' or 'sequence'.")
        
        # Check cache
        cache_key = user_id
        now = time.time()
        
        if cache_key in cache:
            cached_data, cached_time = cache[cache_key]
            if now - cached_time < cache_ttl:
                cache_stats["hits"] += 1
                sequence_item_ids, sequence_timestamps = cached_data
            else:
                del cache[cache_key]
                cached_data = None
        else:
            cached_data = None
        
        if cached_data is None:
            cache_stats["misses"] += 1
            response = featurestore_client.get_record(
                FeatureGroupName=f"user-interactions-{dataset_name}",
                RecordIdentifierValueAsString=str(user_id),
            )
            features = {
                r["FeatureName"]: r["ValueAsString"] for r in response["Record"]
            }
            sequence_item_ids = [
                int(x) for x in features["sequence_item_ids"].split(",") if x
            ]
            sequence_timestamps = [
                int(x) for x in features["sequence_timestamps"].split(",") if x
            ]
            
            # Evict oldest if cache full
            if len(cache) >= cache_size:
                oldest_key = min(cache.keys(), key=lambda k: cache[k][1])
                del cache[oldest_key]
            
            cache[cache_key] = ((sequence_item_ids, sequence_timestamps), now)
        
        if cache_stats["hits"] + cache_stats["misses"] % 100 == 0:
            total = cache_stats["hits"] + cache_stats["misses"]
            hit_rate = cache_stats["hits"] / total if total > 0 else 0
            logger.info(f"Cache stats: {cache_stats['hits']}/{total} hits ({hit_rate:.2%})")

    if not sequence_item_ids:
        return {"user_id": user_id, "item_ids": []}

    # Take the most recent max_sequence_length items (oldest-first ordering
    # is preserved — model expects oldest at index 0, newest at seq_len-1).
    item_ids_trunc = sequence_item_ids[-max_sequence_length:]
    timestamps_trunc = sequence_timestamps[-max_sequence_length:]
    seq_len = len(item_ids_trunc)

    # -------------------------------------------------------------------
    # 2. Build SequentialFeatures
    # -------------------------------------------------------------------
    past_ids = torch.zeros((1, max_sequence_length), dtype=torch.long)
    past_timestamps = torch.zeros((1, max_sequence_length), dtype=torch.long)
    past_ids[0, :seq_len] = torch.tensor(item_ids_trunc, dtype=torch.long)
    past_timestamps[0, :seq_len] = torch.tensor(timestamps_trunc, dtype=torch.long)
    past_lengths = torch.tensor([seq_len], dtype=torch.long)

    seq_features = SequentialFeatures(
        past_lengths=past_lengths.to(device),
        past_ids=past_ids.to(device),
        past_embeddings=None,
        past_payloads={"timestamps": past_timestamps.to(device)},
    )

    # -------------------------------------------------------------------
    # 3. Encode user sequence → query embedding (B, D), already L2-normed
    #    by the model's output_postproc_module (L2NormEmbeddingPostprocessor)
    # -------------------------------------------------------------------
    with torch.no_grad():
        past_embeddings = model.get_item_embeddings(seq_features.past_ids)
        query_embedding = model.encode(
            past_lengths=seq_features.past_lengths,
            past_ids=seq_features.past_ids,
            past_embeddings=past_embeddings,
            past_payloads=seq_features.past_payloads,
        )  # (1, D)

    # -------------------------------------------------------------------
    # 4. Top-K retrieval via dot product with pre-computed item embeddings
    # -------------------------------------------------------------------
    scores = torch.matmul(query_embedding, item_embeddings.T)  # (1, N)

    # Mask out items the user has already interacted with.
    seen_set = set(item_ids_trunc)
    for item_id in seen_set:
        idx = item_id - 1  # item IDs are 1-indexed; embeddings are 0-indexed
        if 0 <= idx < scores.shape[1]:
            scores[0, idx] = float("-inf")

    _, top_k_indices = torch.topk(scores[0], k=min(top_k, scores.shape[1]))
    top_k_item_ids = (top_k_indices + 1).cpu().tolist()  # back to 1-indexed IDs

    return {"user_id": user_id, "item_ids": top_k_item_ids}


# ---------------------------------------------------------------------------
# output_fn
# ---------------------------------------------------------------------------

def output_fn(prediction: dict, accept: str = "application/json") -> str:
    return json.dumps(prediction)
