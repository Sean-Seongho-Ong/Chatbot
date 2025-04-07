from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from typing import List, Optional, TypedDict, Annotated, Sequence, Dict, Any, Tuple, Set
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from peft import PeftModel, PeftConfig
import logging
import time
import uuid
import asyncio
from fastapi.responses import StreamingResponse
import sys
from langchain_community.llms import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from huggingface_hub import login, snapshot_download
import qdrant_client
import traceback
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langchain_core.documents import Document
from dotenv import load_dotenv
import shutil
from functools import lru_cache
from prometheus_client import Counter, Histogram, start_http_server, CollectorRegistry
import psutil
import gc
import glob
from contextlib import asynccontextmanager
import re
import json
from datetime import datetime

import uvicorn

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# .env 파일 로드
load_dotenv()

# 환경 변수 설정
ENV = os.getenv("ENV", "development")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_BASE_MODEL_PATH = os.getenv("LOCAL_BASE_MODEL_PATH")
LOCAL_ADAPTER_PATH = os.getenv("LOCAL_ADAPTER_PATH")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-large-en-v1.5")
SUMMARIZER_MODEL_NAME = os.getenv("SUMMARIZER_MODEL_NAME", "facebook/bart-large-cnn")
URL_LINK = os.getenv("URL_LINK", "C:/Regluatory SaaS/FCC_PDFs")

# Qdrant 설정
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "fcc_kdb_qna")
COLLECTION_NAME_2 = os.getenv("COLLECTION_NAME_2", "fcc_kdb_docs")

# 명시적으로 CollectorRegistry 생성
registry = CollectorRegistry()

# Counter 및 Histogram 생성 시 registry를 명시적으로 지정
request_count = Counter('chat_requests_total', 'Total chat requests', registry=registry)
response_time = Histogram('response_time_seconds', 'Response time in seconds', registry=registry)
error_count = Counter('chat_errors_total', 'Total chat errors', registry=registry)
memory_usage = Histogram('memory_usage_bytes', 'Memory usage in bytes', registry=registry)

# API 키 검증
api_key_header = APIKeyHeader(name="X-API-Key")

async def verify_api_key(api_key: str = Depends(api_key_header)):
    # API 키 검증 비활성화
    return True

# 메모리 관리 함수
def cleanup_resources():
    """메모리 정리 함수"""
    global model, tokenizer
    try:
        if 'model' in globals() and model is not None:
            del model
        if 'tokenizer' in globals() and tokenizer is not None:
            del tokenizer
        torch.cuda.empty_cache()
        gc.collect()
        logger.info("메모리 정리 완료")
    except Exception as e:
        logger.error(f"메모리 정리 중 오류 발생: {e}")
        logger.error(traceback.format_exc())

# 메모리 사용량 모니터링
def log_memory_usage():
    process = psutil.Process()
    memory_info = process.memory_info()
    memory_usage.observe(memory_info.rss)
    logger.info(f"Memory usage: {memory_info.rss / 1024 / 1024:.2f} MB")

# 로깅 설정
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Hugging Face 로그인 - 토큰이 유효하지 않을 경우 경고만 표시
try:
    if HF_TOKEN and HF_TOKEN != "your_huggingface_token_here":
        login(token=HF_TOKEN)
        logger.info("Hugging Face 로그인 성공")
    else:
        logger.warning("유효한 Hugging Face 토큰이 설정되지 않았습니다. 일부 기능이 제한될 수 있습니다.")
except Exception as e:
    logger.warning(f"Hugging Face 로그인 실패: {e}. 일부 기능이 제한될 수 있습니다.")

# 전역 변수 선언
model = None
tokenizer = None
base_model = None
local_llm = None
summarization_llm = None
is_initialized = False
qdrant_client_instance = None
embedding_model = None
vector_store = None
vector_store_2 = None
langgraph_app = None

# 디바이스 설정
device = "cuda" if torch.cuda.is_available() else "cpu"

# 명시적 확인 추가
if torch.cuda.is_available():
    logger.info(f"CUDA 사용 가능: GPU '{torch.cuda.get_device_name(0)}'")
    device = "cuda:0"  # 명시적 CUDA 장치 지정
else:
    logger.info("CUDA 사용 불가: CPU 모드 사용")
    device = "cpu"

logger.info(f"사용하는 디바이스: {device}")
if device == "cuda":
    logger.info(f"GPU 정보: {torch.cuda.get_device_name(0)}")

# 모델 설정
BASE_MODEL = "meta-llama/Llama-2-13b-chat-hf"
ADAPTER_REPO = LOCAL_ADAPTER_PATH or "Sean-Ong/STA_Reg" # .env 값이 없으면 기본값 사용
OFFLOAD_DIR = "./offload"
MODEL_CACHE_DIR = "./model_cache"

# Pydantic 모델
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 2048
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = False

class DocumentWithScore(BaseModel):
    content: str
    score: float
    metadata: dict

class ChatResponse(BaseModel):
    initial_answer: str
    final_answer: str
    documents: List[DocumentWithScore]
    message_id: str
    rag_draft: Optional[str] = None
    kdb_numbers: Optional[List[str]] = None
    url_links: Optional[Dict[str, str]] = None

class ReferencedKDB(BaseModel):
    kdb_number: str
    title: Optional[str] = None

# GraphState 정의 수정
class GraphState(TypedDict):
    """RAG 및 요약 파이프라인 상태"""
    question: str
    documents: Sequence[DocumentWithScore] # 검색된 문서 (DocumentWithScore 객체 리스트)
    initial_answer: str # RAG 또는 LLM 직접 답변
    final_answer: str # 최종 요약 답변 또는 재질문 유도 메시지
    error: Optional[str]
    should_proceed_to_summary: bool # 요약 단계로 진행할지 여부
    rag_draft: Optional[str] # 초기 RAG 답변
    kdb_numbers: Optional[List[str]] # 사용된 KDB 번호 목록

# 메모리 관리 함수
def cleanup_resources():
    """메모리 정리 함수"""
    global model, tokenizer
    try:
        if 'model' in globals() and model is not None:
            del model
        if 'tokenizer' in globals() and tokenizer is not None:
            del tokenizer
        torch.cuda.empty_cache()
        gc.collect()
        logger.info("메모리 정리 완료")
    except Exception as e:
        logger.error(f"메모리 정리 중 오류 발생: {e}")
        logger.error(traceback.format_exc())

# 캐시된 응답 관리
@lru_cache(maxsize=100)
def get_cached_response(question: str):
    try:
        # 캐시된 응답이 있는지 확인
        if hasattr(get_cached_response, 'cache') and question in get_cached_response.cache:
            return get_cached_response.cache[question]
        return None
    except Exception as e:
        logger.error(f"캐시 조회 중 오류 발생: {e}")
        return None


def check_local_model_files(directory):
    """주어진 디렉토리에 모델 및 토크나이저 필수 파일이 있는지 확인"""
    required_files = ["config.json", "pytorch_model.bin.index.json", # 모델 파일 지표
                      "tokenizer.json", "special_tokens_map.json", "tokenizer_config.json"] # 토크나이저 파일
    if not os.path.isdir(directory):
        return False
    for filename in required_files:
        # pytorch_model.bin.index.json 또는 pytorch_model-*.bin 패턴 확인
        if filename == "pytorch_model.bin.index.json":
             if not glob.glob(os.path.join(directory, "pytorch_model*.bin")) and \
                not os.path.exists(os.path.join(directory, filename)):
                 logger.warning(f"필수 모델 파일 없음: {os.path.join(directory, filename)} 또는 pytorch_model-*.bin")
                 return False
        elif not os.path.exists(os.path.join(directory, filename)):
            logger.warning(f"필수 파일 없음: {os.path.join(directory, filename)}")
            return False
    return True

def initialize_model_and_tokenizer():
    """모델과 토크나이저 초기화 (로컬 우선 확인)"""
    global model, tokenizer, base_model, local_llm, summarization_llm, is_initialized
    
    if is_initialized:
        logger.info("모델이 이미 초기화되어 있습니다.")
        return True
    
    try:
        model_path_to_load = BASE_MODEL # 기본값은 Hugging Face ID
        use_adapter = False
        adapter_path = ADAPTER_REPO # 미리 어댑터 경로 설정
        
        # 1. 로컬 모델 경로 확인 (.env 값 사용)
        logger.info(f"로컬 모델 경로(.env) 확인 중: {LOCAL_BASE_MODEL_PATH}")
        # LOCAL_BASE_MODEL_PATH 가 설정되어 있고, 해당 경로에 파일이 있는지 확인
        if LOCAL_BASE_MODEL_PATH and check_local_model_files(LOCAL_BASE_MODEL_PATH):
            logger.info(f"로컬 모델 파일 확인됨. 로컬 경로 사용: {LOCAL_BASE_MODEL_PATH}")
            model_path_to_load = LOCAL_BASE_MODEL_PATH
            # 로컬 어댑터 경로도 확인 (.env 값 사용)
            if LOCAL_ADAPTER_PATH and os.path.isdir(LOCAL_ADAPTER_PATH):
                logger.info(f"로컬 어댑터 경로 확인됨: {LOCAL_ADAPTER_PATH}")
                use_adapter = True
                adapter_path = LOCAL_ADAPTER_PATH # 사용할 어댑터 경로 확정
            else:
                logger.warning(f"로컬 어댑터 경로 없음/유효하지 않음: {LOCAL_ADAPTER_PATH}. 어댑터 없이 기본 모델만 로드됩니다.")
                use_adapter = False
        else:
            logger.info(f"로컬 모델 파일 확인 실패 또는 경로 미설정. Hugging Face Hub에서 모델 다운로드 시도: {BASE_MODEL}")
            
            # 로컬 모델 디렉토리가 없으면 생성
            if not os.path.exists(LOCAL_BASE_MODEL_PATH):
                os.makedirs(LOCAL_BASE_MODEL_PATH, exist_ok=True)
                logger.info(f"로컬 모델 디렉토리 생성: {LOCAL_BASE_MODEL_PATH}")
            
            # Hugging Face Hub에서 모델 다운로드
            try:
                logger.info(f"Hugging Face Hub에서 모델 다운로드 중: {BASE_MODEL}")
                # 토큰이 유효한 경우에만 토큰 사용
                token_param = {}
                if HF_TOKEN and HF_TOKEN != "your_huggingface_token_here":
                    token_param = {"token": HF_TOKEN}
                
                snapshot_download(
                    repo_id=BASE_MODEL,
                    local_dir=LOCAL_BASE_MODEL_PATH,
                    local_dir_use_symlinks=False,
                    **token_param
                )
                logger.info(f"모델 다운로드 완료: {LOCAL_BASE_MODEL_PATH}")
                model_path_to_load = LOCAL_BASE_MODEL_PATH
            except Exception as download_err:
                logger.error(f"모델 다운로드 실패: {download_err}")
                logger.error(traceback.format_exc())
                return False
            
            # Hugging Face Hub 사용 시 어댑터는 기본값 사용 시도
            adapter_path = "Sean-Ong/STA_Reg"
            logger.info(f"Hugging Face 어댑터 사용 시도: {adapter_path}")
            use_adapter = True # Hub 모델 사용 시 기본적으로 어댑터 로드 시도

        # 양자화 설정 활성화 (수정된 코드)
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )
        
        # 토크나이저 로드
        logger.info(f"토크나이저 로드 시도: {model_path_to_load}")
        # 토큰이 유효한 경우에만 토큰 사용
        token_param = {}
        if HF_TOKEN and HF_TOKEN != "your_huggingface_token_here":
            token_param = {"token": HF_TOKEN}
            
        tokenizer = AutoTokenizer.from_pretrained(
            model_path_to_load,
            use_fast=True,
            padding_side="left",
            **token_param
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        logger.info("토크나이저 로드 완료")
        
        # 기본 모델 로드
        logger.info(f"기본 모델 로드 시도: {model_path_to_load}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path_to_load,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,  # GPU에 적합한 dtype
            low_cpu_mem_usage=True,  # CPU 메모리 사용 최적화
            **token_param
        )
        logger.info("기본 모델 로드 완료")
        
        # 어댑터(LoRA) 적용
        if use_adapter:
            try:
                logger.info(f"어댑터 로드 시도: {adapter_path}")
                # 어댑터 로드 전 모델 장치 확인
                logger.info(f"어댑터 로드 전 모델 장치: {next(model.parameters()).device}")
                
                # 어댑터 로드 시 장치 매핑 추가
                model = PeftModel.from_pretrained(
                    model, 
                    adapter_path,
                    torch_dtype=torch.bfloat16,  # 일관된 dtype 사용
                    device_map={"": device}  # 명시적 장치 매핑
                )
                
                # 어댑터 로드 후 명시적 장치 이동
                model = model.to(device)
                logger.info(f"어댑터 로드 후 모델 장치: {next(model.parameters()).device}")
                logger.info("어댑터(LoRA) 적용 완료")
            except Exception as peft_err:
                logger.warning(f"어댑터 로드 실패: {peft_err}. 기본 모델을 사용합니다.")
                # 기본 모델도 명시적 장치 이동
                model = model.to(device)
        else:
            # 어댑터 미사용 시에도 명시적 장치 이동
            model = model.to(device)
        
        # 모델 평가 모드로 설정
        model.eval()
        
        # 모든 모델 파라미터를 동일 장치로 명시적 이동
        logger.info(f"모델을 {device} 장치로 명시적 이동")
        model = model.to(device)
        
        # 모델 상태 디버깅
        param_device = next(model.parameters()).device
        logger.info(f"모델 파라미터 장치 확인: {param_device}")
        
        # 파이프라인 생성 (수정된 코드)
        text_generation_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=1024,  # 생성 토큰 수 감소
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.15,
            do_sample=True,  # 샘플링 활성화
            use_cache=True    # KV 캐시 활용 활성화
        )
        
        # 파이프라인 장치 확인
        logger.info(f"파이프라인 장치: {text_generation_pipeline.device}")
        
        local_llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
        logger.info(f"LLM 파이프라인 생성 완료 (device: {text_generation_pipeline.device})")
        
        # 요약 파이프라인 (기존과 동일하게 로드 시도)
        try:
            summarizer_pipeline = pipeline("summarization", model=SUMMARIZER_MODEL_NAME, device=device)
            summarization_llm = HuggingFacePipeline(pipeline=summarizer_pipeline)
            logger.info("요약 모델 로드 완료.")
        except Exception as summ_err:
            logger.warning(f"요약 모델 로드 실패: {summ_err}. 요약 기능이 비활성화될 수 있습니다.")
            summarization_llm = None
        
        is_initialized = True
        logger.info("모델 및 토크나이저 초기화 완료")
        return True
        
    except Exception as e:
        logger.error(f"모델 초기화 중 오류 발생: {e}")
        logger.error(traceback.format_exc())
        is_initialized = False
        return False

# Qdrant 클라이언트 초기화
def init_qdrant_client():
    try:
        logger.info(f"Qdrant 클라이언트 초기화 시작 (host: {QDRANT_HOST}, port: {QDRANT_PORT})")
        client = QdrantClient(
            host=QDRANT_HOST,
            port=QDRANT_PORT,
            timeout=10.0  # 타임아웃 설정 추가
        )
        
        # 연결 테스트
        collections = client.get_collections()
        logger.info(f"Qdrant 연결 성공. 사용 가능한 컬렉션: {[c.name for c in collections.collections]}")
        return client
    except Exception as e:
        logger.error(f"Qdrant 클라이언트 초기화 실패: {str(e)}")
        logger.error(traceback.format_exc())
        return None

# 벡터 저장소 초기화 함수 수정
def init_vector_store():
    """Qdrant 벡터 저장소 초기화"""
    global vector_store, vector_store_2, qdrant_client_instance, embedding_model
    
    try:
        # Qdrant 클라이언트 초기화
        qdrant_client_instance = init_qdrant_client()
        if not qdrant_client_instance:
            logger.error("Qdrant 클라이언트 초기화 실패")
            return None, None
        
        # 컬렉션 존재 여부 확인 및 생성
        try:
            collections = qdrant_client_instance.get_collections().collections
            collection_names = [collection.name for collection in collections]
            
            # 첫 번째 컬렉션 확인 및 생성
            if COLLECTION_NAME not in collection_names:
                logger.warning(f"Qdrant 컬렉션 '{COLLECTION_NAME}'이 존재하지 않습니다. 새로 생성합니다.")
                # 임베딩 모델 초기화
                embedding_model = HuggingFaceEmbeddings(
                    model_name=EMBEDDING_MODEL_NAME,
                    model_kwargs={'device': device}
                )
                
                # 컬렉션 생성
                qdrant_client_instance.create_collection(
                    collection_name=COLLECTION_NAME,
                    vectors_config=qdrant_models.VectorParams(
                        size=384,  # MiniLM-L6-v2 임베딩 크기
                        distance=qdrant_models.Distance.COSINE
                    )
                )
                logger.info(f"Qdrant 컬렉션 '{COLLECTION_NAME}' 생성 완료")
            else:
                logger.info(f"Qdrant 컬렉션 '{COLLECTION_NAME}' 확인 완료")
                # 컬렉션 크기 확인
                points_count = qdrant_client_instance.count(COLLECTION_NAME).count
                logger.info(f"컬렉션 '{COLLECTION_NAME}'의 문서 수: {points_count}")
                
            # 두 번째 컬렉션 확인 및 생성
            if COLLECTION_NAME_2 not in collection_names:
                logger.warning(f"Qdrant 컬렉션 '{COLLECTION_NAME_2}'이 존재하지 않습니다. 새로 생성합니다.")
                # 임베딩 모델 초기화 (아직 초기화되지 않은 경우)
                if embedding_model is None:
                    embedding_model = HuggingFaceEmbeddings(
                        model_name=EMBEDDING_MODEL_NAME,
                        model_kwargs={'device': device}
                    )
                
                # 컬렉션 생성
                qdrant_client_instance.create_collection(
                    collection_name=COLLECTION_NAME_2,
                    vectors_config=qdrant_models.VectorParams(
                        size=384,  # MiniLM-L6-v2 임베딩 크기
                        distance=qdrant_models.Distance.COSINE
                    )
                )
                logger.info(f"Qdrant 컬렉션 '{COLLECTION_NAME_2}' 생성 완료")
            else:
                logger.info(f"Qdrant 컬렉션 '{COLLECTION_NAME_2}' 확인 완료")
                # 컬렉션 크기 확인
                points_count = qdrant_client_instance.count(COLLECTION_NAME_2).count
                logger.info(f"컬렉션 '{COLLECTION_NAME_2}'의 문서 수: {points_count}")
                
        except Exception as e:
            logger.error(f"Qdrant 컬렉션 확인/생성 중 오류 발생: {e}")
            logger.error(traceback.format_exc())
            return None, None
        
        # 임베딩 모델 초기화 (아직 초기화되지 않은 경우)
        if embedding_model is None:
            embedding_model = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL_NAME,
                model_kwargs={'device': device}
            )
            logger.info("임베딩 모델 로드 완료")
        
        # 벡터 저장소 초기화 (두 컬렉션 모두 사용)
        vector_store = Qdrant(
            client=qdrant_client_instance,
            collection_name=COLLECTION_NAME,  # 기본 컬렉션
            embeddings=embedding_model
        )
        
        # 두 번째 벡터 저장소 초기화
        vector_store_2 = Qdrant(
            client=qdrant_client_instance,
            collection_name=COLLECTION_NAME_2,  # 두 번째 컬렉션
            embeddings=embedding_model
        )
        
        logger.info("벡터 저장소 초기화 완료 (두 컬렉션)")
        return vector_store, vector_store_2
        
    except Exception as e:
        logger.error(f"벡터 저장소 초기화 중 오류 발생: {e}")
        logger.error(traceback.format_exc())
        return None, None

def retrieve_node(question: str) -> List[Tuple[Document, float]]:
    """주어진 질문에 대해 관련 문서를 검색하고 중복 제거, 유사도 점수 포함"""
    try:
        logger.info(f"문서 검색 시작: {question}")
        
        # 1. 첫 번째 컬렉션에서 검색
        try:
            results_1 = vector_store.similarity_search_with_score(
                question,
                k=5  # 상위 5개 문서 검색
            )
            logger.info(f"첫 번째 컬렉션 검색 결과: {len(results_1)}개 문서")
            
            # Qdrant에서 가져온 페이로드 필드를 제대로 처리
            for i, (doc, score) in enumerate(results_1):
                if not hasattr(doc, 'metadata') or doc.metadata is None:
                    doc.metadata = {}
                doc.metadata['score'] = score
                # 필요하다면 content 필드를 page_content로 복사
                if not doc.page_content and 'content' in doc.metadata:
                    doc.page_content = doc.metadata['content']
            
        except Exception as e:
            logger.error(f"첫 번째 컬렉션 검색 중 오류 발생: {e}")
            logger.error(traceback.format_exc())
            results_1 = []
        
        # 2. 두 번째 컬렉션에서 검색
        try:
            # 네이티브 Qdrant 클라이언트를 사용하여 직접 검색
            search_vector = embedding_model.embed_query(question)
            search_results = qdrant_client_instance.search(
                collection_name=COLLECTION_NAME_2,
                query_vector=search_vector,
                limit=5,
                with_payload=True
            )
            
            # 검색 결과를 Document 객체로 변환
            results_2 = []
            for hit in search_results:
                if hit.payload:
                    # 페이로드에서 콘텐츠 필드 이름 확인 (page_content 또는 content 가능)
                    content_field = 'content'
                    if 'page_content' in hit.payload:
                        content_field = 'page_content'
                    
                    content = hit.payload.get(content_field)
                    if content:
                        # 메타데이터는 페이로드 전체를 포함
                        metadata = hit.payload.copy()
                        # 콘텐츠 필드 제거 (중복 방지)
                        if content_field in metadata:
                            del metadata[content_field]
                        # 유사도 점수 추가
                        metadata['score'] = hit.score
                        doc = Document(page_content=content, metadata=metadata)
                        results_2.append((doc, hit.score))
            
            logger.info(f"두 번째 컬렉션 검색 결과: {len(results_2)}개 문서")
        except Exception as e:
            logger.error(f"두 번째 컬렉션 검색 중 오류 발생: {e}")
            logger.error(traceback.format_exc())
            results_2 = []
        
        # 3. 결과 병합 및 중복 제거 (KDB 번호 기준)
        collection2_kdb_numbers = set()
        unique_results = []
        
        # 먼저 컬렉션2에서 가져온 KDB 번호들 수집
        for doc, score in results_2:
            try:
                # 여기서 페이로드의 상위 레벨에 저장된 kdb_number에 접근
                kdb_number = doc.metadata.get('kdb_number')
                if kdb_number:
                    if isinstance(kdb_number, list):
                        collection2_kdb_numbers.update(kdb_number)
                    else:
                        collection2_kdb_numbers.add(kdb_number)
                
                # 유효한 문서는 바로 결과에 추가
                logger.debug(f"문서 내용 확인(컬렉션2): {doc.page_content[:100]}... (길이: {len(doc.page_content)})")
                unique_results.append((doc, score))
            except Exception as doc_err:
                logger.error(f"문서 처리 중 오류 발생: {doc_err}")
                continue
        
        # 컬렉션1에서 가져온 문서 중 컬렉션2에 없는 KDB 번호만 추가
        for doc, score in results_1:
            try:
                # 여기서 페이로드의 상위 레벨에 저장된 kdb_number에 접근
                kdb_number = doc.metadata.get('kdb_number')
                # KDB 번호가 없거나 컬렉션2에 이미 있는 경우 제외
                if not kdb_number or kdb_number in collection2_kdb_numbers:
                    continue
                
                # 메타데이터에 유사도 점수 추가
                doc.metadata['score'] = score
                
                content = doc.page_content
                if content:
                    logger.debug(f"문서 내용 확인(컬렉션1): {content[:100]}... (길이: {len(content)})")
                    unique_results.append((doc, score))
            except Exception as doc_err:
                logger.error(f"문서 처리 중 오류 발생: {doc_err}")
                continue
        
        logger.info(f"최종 검색 결과: {len(unique_results)}개 고유 문서")
        return unique_results
        
    except Exception as e:
        logger.error(f"문서 검색 중 오류 발생: {e}")
        logger.error(traceback.format_exc())
        return []

async def generate_rag_answer_node(question: str, documents: List[Document]) -> str:
    """문서를 일괄 처리하여 RAG 답변 생성"""
    try:
        logger.info("초기 RAG 답변 생성 시작")
        
        # 문서가 없는 경우 기존 코드와 동일하게 처리
        if not documents:
            # ... 기존 코드 유지 ...
        
        # 일괄 처리를 위한 문서 설정
            logger.info(f"총 {len(documents)}개 문서 일괄 처리 시작")
        
        # 모든 문서를 하나의 프롬프트로 결합
        combined_prompt = f"""<s>[INST] <<SYS>>
You are an expert in RF equipment certification. Your task is to answer the user's question based on the provided documents.
For each document, provide a separate answer with the following structure:

1. Answer: A detailed explanation based on the document's content
2. Summary: 2-3 sentences summarizing key points
3. Related KDB: The KDB number with a brief description

The user's question is: {question}

Here are the documents:
</</SYS>>

"""
        # 문서 내용 최적화를 위한 최대 토큰 수 설정
        max_tokens_per_doc = 500  # 각 문서당 최대 토큰 수 제한
        
        # 각 문서 내용 추가
        for i, doc in enumerate(documents):
            # 문서 내용 길이 제한
            content = doc.page_content
            # 내용이 너무 길면 자르기
            if len(content) > max_tokens_per_doc * 4:  # 대략 1 토큰 = 4 글자로 가정
                content = content[:max_tokens_per_doc * 4] + "..."
            
            # 문서 메타데이터 추출
            kdb_number = doc.metadata.get('kdb_number', 'N/A')
            title = doc.metadata.get('title', '')
            
            # 프롬프트에 문서 추가
            combined_prompt += f"""
## Document {i+1}:
KDB Number: {kdb_number}
Title: {title}
Content: {content}

"""
        
        # 프롬프트 완성
        combined_prompt += """
Please analyze each document separately and provide an answer for each document following the format:

=== Document 1 ===
1. Answer: [Your detailed answer based on document 1]
2. Summary: [2-3 sentence summary]
3. Related KDB: [KDB number with brief description]

=== Document 2 ===
...and so on for each document.

[/INST]"""
        
        # 한 번의 LLM 호출로 모든 문서에 대한 답변 생성
        try:
            logger.info("LLM을 사용하여 일괄 답변 생성 시작")
            response = await local_llm.ainvoke(combined_prompt)
            combined_answer = response.content if hasattr(response, 'content') else str(response)
            
            # 프롬프트 제거
            combined_answer = clean_prompt_from_response(combined_answer, combined_prompt)
            
            # 최종 답변 형식 조정
            formatted_answer = f"질문: {question}\n\n{combined_answer}"
            
            logger.info("일괄 RAG 답변 생성 완료")
            return formatted_answer
            
        except Exception as e:
            # 오류 발생 시 기존 방식으로 폴백 (기존 코드 활용)
            logger.warning("일괄 처리 실패. 개별 문서 처리 방식으로 폴백합니다.")
            # ... 기존 코드 실행 ...
    
    except Exception as e:
        # ... 기존 예외 처리 ...
        logger.error(f"일괄 답변 생성 중 오류 발생: {e}")

async def generate_final_answer_node(question: str, initial_answer: str, documents: List[Document]) -> str:
    """문서별 초기 RAG 답변을 기반으로 최종 답변 생성"""
    try:
        logger.info("최종 답변 생성 시작")
        
        # 문서가 없는 경우 처리
        if not documents:
            logger.warning("검색된 문서가 없습니다.")
            return initial_answer
        
        # 프롬프트 생성
        prompt = f"""<s>[INST] <<SYS>>
You are a certification expert specializing in RF regulations. Your task is to provide a comprehensive and detailed answer based on the provided information.

Guidelines for answering:
1. Start with a clear definition and explanation
2. Include all relevant technical specifications
3. Explain regulatory requirements in detail
4. Provide practical implications and applications
5. Use specific examples when possible
6. Ensure technical accuracy while being clear and understandable
7. If there are conflicting information between documents, explain the differences
8. Prioritize information from documents with higher relevance scores
9. Keep your answer concise but complete - do not cut off mid-sentence
10. If you need to summarize, do so at the end of your answer
        
<</SYS>>
        
Based on the following document-specific answers, please provide a comprehensive answer:

{initial_answer}

Question: {question}

Provide a detailed answer that covers all aspects of the question. Make sure to complete all sentences and thoughts:
[/INST]"""
        
        # LLM으로 답변 생성
        try:
            response = await local_llm.ainvoke(prompt)
            final_answer = response.content if hasattr(response, 'content') else str(response)
            
            # 프롬프트 제거 로직 추가
            final_answer = clean_prompt_from_response(final_answer, prompt)
            
            logger.info("최종 답변 생성 완료")
            return final_answer
        except Exception as e:
            logger.error(f"최종 답변 생성 중 오류 발생: {e}")
            logger.error(traceback.format_exc())
            return initial_answer  # 오류 발생 시 초기 답변 반환
        
    except Exception as e:
        logger.error(f"최종 답변 생성 중 오류 발생: {e}")
        logger.error(traceback.format_exc())
        return initial_answer

async def summarize_answer_node(answer: str) -> str:
    """생성된 답변을 요약"""
    try:
        logger.info("답변 요약 시작")
        
        if not summarization_llm:
            logger.warning("요약 모델이 초기화되지 않았습니다.")
            return answer
        
        # 답변이 너무 짧으면 그대로 반환
        if len(answer) < 200:
            logger.info("답변이 너무 짧아 요약이 필요하지 않습니다.")
            return answer
        
        # 프롬프트 생성
        prompt = f"""<s>[INST] <<SYS>>
You are an expert assistant specializing in FCC radio frequency (RF) equipment certification.
Your task is to summarize technical answers retrieved from FCC official documentation clearly and accurately.

Guidelines:
- Summarize the content into clear sections.
- Do not invent or assume any information not included in the original answer.
- Maintain the technical tone suitable for engineers and regulatory professionals.
- Prefer structured formats such as bullet points, numbered lists, or short paragraphs.
- Use headers if appropriate to separate key topics.

When possible, include:
- Key certification rules
- Referenced KDB documents and their purpose
- Measurement/test conditions or requirements
- Regulatory caveats or exceptions

<</SYS>>

Here is the full technical answer retrieved from the documents:

{answer}

Now, provide a structured and concise summary based on the above answer.

[Summary Start]
[/INST]"""
        
        # 요약 생성 - 비동기 호출로 변경
        try:
            response = await local_llm.ainvoke(prompt)  # invoke -> ainvoke
            summary = response.content if hasattr(response, 'content') else str(response)
            
            # 프롬프트 제거 로직 추가
            summary = clean_prompt_from_response(summary, prompt)
            
            logger.info("답변 요약 완료")
            return summary
        except Exception as e:
            logger.error(f"요약 생성 중 오류 발생: {e}")
            logger.error(traceback.format_exc())
            return answer
            
    except Exception as e:
        logger.error(f"답변 요약 중 오류 발생: {e}")
        logger.error(traceback.format_exc())
        return answer

# 프롬프트 제거 헬퍼 함수 추가
def clean_prompt_from_response(response: str, prompt: str) -> str:
    """응답에서 프롬프트 부분을 제거하는 함수"""
    try:
        # 프롬프트가 응답에 포함된 경우 제거
        if prompt in response:
            cleaned = response.replace(prompt, "").strip()
            return cleaned
            
        # 출력에서 [INST] 태그 및 시스템 지시어 제거 (정규식 접근)
        import re
        # [INST] 태그 및 내용 제거
        cleaned = re.sub(r'<s>\s*\[INST\].*?(\[/INST\])', '', response, flags=re.DOTALL)
        # <<SYS>> 태그 및 내용 제거
        cleaned = re.sub(r'<<SYS>>.*?<</SYS>>', '', cleaned, flags=re.DOTALL)
        # [Summary Start] 등 마커 제거
        cleaned = re.sub(r'\[Summary Start\]', '', cleaned, flags=re.DOTALL)
        # 남은 태그 제거
        cleaned = re.sub(r'</?s>', '', cleaned)
        cleaned = re.sub(r'\[/INST\]', '', cleaned)
        
        # 여러 줄 공백 제거
        cleaned = re.sub(r'\n\s*\n', '\n\n', cleaned)
        
        return cleaned.strip()
    except Exception as e:
        logger.error(f"응답 정리 중 오류 발생: {e}")
        logger.error(traceback.format_exc())
        # 오류 발생 시 원본 응답 반환
        return response

def extract_kdb_numbers(text: str = None, documents: List[Document] = None) -> List[str]:
    """Qdrant 컬렉션에서 KDB 번호를 직접 추출하거나 텍스트에서 KDB 번호를 추출하는 함수"""
    try:
        kdb_numbers = []
        
        # 문서에서 직접 추출 (문서가 제공된 경우)
        if documents:
            logger.info("문서에서 직접 KDB 번호 추출")
            for doc in documents:
                # 1. 페이로드 상위 레벨의 kdb_number를 가져오려면 metadata 전체를 확인
                metadata = doc.metadata
                if not metadata:
                    continue
                    
                kdb_number = metadata.get('kdb_number')
                if kdb_number:
                    if isinstance(kdb_number, list):
                        kdb_numbers.extend([str(k) for k in kdb_number])
                    else:
                        kdb_numbers.append(str(kdb_number))
            
            # 중복 제거 및 정렬
            kdb_numbers = sorted(list(set(kdb_numbers)))
            logger.info(f"문서에서 추출된 KDB 번호: {kdb_numbers}")
            return kdb_numbers
        
        # 텍스트에서 추출 (텍스트가 제공된 경우)
        if text:
            # KDB 번호 패턴: 숫자로 시작하는 6자리 또는 7자리 숫자
            kdb_pattern = r'\b\d{6,7}\b'
            extracted_numbers = re.findall(kdb_pattern, text)
            
            # 중복 제거 및 정렬
            kdb_numbers = sorted(list(set(extracted_numbers)))
            logger.info(f"텍스트에서 추출된 KDB 번호: {kdb_numbers}")
        
        return kdb_numbers
    except Exception as e:
        logger.error(f"KDB 번호 추출 중 오류 발생: {e}")
        logger.error(traceback.format_exc())
        return []

@app.get("/health")
async def health_check():
    """서버 상태 확인 엔드포인트"""
    try:
        # 모델 초기화 상태 확인
        model_status = "initialized" if is_initialized else "not initialized"
        
        # Qdrant 연결 상태 확인
        qdrant_status = "connected" if qdrant_client_instance else "not connected"
        
        return {
            "status": "healthy",
            "model_status": model_status,
            "qdrant_status": qdrant_status,
            "device": device
        }
    except Exception as e:
        logger.error(f"Health check 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    """채팅 API 엔드포인트"""
    try:
        # 요청 카운터 증가
        request_count.inc()
        
        # 메시지 ID 생성
        message_id = str(uuid.uuid4())
        
        # 사용자 질문 추출
        user_message = next((msg.content for msg in request.messages if msg.role == "user"), None)
        if not user_message:
            raise HTTPException(status_code=400, detail="사용자 메시지를 찾을 수 없습니다.")
        
        # 관련 문서 검색 (유사도 점수 포함)
        retrieved_docs_with_scores = retrieve_node(user_message)
        documents = [doc for doc, score in retrieved_docs_with_scores]
        
        # 비동기 호출로 변경
        initial_answer = await generate_rag_answer_node(user_message, documents)
        final_answer = await generate_final_answer_node(user_message, initial_answer, documents)
        summary = await summarize_answer_node(final_answer)
        
        # KDB 번호 추출 (문서에서 직접 추출)
        kdb_numbers = extract_kdb_numbers(documents=documents)
        if not kdb_numbers:
            # 문서에서 추출 실패한 경우 텍스트에서 추출 시도
            kdb_numbers = extract_kdb_numbers(text=initial_answer)
        
        # URL 링크 생성
        url_links = {kdb: f"{URL_LINK}/{kdb}.pdf" for kdb in kdb_numbers} if kdb_numbers else {}
        
        # 응답 생성 (유사도 점수 포함)
        documents_with_scores = []
        for doc, score in retrieved_docs_with_scores:
            # 유사도 점수가 메타데이터에 있을 수 있으므로 확인
            doc_score = doc.metadata.get('score', score)
            documents_with_scores.append(
                DocumentWithScore(
                    content=doc.page_content, 
                    score=doc_score, 
                    metadata=doc.metadata
                )
            )
        
        # 응답 구조 수정: 초기 RAG 답변이 상세 답변으로, 최종 답변이 final_answer로
        response = ChatResponse(
            message_id=message_id,
            initial_answer=initial_answer,  # 초기 RAG 답변 (문서별 답변들)
            final_answer=final_answer,     # 최종 RAG 답변
            documents=documents_with_scores,
            rag_draft=summary,            # 요약된 답변
            kdb_numbers=kdb_numbers,
            url_links=url_links
        )
        
        return response
        
    except Exception as e:
        # 에러 카운터 증가
        error_count.inc()
        logger.error(f"Chat endpoint error: {e}")
        logger.error(traceback.format_exc()) 
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
async def startup_event():
    """서버 시작 시 실행되는 초기화 함수"""
    try:
        # 모델 초기화
        if not initialize_model_and_tokenizer():
            logger.error("모델 초기화 실패")
            return
            
        # 벡터 저장소 초기화
        global vector_store, vector_store_2
        vector_store, vector_store_2 = init_vector_store()
        if not vector_store or not vector_store_2:
            logger.error("벡터 저장소 초기화 실패")
            return
            
        logger.info("서버 초기화 완료")
    except Exception as e:
        logger.error(f"서버 초기화 중 오류 발생: {e}")
        logger.error(traceback.format_exc()) 