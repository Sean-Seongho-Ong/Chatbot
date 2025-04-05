# RF 규제 질의응답 챗봇

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

RF 규제 관련 문서를 기반으로 사용자의 질문에 답변하는 RAG(Retrieval-Augmented Generation) 챗봇입니다.

## 🚀 주요 기능

*   **문서 기반 답변:** RF 규제 문서 벡터 데이터베이스에서 관련 정보를 검색하여 답변 생성
*   **RAG 파이프라인:** LangChain과 LangGraph를 활용하여 효율적인 정보 검색 및 답변 생성
*   **LLM 활용:** Llama-2-13b-chat-hf 모델 (QLoRA 파인튜닝) 기반 답변 생성 및 요약
*   **실시간 상태 확인:** 웹 인터페이스에서 백엔드 서버 연결 상태 표시
*   **모니터링:** Prometheus를 이용한 서버 메트릭 수집

## 🛠️ 기술 스택

*   **Backend:** Python, FastAPI
*   **Frontend:** React.js
*   **LLM:** Llama-2-13b-chat-hf (with QLoRA)
*   **RAG Framework:** LangChain, LangGraph
*   **Vector Store:** Qdrant
*   **Embedding:** `sentence-transformers/all-MiniLM-L6-v2`
*   **Monitoring:** Prometheus

## 📂 프로젝트 구조

```
. (루트)
├── backend/             # 백엔드 FastAPI 애플리케이션
│   ├── main.py          # FastAPI 앱 진입점
│   └── ...              # 기타 백엔드 관련 파일
├── frontend/            # 프론트엔드 React 애플리케이션
│   ├── public/
│   ├── src/
│   │   ├── App.js       # 메인 React 컴포넌트
│   │   └── ...          # 기타 프론트엔드 관련 파일
│   └── package.json
├── .env                 # 환경 변수 설정 파일 (로컬에서 생성)
├── .env.example         # 환경 변수 예시 파일
├── .gitignore           # Git 추적 제외 파일 목록
├── requirements.txt     # Python 의존성 목록
├── run_chatbot.bat      # 애플리케이션 실행 스크립트 (Windows)
└── README.md            # 프로젝트 설명 파일 (현재 파일)
```

## ⚙️ 설치 및 실행

**요구 사항:**

*   Python (3.8 이상 권장)
*   Node.js 및 npm (프론트엔드 실행용)
*   Qdrant (벡터 데이터베이스)
*   CUDA 지원 GPU (LLM 가속용)

**설치:**

1. **저장소 복제:**
   ```bash
   git clone https://github.com/Sean-Seongho-Ong/Chatbot.git
   cd /c/Regluatory\ SaaS/Platform\ SaaS/ChatBot_interface_Web/New_version
   ```
   또는 Windows CMD/PowerShell의 경우:
   ```powershell
   git clone https://github.com/Sean-Seongho-Ong/Chatbot.git
   cd "C:\Regluatory SaaS\Platform SaaS\ChatBot_interface_Web\New_version"
   ```
2. **Python 의존성 설치:**
    ```bash
    pip install -r requirements.txt
    ```
3. **프론트엔드 의존성 설치:**
    ```bash
    cd frontend
    npm install
    cd ..
    ```

**실행:**

1. **백엔드 서버 시작:**
    ```bash
    cd backend
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
    ```

2.  **프론트엔드 서버 시작:** (새 터미널에서)
    ```bash
    cd frontend
    npm start
    ```

애플리케이션이 시작되면 웹 브라우저에서 `http://localhost:3000`으로 접속합니다.

## 📚 API 문서

백엔드 서버 실행 후 다음 URL에서 API 문서를 확인할 수 있습니다:

*   Swagger UI: `http://localhost:8000/docs`
*   ReDoc: `http://localhost:8000/redoc`

## 📊 모니터링

Prometheus 메트릭은 `http://localhost:9090` (또는 설정된 주소)에서 확인할 수 있습니다.

## 📄 라이선스

이 프로젝트는 [MIT 라이선스](LICENSE) 하에 배포됩니다. 자세한 내용은 `LICENSE` 파일을 참고하세요. 