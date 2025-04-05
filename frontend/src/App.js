import React, { useState, useEffect, useRef } from 'react';
import './App.css';
import axios from 'axios';

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [backendStatus, setBackendStatus] = useState('연결 중...');
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    const checkBackendStatus = async () => {
      try {
        const response = await axios.get('http://localhost:8000/health');
        setBackendStatus(`연결됨 (${response.data.device}) - 
          모델: ${response.data.model_status}, 
          Qdrant: ${response.data.qdrant_status}`);
      } catch (error) {
        setBackendStatus('연결 실패');
        console.error('Backend connection error:', error);
      }
    };

    checkBackendStatus();
    const interval = setInterval(checkBackendStatus, 30000);
    return () => clearInterval(interval);
  }, []);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;

    const userMessage = { role: 'user', content: input };
    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const response = await axios.post('http://localhost:8000/api/chat', {
        messages: [userMessage],
      });

      const { initial_answer, final_answer, documents } = response.data;
      
      // 응답 데이터 구조화
      const assistantMessage = {
        role: 'assistant',
        content: final_answer,
        rag_answer: initial_answer,
        referenced_kdbs: documents.map(doc => ({
          kdb_number: doc.metadata.kdb_number,
          title: doc.metadata.title || '제목 없음',
          score: doc.score
        }))
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (error) {
      console.error('Error sending message:', error);
      setMessages((prev) => [...prev, { 
        role: 'assistant', 
        content: '죄송합니다. 응답을 생성하는 중에 오류가 발생했습니다.' 
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  // RGB 버튼 색상 결정
  const getStatusColor = () => {
    if (backendStatus.includes('연결됨')) return 'green';
    if (backendStatus.includes('연결 실패')) return 'red';
    return 'yellow'; // 기본값: 연결 중...
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>챗봇</h1>
        <div className="status-indicator" style={{ backgroundColor: getStatusColor() }}>
          {backendStatus}
        </div>
      </header>
      <div className="chat-container">
        <div className="messages">
          {messages.map((message, index) => (
            <div key={index} className={`message ${message.role}`}>
              {message.role === 'assistant' ? (
                <>
                  {/* 상세 답변 (RAG) */}
                  {message.rag_answer && message.rag_answer.trim() && (
                    <div className="original-answer">
                      <h4>상세 답변 (RAG):</h4>
                      <div className="answer-content">{message.rag_answer}</div>
                    </div>
                  )}

                  {/* 요약 답변 */}
                  {message.content && message.content.trim() && (
                    <div className="summary-answer">
                      <h4>요약:</h4>
                      <div className="answer-content">{message.content}</div>
                    </div>
                  )}

                  {/* 참고 KDB 목록 */}
                  {message.referenced_kdbs && message.referenced_kdbs.length > 0 && (
                    <div className="referenced-kdb-section">
                      <h4>참고 KDB:</h4>
                      <ul>
                        {message.referenced_kdbs.map((kdb, kdbIndex) => (
                          <li key={kdbIndex}>
                            <strong>{kdb.kdb_number}:</strong> {kdb.title}
                            <span className="similarity-score">(유사도: {(kdb.score * 100).toFixed(1)}%)</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </>
              ) : (
                <div className="message-content">{message.content}</div>
              )}
            </div>
          ))}
          <div ref={messagesEndRef} />
        </div>
        <form onSubmit={handleSubmit} className="input-form">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="질문을 입력하세요..."
            disabled={isLoading}
          />
          <button type="submit" disabled={isLoading}>전송</button>
        </form>
      </div>
    </div>
  );
}

export default App;

