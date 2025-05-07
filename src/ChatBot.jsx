import React, { useState, useEffect, useRef } from "react";
import "./Chatbot.css";
import axios from 'axios';

const ChatMessage = ({ sender, message }) => (
  <div className={`message ${sender}`}>
    <div className="message-bubble">{message}</div>
  </div>
);

export default function AIChatbotUI() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const messagesEndRef = useRef(null);

  const handleSend = async () => {
    if (!input.trim()) return;
    const newMessage = { sender: "user", message: input };
    setMessages((prev) => [...prev, newMessage]);
    setInput("");

    try {
      const response = await axios.post("http://127.0.0.1:8000/chat", { message: input });
      console.log(response);
      const aiReply = { sender: "ai", message: response.data.message};
      setMessages(response.data);
    } catch (error) {
      console.error("Error:", error);
      setMessages((prev) => [...prev, { sender: "ai", message: "⚠️ Error fetching response." }]);
    }
  };

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  return (
    <div className="chat-container">
      <div className="chat-box">
        <div className="messages-area">
          {messages.map((msg, idx) => (
            <ChatMessage key={idx} sender={msg.sender} message={msg.message} />
          ))}
          <div ref={messagesEndRef} />
        </div>
        <div className="input-area">
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleSend()}
            placeholder="Ask something..."
          />
          <button onClick={handleSend}>Send</button>
        </div>
      </div>
    </div>
  );
}
