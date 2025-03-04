import React from 'react';
import { ChatBox } from '../../components/chatbox/chatbox';
import './chatscreen.styles.css';

function ChatScreen() {
    const handleSendMessage = (msg: string) => {
        // Handle sending the message
        console.log("User sent:", msg);
    };

    return (
        <div className="chat-container">
            <div className="chat-header">What can I help with?</div>
            <div className="chat-messages">
                {/* Render chat messages here */}
            </div>
            <ChatBox holder={"Ask anything"} onSend={handleSendMessage} />
        </div>
    );
}

export default ChatScreen;
