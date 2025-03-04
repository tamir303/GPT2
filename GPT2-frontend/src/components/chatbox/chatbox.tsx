import React, { useState } from "react";
import styles from "./chatbox.module.css"

interface ChatBoxProps {
    holder: string;
    onSend: (message: string) => void;
    children?: React.ReactNode;
}

export const ChatBox: React.FC<ChatBoxProps> = ({ holder, onSend ,children }) => {
    const [userMessage, setUserMessage] = useState<string>('')

    const handleSendMessage = () => {
        if (userMessage.trim()) {
            onSend(userMessage)
            setUserMessage('')
        }
    };

    return (
        <div className={styles.chatInput}>
            <textarea
                placeholder={holder}
                value={userMessage}
                onChange={e => setUserMessage(e.target.value)}
                rows={5}
            >
            </textarea>
        </div>
    )
}