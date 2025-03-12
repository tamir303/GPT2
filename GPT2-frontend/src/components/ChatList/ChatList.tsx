import React, { useRef, useEffect } from 'react';
import { List, ListItem, Box, Typography } from '@mui/material';
import { useSelector } from 'react-redux';
import { chatListStyles, messageStyles } from './ChatList.styles';
import { ChatState } from "../../actions/actions";

const ChatList: React.FC = () => {
    const messages = useSelector((state: ChatState) => state.messages);
    const listRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        if (listRef.current) {
            listRef.current.scrollTop = listRef.current.scrollHeight;
        }
    }, [messages]);

    return (
        <Box ref={listRef} sx={chatListStyles}>
            <List>
                {messages.map((message, index) => (
                    <ListItem
                        key={index}
                        sx={{
                            justifyContent: message.sender === 'user' ? 'flex-end' : 'flex-start',
                        }}
                    >
                        <Box sx={messageStyles(message.sender)}>
                            <Typography>{message.text}</Typography>
                        </Box>
                    </ListItem>
                ))}
            </List>
        </Box>
    );
};

export default ChatList;