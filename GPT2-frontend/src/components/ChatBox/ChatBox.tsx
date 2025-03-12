import React from 'react';
import { Box } from '@mui/material';
import ChatList from '../ChatList/ChatList';
import InputField from '../InputField/InputField';
import { chatBoxStyles } from './ChatBox.styles';

const ChatBox: React.FC = () => {
    return (
        <Box sx={chatBoxStyles}>
            <ChatList />
            <InputField />
        </Box>
    );
};

export default ChatBox;