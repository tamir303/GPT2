import React from 'react';
import { Box } from '@mui/material';
import ChatBox from '../../components/ChatBox/ChatBox'
import {screenStyles} from "./ChatScreen.styles";

const Screen: React.FC = () => {
    return (
        <Box sx={screenStyles}>
            <Box component="header">
                <h1>ChatGPT Clone</h1>
            </Box>
            <ChatBox />
        </Box>
    );
};

export default Screen;