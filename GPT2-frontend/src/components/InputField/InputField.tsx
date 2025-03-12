import React, { useState } from 'react';
import { Box, TextField } from '@mui/material';
import SendIcon from '@mui/icons-material/Send';
import Buttons from '../Buttons/Buttons';
import { addUserMessage } from '../../actions/actions';
import { inputFieldStyles } from './InputField.styles';
import { useDispatch } from "react-redux";

const InputField: React.FC = () => {
    const [input, setInput] = useState<string>('');
    const dispatch = useDispatch();

    const handleSend = () => {
        if (input.trim() === '') return;
        dispatch(addUserMessage(input));
        setInput('');
    };

    return (
        <Box sx={inputFieldStyles}>
            <TextField
                fullWidth
                variant="outlined"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Type your message..."
            />
            <Buttons
                text="Send"
                icon={<SendIcon />}
                onClick={handleSend}
                disabled={input.trim() === ''}
                sx={{ color: 'black', backgroundColor: 'white' }}
            />
        </Box>
    );
};

export default InputField;