import { SxProps, Theme } from '@mui/material';

export const chatListStyles: SxProps<Theme> = {
    flex: 1,
    overflowY: 'auto',
    padding: '1rem',
};

export const messageStyles = (sender: 'user' | 'ai'): SxProps<Theme> => ({
    backgroundColor: sender === 'user' ? '#1976d2' : '#e0e0e0',
    color: sender === 'user' ? 'white' : 'black',
    borderRadius: '8px',
    padding: '0.5rem 1rem',
    maxWidth: '70%',
});