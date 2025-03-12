import React from 'react';
import { Button, SxProps } from '@mui/material';
import { buttonsStyles } from './Buttons.styles';

interface ButtonsProps {
    text: string;
    icon: React.ReactNode;
    onClick: () => void;
    disabled?: boolean;
    sx?: SxProps;
}

const Buttons: React.FC<ButtonsProps> = ({ text, icon, onClick, disabled = false, sx }) => {
    const completeSx = { ...buttonsStyles, ...sx }
    return (
        <Button
            variant="contained"
            startIcon={icon}
            onClick={onClick}
            disabled={disabled}
            sx={completeSx}
        >
            {text}
        </Button>
    );
};

export default Buttons;