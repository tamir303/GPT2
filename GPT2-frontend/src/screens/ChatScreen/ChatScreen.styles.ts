import { SxProps, Theme } from '@mui/material';

export const screenStyles: SxProps<Theme> = {
    backgroundColor: '#303030',
    display: 'flex',
    flexDirection: 'column',
    height: '100vh',
'& header': {
    backgroundColor: '#202020',
    color: 'white',
    padding: '1rem',
    textAlign: 'center',
},
};