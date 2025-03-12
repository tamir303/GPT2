export const ADD_USER_MESSAGE = 'ADD_USER_MESSAGE';
export const ADD_AI_MESSAGE = 'ADD_AI_MESSAGE';

interface Message {
    sender: 'user' | 'ai';
    text: string;
}

export interface ChatState {
    messages: Message[];
}

export const addUserMessage = (text: string) => ({
    type: ADD_USER_MESSAGE,
    payload: { sender: 'user', text },
});

export const addAiMessage = (text: string) => ({
    type: ADD_AI_MESSAGE,
    payload: { sender: 'ai', text },
});