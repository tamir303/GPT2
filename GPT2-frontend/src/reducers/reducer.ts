import { ADD_USER_MESSAGE, ADD_AI_MESSAGE, ChatState } from '../actions/actions';

const initialState: ChatState = {
    messages: [{ sender: 'ai', text: 'Hello! How can I assist you today?' }],
};

const chatReducer = (state = initialState, action: any): ChatState => {
    switch (action.type) {
        case ADD_USER_MESSAGE:
        case ADD_AI_MESSAGE:
            return {
                ...state,
                messages: [...state.messages, action.payload],
            };
        default:
            return state;
    }
};

export default chatReducer;