import { createStore } from 'redux';
import chatReducer from '../reducers/reducer';
import { persistStore, persistReducer } from 'redux-persist';
import storage from 'redux-persist/lib/storage';

const persistConfig = {
    key: 'root',
    storage,
};

const persistedReducer = persistReducer(persistConfig, chatReducer);
const store = createStore(persistedReducer);
const persistor = persistStore(store);

export { store, persistor };