import React, { createContext, useContext, useEffect, useState } from 'react';
import io from 'socket.io-client';

const SocketContext = createContext();

export const useSocket = () => {
  const context = useContext(SocketContext);
  if (!context) {
    throw new Error('useSocket must be used within a SocketProvider');
  }
  return context;
};

export const SocketProvider = ({ children }) => {
  const [socket, setSocket] = useState(null);
  const [connected, setConnected] = useState(false);
  const [evolutionData, setEvolutionData] = useState({});
  const [systemMetrics, setSystemMetrics] = useState({});

  useEffect(() => {
    const newSocket = io(window.location.origin, {
      transports: ['websocket', 'polling']
    });

    newSocket.on('connect', () => {
      console.log('Connected to DGM server');
      setConnected(true);
    });

    newSocket.on('disconnect', () => {
      console.log('Disconnected from DGM server');
      setConnected(false);
    });

    newSocket.on('evolution-update', (data) => {
      setEvolutionData(prev => ({
        ...prev,
        [data.sessionId]: data
      }));
    });

    newSocket.on('system-metrics', (metrics) => {
      setSystemMetrics(metrics);
    });

    newSocket.on('error', (error) => {
      console.error('Socket error:', error);
    });

    setSocket(newSocket);

    return () => {
      newSocket.close();
    };
  }, []);

  const subscribeToSession = (sessionId) => {
    if (socket) {
      socket.emit('subscribe-session', sessionId);
    }
  };

  const unsubscribeFromSession = (sessionId) => {
    if (socket) {
      socket.emit('unsubscribe-session', sessionId);
    }
  };

  const value = {
    socket,
    connected,
    evolutionData,
    systemMetrics,
    subscribeToSession,
    unsubscribeFromSession
  };

  return (
    <SocketContext.Provider value={value}>
      {children}
    </SocketContext.Provider>
  );
};