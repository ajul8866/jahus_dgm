import React, { useState, useEffect } from 'react';
import { useSocket } from '../services/SocketService';
import apiService from '../services/ApiService';
import './EvolutionControl.css';

const EvolutionControl = ({ currentSession, setCurrentSession }) => {
  const { connected, evolutionData, subscribeToSession, unsubscribeFromSession } = useSocket();
  const [sessions, setSessions] = useState([]);
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [loading, setLoading] = useState(false);
  const [evolutionParams, setEvolutionParams] = useState({
    populationSize: 100,
    maxGenerations: 50,
    mutationRate: 0.1,
    crossoverRate: 0.8,
    elitismRate: 0.1,
    selectionStrategy: 'tournament',
    fitnessFunction: 'multi_objective',
    diversityMetric: 'behavioral',
    generationInterval: 1000
  });

  useEffect(() => {
    loadSessions();
  }, []);

  useEffect(() => {
    if (currentSession) {
      subscribeToSession(currentSession.id);
      return () => unsubscribeFromSession(currentSession.id);
    }
  }, [currentSession]);

  const loadSessions = async () => {
    try {
      const data = await apiService.getAllSessions();
      setSessions(data);
    } catch (error) {
      console.error('Failed to load sessions:', error);
    }
  };

  const createSession = async () => {
    try {
      setLoading(true);
      const config = {
        name: `Evolution Session ${Date.now()}`,
        description: 'Advanced evolution session',
        parameters: evolutionParams
      };
      
      const result = await apiService.createSession(config);
      setCurrentSession(result.session);
      setSessions(prev => [...prev, result.session]);
      setShowCreateModal(false);
    } catch (error) {
      console.error('Failed to create session:', error);
    } finally {
      setLoading(false);
    }
  };

  const startEvolution = async (sessionId) => {
    try {
      setLoading(true);
      await apiService.startEvolution(sessionId, evolutionParams);
      loadSessions();
    } catch (error) {
      console.error('Failed to start evolution:', error);
    } finally {
      setLoading(false);
    }
  };

  const stopEvolution = async (sessionId) => {
    try {
      setLoading(true);
      await apiService.stopEvolution(sessionId);
      loadSessions();
    } catch (error) {
      console.error('Failed to stop evolution:', error);
    } finally {
      setLoading(false);
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'running': return 'var(--accent-green)';
      case 'completed': return 'var(--accent-blue)';
      case 'stopped': return 'var(--accent-red)';
      default: return 'var(--text-muted)';
    }
  };

  const formatDuration = (startTime) => {
    const now = new Date();
    const start = new Date(startTime);
    const diff = now - start;
    
    const hours = Math.floor(diff / 3600000);
    const minutes = Math.floor((diff % 3600000) / 60000);
    
    if (hours > 0) {
      return `${hours}h ${minutes}m`;
    }
    return `${minutes}m`;
  };

  return (
    <div className="evolution-control">
      {/* Header */}
      <div className="control-header">
        <div className="header-left">
          <h2>Evolution Control Center</h2>
          <p>Manage and monitor evolution processes</p>
        </div>
        <div className="header-right">
          <button 
            className="btn btn-primary"
            onClick={() => setShowCreateModal(true)}
            disabled={!connected}
          >
            <i className="fas fa-plus"></i>
            New Evolution Session
          </button>
        </div>
      </div>

      {/* Current Session Panel */}
      {currentSession && (
        <div className="current-session-panel">
          <div className="panel-header">
            <div className="session-info">
              <h3>Current Session</h3>
              <div className="session-details">
                <span className="session-id">{currentSession.id}</span>
                <div className={`status-badge ${currentSession.status}`}>
                  <div className="status-dot"></div>
                  {currentSession.status.toUpperCase()}
                </div>
              </div>
            </div>
            <div className="session-controls">
              {currentSession.status === 'initialized' && (
                <button 
                  className="btn btn-success"
                  onClick={() => startEvolution(currentSession.id)}
                  disabled={loading}
                >
                  <i className="fas fa-play"></i>
                  Start Evolution
                </button>
              )}
              {currentSession.status === 'running' && (
                <button 
                  className="btn btn-danger"
                  onClick={() => stopEvolution(currentSession.id)}
                  disabled={loading}
                >
                  <i className="fas fa-stop"></i>
                  Stop Evolution
                </button>
              )}
              <button className="btn btn-secondary">
                <i className="fas fa-download"></i>
                Export
              </button>
            </div>
          </div>

          <div className="session-content">
            <div className="session-stats">
              <div className="stat-group">
                <div className="stat-item">
                  <div className="stat-label">Generation</div>
                  <div className="stat-value">{currentSession.currentGeneration || 0}</div>
                </div>
                <div className="stat-item">
                  <div className="stat-label">Population</div>
                  <div className="stat-value">{currentSession.population?.length || 0}</div>
                </div>
                <div className="stat-item">
                  <div className="stat-label">Best Fitness</div>
                  <div className="stat-value">
                    {currentSession.bestAgent?.fitness ? 
                      (currentSession.bestAgent.fitness * 100).toFixed(1) + '%' : '0%'}
                  </div>
                </div>
                <div className="stat-item">
                  <div className="stat-label">Duration</div>
                  <div className="stat-value">{formatDuration(currentSession.startTime)}</div>
                </div>
              </div>

              {currentSession.status === 'running' && evolutionData[currentSession.id] && (
                <div className="realtime-stats">
                  <div className="realtime-item">
                    <div className="realtime-label">Mutations</div>
                    <div className="realtime-value">
                      {evolutionData[currentSession.id].data?.mutations || 0}
                    </div>
                  </div>
                  <div className="realtime-item">
                    <div className="realtime-label">Crossovers</div>
                    <div className="realtime-value">
                      {evolutionData[currentSession.id].data?.crossovers || 0}
                    </div>
                  </div>
                  <div className="realtime-item">
                    <div className="realtime-label">Elites</div>
                    <div className="realtime-value">
                      {evolutionData[currentSession.id].data?.elites || 0}
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Evolution Progress */}
            {currentSession.evolutionStats?.fitnessHistory && (
              <div className="evolution-progress">
                <div className="progress-header">
                  <h4>Evolution Progress</h4>
                  <div className="progress-controls">
                    <button className="btn btn-sm btn-secondary">
                      <i className="fas fa-chart-line"></i>
                      View Details
                    </button>
                  </div>
                </div>
                <div className="progress-chart">
                  <canvas width={600} height={200} className="fitness-chart"></canvas>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Sessions List */}
      <div className="sessions-container">
        <div className="container-header">
          <h3>Evolution Sessions</h3>
          <div className="session-filters">
            <button className="filter-btn active">All</button>
            <button className="filter-btn">Running</button>
            <button className="filter-btn">Completed</button>
            <button className="filter-btn">Stopped</button>
          </div>
        </div>

        <div className="sessions-grid">
          {sessions.map(session => (
            <div 
              key={session.id} 
              className={`session-card ${currentSession?.id === session.id ? 'active' : ''}`}
              onClick={() => setCurrentSession(session)}
            >
              <div className="card-header">
                <div className="session-title">
                  <h4>Session {session.id.substring(0, 8)}...</h4>
                  <div className={`status-indicator ${session.status}`}>
                    <div className="status-dot"></div>
                    <span>{session.status}</span>
                  </div>
                </div>
                <div className="session-actions">
                  <button className="action-btn" title="View Details">
                    <i className="fas fa-eye"></i>
                  </button>
                  <button className="action-btn" title="Clone Session">
                    <i className="fas fa-copy"></i>
                  </button>
                  <button className="action-btn danger" title="Delete Session">
                    <i className="fas fa-trash"></i>
                  </button>
                </div>
              </div>

              <div className="card-content">
                <div className="session-metrics">
                  <div className="metric">
                    <span className="metric-label">Generation:</span>
                    <span className="metric-value">{session.currentGeneration || 0}</span>
                  </div>
                  <div className="metric">
                    <span className="metric-label">Population:</span>
                    <span className="metric-value">{session.population?.length || 0}</span>
                  </div>
                  <div className="metric">
                    <span className="metric-label">Best Fitness:</span>
                    <span className="metric-value">
                      {session.bestAgent?.fitness ? 
                        (session.bestAgent.fitness * 100).toFixed(1) + '%' : '0%'}
                    </span>
                  </div>
                  <div className="metric">
                    <span className="metric-label">Duration:</span>
                    <span className="metric-value">{formatDuration(session.startTime)}</span>
                  </div>
                </div>

                <div className="session-progress">
                  <div className="progress-bar">
                    <div 
                      className="progress-fill"
                      style={{ 
                        width: `${((session.currentGeneration || 0) / 50) * 100}%` 
                      }}
                    ></div>
                  </div>
                  <div className="progress-text">
                    {session.currentGeneration || 0} / 50 generations
                  </div>
                </div>
              </div>

              <div className="card-footer">
                <div className="session-time">
                  Started: {new Date(session.startTime).toLocaleString()}
                </div>
                <div className="session-controls">
                  {session.status === 'initialized' && (
                    <button 
                      className="btn btn-sm btn-success"
                      onClick={(e) => {
                        e.stopPropagation();
                        startEvolution(session.id);
                      }}
                    >
                      <i className="fas fa-play"></i>
                      Start
                    </button>
                  )}
                  {session.status === 'running' && (
                    <button 
                      className="btn btn-sm btn-danger"
                      onClick={(e) => {
                        e.stopPropagation();
                        stopEvolution(session.id);
                      }}
                    >
                      <i className="fas fa-stop"></i>
                      Stop
                    </button>
                  )}
                  {session.status === 'completed' && (
                    <button className="btn btn-sm btn-primary">
                      <i className="fas fa-download"></i>
                      Export
                    </button>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Create Session Modal */}
      {showCreateModal && (
        <div className="modal-overlay">
          <div className="modal create-session-modal">
            <div className="modal-header">
              <h3>Create New Evolution Session</h3>
              <button 
                className="modal-close"
                onClick={() => setShowCreateModal(false)}
              >
                <i className="fas fa-times"></i>
              </button>
            </div>

            <div className="modal-body">
              <div className="form-grid">
                <div className="form-group">
                  <label className="form-label">Population Size</label>
                  <input
                    type="number"
                    className="form-input"
                    value={evolutionParams.populationSize}
                    onChange={(e) => setEvolutionParams(prev => ({
                      ...prev,
                      populationSize: parseInt(e.target.value)
                    }))}
                    min="10"
                    max="1000"
                  />
                </div>

                <div className="form-group">
                  <label className="form-label">Max Generations</label>
                  <input
                    type="number"
                    className="form-input"
                    value={evolutionParams.maxGenerations}
                    onChange={(e) => setEvolutionParams(prev => ({
                      ...prev,
                      maxGenerations: parseInt(e.target.value)
                    }))}
                    min="1"
                    max="1000"
                  />
                </div>

                <div className="form-group">
                  <label className="form-label">Mutation Rate</label>
                  <input
                    type="range"
                    className="form-range"
                    value={evolutionParams.mutationRate}
                    onChange={(e) => setEvolutionParams(prev => ({
                      ...prev,
                      mutationRate: parseFloat(e.target.value)
                    }))}
                    min="0"
                    max="1"
                    step="0.01"
                  />
                  <span className="range-value">{evolutionParams.mutationRate}</span>
                </div>

                <div className="form-group">
                  <label className="form-label">Crossover Rate</label>
                  <input
                    type="range"
                    className="form-range"
                    value={evolutionParams.crossoverRate}
                    onChange={(e) => setEvolutionParams(prev => ({
                      ...prev,
                      crossoverRate: parseFloat(e.target.value)
                    }))}
                    min="0"
                    max="1"
                    step="0.01"
                  />
                  <span className="range-value">{evolutionParams.crossoverRate}</span>
                </div>

                <div className="form-group">
                  <label className="form-label">Selection Strategy</label>
                  <select
                    className="form-select"
                    value={evolutionParams.selectionStrategy}
                    onChange={(e) => setEvolutionParams(prev => ({
                      ...prev,
                      selectionStrategy: e.target.value
                    }))}
                  >
                    <option value="tournament">Tournament Selection</option>
                    <option value="roulette">Roulette Wheel</option>
                    <option value="rank">Rank Selection</option>
                    <option value="nsga2">NSGA-II</option>
                  </select>
                </div>

                <div className="form-group">
                  <label className="form-label">Fitness Function</label>
                  <select
                    className="form-select"
                    value={evolutionParams.fitnessFunction}
                    onChange={(e) => setEvolutionParams(prev => ({
                      ...prev,
                      fitnessFunction: e.target.value
                    }))}
                  >
                    <option value="single_objective">Single Objective</option>
                    <option value="multi_objective">Multi-Objective</option>
                    <option value="adaptive">Adaptive Fitness</option>
                  </select>
                </div>
              </div>

              <div className="advanced-options">
                <h4>Advanced Options</h4>
                <div className="options-grid">
                  <div className="form-group">
                    <label className="form-label">Elitism Rate</label>
                    <input
                      type="range"
                      className="form-range"
                      value={evolutionParams.elitismRate}
                      onChange={(e) => setEvolutionParams(prev => ({
                        ...prev,
                        elitismRate: parseFloat(e.target.value)
                      }))}
                      min="0"
                      max="0.5"
                      step="0.01"
                    />
                    <span className="range-value">{evolutionParams.elitismRate}</span>
                  </div>

                  <div className="form-group">
                    <label className="form-label">Generation Interval (ms)</label>
                    <input
                      type="number"
                      className="form-input"
                      value={evolutionParams.generationInterval}
                      onChange={(e) => setEvolutionParams(prev => ({
                        ...prev,
                        generationInterval: parseInt(e.target.value)
                      }))}
                      min="100"
                      max="10000"
                    />
                  </div>
                </div>
              </div>
            </div>

            <div className="modal-footer">
              <button 
                className="btn btn-secondary"
                onClick={() => setShowCreateModal(false)}
              >
                Cancel
              </button>
              <button 
                className="btn btn-primary"
                onClick={createSession}
                disabled={loading}
              >
                {loading ? (
                  <>
                    <div className="loading-spinner small"></div>
                    Creating...
                  </>
                ) : (
                  <>
                    <i className="fas fa-plus"></i>
                    Create Session
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default EvolutionControl;