import React, { useState, useEffect, useRef } from 'react';
import { useSocket } from '../services/SocketService';
import apiService from '../services/ApiService';
import './Dashboard.css';

const Dashboard = ({ currentSession }) => {
  const { connected, systemMetrics, evolutionData } = useSocket();
  const [sessions, setSessions] = useState([]);
  const [realtimeData, setRealtimeData] = useState({
    fitness: [],
    diversity: [],
    performance: []
  });
  const [selectedTimeRange, setSelectedTimeRange] = useState('1h');
  const [loading, setLoading] = useState(true);
  const [activeMetric, setActiveMetric] = useState('fitness');
  
  const chartRef = useRef(null);
  const performanceChartRef = useRef(null);
  const diversityChartRef = useRef(null);

  // Load initial data
  useEffect(() => {
    const loadDashboardData = async () => {
      try {
        setLoading(true);
        const sessionsData = await apiService.getAllSessions();
        setSessions(sessionsData);
        
        // Generate mock realtime data
        generateMockRealtimeData();
      } catch (error) {
        console.error('Failed to load dashboard data:', error);
      } finally {
        setLoading(false);
      }
    };

    loadDashboardData();
  }, []);

  // Update realtime data
  useEffect(() => {
    const interval = setInterval(() => {
      generateMockRealtimeData();
    }, 2000);

    return () => clearInterval(interval);
  }, [selectedTimeRange]);

  // Update charts when data changes
  useEffect(() => {
    if (realtimeData.fitness.length > 0) {
      updateCharts();
    }
  }, [realtimeData, activeMetric]);

  const generateMockRealtimeData = () => {
    const now = new Date();
    const timePoints = [];
    const fitnessData = [];
    const diversityData = [];
    const performanceData = [];

    // Generate data points based on time range
    const ranges = {
      '1h': { points: 60, interval: 60000 },
      '6h': { points: 72, interval: 300000 },
      '24h': { points: 96, interval: 900000 },
      '7d': { points: 168, interval: 3600000 }
    };

    const range = ranges[selectedTimeRange];
    
    for (let i = range.points - 1; i >= 0; i--) {
      const time = new Date(now.getTime() - (i * range.interval));
      timePoints.push(time);
      
      // Generate realistic fitness progression
      const baseFitness = 0.3 + (Math.sin((range.points - i) * 0.1) * 0.2) + ((range.points - i) * 0.005);
      fitnessData.push({
        time,
        best: Math.min(1, baseFitness + (Math.random() * 0.2)),
        average: Math.min(1, baseFitness + (Math.random() * 0.1)),
        worst: Math.max(0, baseFitness - (Math.random() * 0.2))
      });
      
      // Generate diversity data
      diversityData.push({
        time,
        behavioral: Math.max(0.1, 0.8 - ((range.points - i) * 0.01) + (Math.random() * 0.2)),
        structural: Math.max(0.1, 0.7 - ((range.points - i) * 0.008) + (Math.random() * 0.15)),
        functional: Math.max(0.1, 0.6 - ((range.points - i) * 0.006) + (Math.random() * 0.1))
      });
      
      // Generate performance data
      performanceData.push({
        time,
        cpu: 20 + (Math.sin((range.points - i) * 0.2) * 30) + (Math.random() * 20),
        memory: 30 + (Math.sin((range.points - i) * 0.15) * 20) + (Math.random() * 15),
        throughput: 50 + (Math.sin((range.points - i) * 0.1) * 25) + (Math.random() * 10)
      });
    }

    setRealtimeData({
      fitness: fitnessData,
      diversity: diversityData,
      performance: performanceData
    });
  };

  const updateCharts = () => {
    // Update main chart
    if (chartRef.current) {
      updateMainChart();
    }
    
    // Update performance chart
    if (performanceChartRef.current) {
      updatePerformanceChart();
    }
    
    // Update diversity chart
    if (diversityChartRef.current) {
      updateDiversityChart();
    }
  };

  const updateMainChart = () => {
    const canvas = chartRef.current;
    const ctx = canvas.getContext('2d');
    const { width, height } = canvas;
    
    // Clear canvas
    ctx.clearRect(0, 0, width, height);
    
    // Set up chart area
    const padding = 40;
    const chartWidth = width - (padding * 2);
    const chartHeight = height - (padding * 2);
    
    // Draw grid
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
    ctx.lineWidth = 1;
    
    // Vertical grid lines
    for (let i = 0; i <= 10; i++) {
      const x = padding + (chartWidth / 10) * i;
      ctx.beginPath();
      ctx.moveTo(x, padding);
      ctx.lineTo(x, height - padding);
      ctx.stroke();
    }
    
    // Horizontal grid lines
    for (let i = 0; i <= 5; i++) {
      const y = padding + (chartHeight / 5) * i;
      ctx.beginPath();
      ctx.moveTo(padding, y);
      ctx.lineTo(width - padding, y);
      ctx.stroke();
    }
    
    // Draw data based on active metric
    const data = realtimeData[activeMetric];
    if (data.length === 0) return;
    
    if (activeMetric === 'fitness') {
      drawFitnessLines(ctx, data, padding, chartWidth, chartHeight);
    } else if (activeMetric === 'diversity') {
      drawDiversityLines(ctx, data, padding, chartWidth, chartHeight);
    } else if (activeMetric === 'performance') {
      drawPerformanceLines(ctx, data, padding, chartWidth, chartHeight);
    }
    
    // Draw axes labels
    drawAxesLabels(ctx, width, height, padding);
  };

  const drawFitnessLines = (ctx, data, padding, chartWidth, chartHeight) => {
    const colors = {
      best: '#10b981',
      average: '#00d4ff',
      worst: '#ef4444'
    };
    
    Object.keys(colors).forEach(key => {
      ctx.strokeStyle = colors[key];
      ctx.lineWidth = 2;
      ctx.beginPath();
      
      data.forEach((point, index) => {
        const x = padding + (chartWidth / (data.length - 1)) * index;
        const y = padding + chartHeight - (point[key] * chartHeight);
        
        if (index === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      });
      
      ctx.stroke();
    });
  };

  const drawDiversityLines = (ctx, data, padding, chartWidth, chartHeight) => {
    const colors = {
      behavioral: '#8b5cf6',
      structural: '#f59e0b',
      functional: '#06b6d4'
    };
    
    Object.keys(colors).forEach(key => {
      ctx.strokeStyle = colors[key];
      ctx.lineWidth = 2;
      ctx.beginPath();
      
      data.forEach((point, index) => {
        const x = padding + (chartWidth / (data.length - 1)) * index;
        const y = padding + chartHeight - (point[key] * chartHeight);
        
        if (index === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      });
      
      ctx.stroke();
    });
  };

  const drawPerformanceLines = (ctx, data, padding, chartWidth, chartHeight) => {
    const colors = {
      cpu: '#ef4444',
      memory: '#f59e0b',
      throughput: '#10b981'
    };
    
    Object.keys(colors).forEach(key => {
      ctx.strokeStyle = colors[key];
      ctx.lineWidth = 2;
      ctx.beginPath();
      
      data.forEach((point, index) => {
        const x = padding + (chartWidth / (data.length - 1)) * index;
        const y = padding + chartHeight - ((point[key] / 100) * chartHeight);
        
        if (index === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      });
      
      ctx.stroke();
    });
  };

  const drawAxesLabels = (ctx, width, height, padding) => {
    ctx.fillStyle = 'rgba(255, 255, 255, 0.6)';
    ctx.font = '12px Inter';
    ctx.textAlign = 'center';
    
    // Y-axis labels
    for (let i = 0; i <= 5; i++) {
      const y = padding + ((height - padding * 2) / 5) * i;
      const value = (1 - (i / 5)).toFixed(1);
      ctx.fillText(value, padding - 20, y + 4);
    }
    
    // X-axis labels (time)
    const timeLabels = getTimeLabels();
    timeLabels.forEach((label, index) => {
      const x = padding + ((width - padding * 2) / (timeLabels.length - 1)) * index;
      ctx.fillText(label, x, height - padding + 20);
    });
  };

  const getTimeLabels = () => {
    const now = new Date();
    const labels = [];
    
    if (selectedTimeRange === '1h') {
      for (let i = 5; i >= 0; i--) {
        const time = new Date(now.getTime() - (i * 10 * 60000));
        labels.push(time.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' }));
      }
    } else if (selectedTimeRange === '6h') {
      for (let i = 5; i >= 0; i--) {
        const time = new Date(now.getTime() - (i * 60 * 60000));
        labels.push(time.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' }));
      }
    } else if (selectedTimeRange === '24h') {
      for (let i = 5; i >= 0; i--) {
        const time = new Date(now.getTime() - (i * 4 * 60 * 60000));
        labels.push(time.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' }));
      }
    } else {
      for (let i = 6; i >= 0; i--) {
        const time = new Date(now.getTime() - (i * 24 * 60 * 60000));
        labels.push(time.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }));
      }
    }
    
    return labels;
  };

  const updatePerformanceChart = () => {
    // Similar implementation for performance chart
  };

  const updateDiversityChart = () => {
    // Similar implementation for diversity chart
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

  if (loading) {
    return (
      <div className="dashboard loading">
        <div className="loading-spinner"></div>
        <div className="loading-text">Loading dashboard...</div>
      </div>
    );
  }

  return (
    <div className="dashboard">
      {/* Header Stats */}
      <div className="dashboard-header">
        <div className="stats-grid">
          <div className="stat-card">
            <div className="stat-icon">
              <i className="fas fa-robot"></i>
            </div>
            <div className="stat-content">
              <div className="stat-value">{systemMetrics.totalAgents || 0}</div>
              <div className="stat-label">Active Agents</div>
              <div className="stat-change positive">+12 today</div>
            </div>
          </div>
          
          <div className="stat-card">
            <div className="stat-icon">
              <i className="fas fa-flask"></i>
            </div>
            <div className="stat-content">
              <div className="stat-value">{sessions.length}</div>
              <div className="stat-label">Evolution Sessions</div>
              <div className="stat-change positive">+3 active</div>
            </div>
          </div>
          
          <div className="stat-card">
            <div className="stat-icon">
              <i className="fas fa-trophy"></i>
            </div>
            <div className="stat-content">
              <div className="stat-value">
                {systemMetrics.bestFitness ? (systemMetrics.bestFitness * 100).toFixed(1) + '%' : '0%'}
              </div>
              <div className="stat-label">Best Fitness</div>
              <div className="stat-change positive">+5.2% today</div>
            </div>
          </div>
          
          <div className="stat-card">
            <div className="stat-icon">
              <i className="fas fa-server"></i>
            </div>
            <div className="stat-content">
              <div className="stat-value">
                {systemMetrics.systemLoad ? systemMetrics.systemLoad.toFixed(0) + '%' : '0%'}
              </div>
              <div className="stat-label">System Load</div>
              <div className={`stat-change ${systemMetrics.systemLoad > 70 ? 'negative' : 'neutral'}`}>
                {systemMetrics.systemLoad > 70 ? 'High' : 'Normal'}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="dashboard-content">
        <div className="dashboard-left">
          {/* Real-time Chart */}
          <div className="chart-container">
            <div className="chart-header">
              <h3>Real-time Metrics</h3>
              <div className="chart-controls">
                <div className="metric-selector">
                  {['fitness', 'diversity', 'performance'].map(metric => (
                    <button
                      key={metric}
                      className={`metric-btn ${activeMetric === metric ? 'active' : ''}`}
                      onClick={() => setActiveMetric(metric)}
                    >
                      {metric.charAt(0).toUpperCase() + metric.slice(1)}
                    </button>
                  ))}
                </div>
                <div className="time-selector">
                  {['1h', '6h', '24h', '7d'].map(range => (
                    <button
                      key={range}
                      className={`time-btn ${selectedTimeRange === range ? 'active' : ''}`}
                      onClick={() => setSelectedTimeRange(range)}
                    >
                      {range}
                    </button>
                  ))}
                </div>
              </div>
            </div>
            
            <div className="chart-body">
              <canvas
                ref={chartRef}
                width={800}
                height={400}
                className="main-chart"
              />
              
              {/* Chart Legend */}
              <div className="chart-legend">
                {activeMetric === 'fitness' && (
                  <>
                    <div className="legend-item">
                      <div className="legend-color" style={{ background: '#10b981' }}></div>
                      <span>Best Fitness</span>
                    </div>
                    <div className="legend-item">
                      <div className="legend-color" style={{ background: '#00d4ff' }}></div>
                      <span>Average Fitness</span>
                    </div>
                    <div className="legend-item">
                      <div className="legend-color" style={{ background: '#ef4444' }}></div>
                      <span>Worst Fitness</span>
                    </div>
                  </>
                )}
                
                {activeMetric === 'diversity' && (
                  <>
                    <div className="legend-item">
                      <div className="legend-color" style={{ background: '#8b5cf6' }}></div>
                      <span>Behavioral</span>
                    </div>
                    <div className="legend-item">
                      <div className="legend-color" style={{ background: '#f59e0b' }}></div>
                      <span>Structural</span>
                    </div>
                    <div className="legend-item">
                      <div className="legend-color" style={{ background: '#06b6d4' }}></div>
                      <span>Functional</span>
                    </div>
                  </>
                )}
                
                {activeMetric === 'performance' && (
                  <>
                    <div className="legend-item">
                      <div className="legend-color" style={{ background: '#ef4444' }}></div>
                      <span>CPU Usage</span>
                    </div>
                    <div className="legend-item">
                      <div className="legend-color" style={{ background: '#f59e0b' }}></div>
                      <span>Memory Usage</span>
                    </div>
                    <div className="legend-item">
                      <div className="legend-color" style={{ background: '#10b981' }}></div>
                      <span>Throughput</span>
                    </div>
                  </>
                )}
              </div>
            </div>
          </div>

          {/* Evolution Progress */}
          {currentSession && (
            <div className="evolution-progress">
              <div className="progress-header">
                <h3>Current Evolution Progress</h3>
                <div className="session-info">
                  <span className="session-id">{currentSession.id.substring(0, 8)}...</span>
                  <div className={`status-indicator ${currentSession.status}`}>
                    <div className="status-dot"></div>
                    <span>{currentSession.status.toUpperCase()}</span>
                  </div>
                </div>
              </div>
              
              <div className="progress-content">
                <div className="progress-stats">
                  <div className="progress-stat">
                    <div className="stat-label">Generation</div>
                    <div className="stat-value">{currentSession.currentGeneration || 0}</div>
                  </div>
                  <div className="progress-stat">
                    <div className="stat-label">Population</div>
                    <div className="stat-value">{currentSession.population?.length || 0}</div>
                  </div>
                  <div className="progress-stat">
                    <div className="stat-label">Best Fitness</div>
                    <div className="stat-value">
                      {currentSession.bestAgent?.fitness ? 
                        (currentSession.bestAgent.fitness * 100).toFixed(1) + '%' : '0%'}
                    </div>
                  </div>
                  <div className="progress-stat">
                    <div className="stat-label">Duration</div>
                    <div className="stat-value">{formatDuration(currentSession.startTime)}</div>
                  </div>
                </div>
                
                {currentSession.evolutionStats?.fitnessHistory && (
                  <div className="mini-chart">
                    <canvas width={400} height={100} className="progress-chart"></canvas>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>

        <div className="dashboard-right">
          {/* Active Sessions */}
          <div className="sessions-panel">
            <div className="panel-header">
              <h3>Active Sessions</h3>
              <button className="btn btn-sm btn-primary">
                <i className="fas fa-plus"></i>
                New Session
              </button>
            </div>
            
            <div className="sessions-list">
              {sessions.slice(0, 5).map(session => (
                <div key={session.id} className="session-item">
                  <div className="session-info">
                    <div className="session-name">
                      Session {session.id.substring(0, 8)}...
                    </div>
                    <div className="session-details">
                      <span className="session-generation">Gen {session.currentGeneration || 0}</span>
                      <span className="session-duration">{formatDuration(session.startTime)}</span>
                    </div>
                  </div>
                  <div className="session-status">
                    <div 
                      className="status-dot" 
                      style={{ background: getStatusColor(session.status) }}
                    ></div>
                    <span className="status-text">{session.status}</span>
                  </div>
                </div>
              ))}
            </div>
            
            <div className="panel-footer">
              <button className="btn btn-sm btn-secondary">View All Sessions</button>
            </div>
          </div>

          {/* System Health */}
          <div className="health-panel">
            <div className="panel-header">
              <h3>System Health</h3>
              <div className={`health-status ${connected ? 'healthy' : 'unhealthy'}`}>
                <div className="health-dot"></div>
                <span>{connected ? 'Healthy' : 'Disconnected'}</span>
              </div>
            </div>
            
            <div className="health-metrics">
              <div className="health-metric">
                <div className="metric-label">
                  <i className="fas fa-microchip"></i>
                  CPU Usage
                </div>
                <div className="metric-value">
                  {systemMetrics.systemLoad ? systemMetrics.systemLoad.toFixed(0) + '%' : '0%'}
                </div>
                <div className="metric-bar">
                  <div 
                    className="metric-fill"
                    style={{ width: `${systemMetrics.systemLoad || 0}%` }}
                  ></div>
                </div>
              </div>
              
              <div className="health-metric">
                <div className="metric-label">
                  <i className="fas fa-memory"></i>
                  Memory Usage
                </div>
                <div className="metric-value">65%</div>
                <div className="metric-bar">
                  <div className="metric-fill" style={{ width: '65%' }}></div>
                </div>
              </div>
              
              <div className="health-metric">
                <div className="metric-label">
                  <i className="fas fa-network-wired"></i>
                  Network I/O
                </div>
                <div className="metric-value">42%</div>
                <div className="metric-bar">
                  <div className="metric-fill" style={{ width: '42%' }}></div>
                </div>
              </div>
            </div>
          </div>

          {/* Recent Activity */}
          <div className="activity-panel">
            <div className="panel-header">
              <h3>Recent Activity</h3>
              <button className="btn btn-sm btn-secondary">
                <i className="fas fa-refresh"></i>
              </button>
            </div>
            
            <div className="activity-list">
              <div className="activity-item">
                <div className="activity-icon success">
                  <i className="fas fa-check"></i>
                </div>
                <div className="activity-content">
                  <div className="activity-title">Evolution completed</div>
                  <div className="activity-description">Session #abc123 reached 95% fitness</div>
                  <div className="activity-time">2 minutes ago</div>
                </div>
              </div>
              
              <div className="activity-item">
                <div className="activity-icon info">
                  <i className="fas fa-robot"></i>
                </div>
                <div className="activity-content">
                  <div className="activity-title">New agent generated</div>
                  <div className="activity-description">Agent #xyz789 shows promising results</div>
                  <div className="activity-time">5 minutes ago</div>
                </div>
              </div>
              
              <div className="activity-item">
                <div className="activity-icon warning">
                  <i className="fas fa-exclamation-triangle"></i>
                </div>
                <div className="activity-content">
                  <div className="activity-title">High system load detected</div>
                  <div className="activity-description">CPU usage at 78%, consider optimization</div>
                  <div className="activity-time">10 minutes ago</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;