import axios from 'axios';

const API_BASE_URL = window.location.origin + '/api';

class ApiService {
  constructor() {
    this.client = axios.create({
      baseURL: API_BASE_URL,
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json'
      }
    });

    // Request interceptor
    this.client.interceptors.request.use(
      (config) => {
        console.log(`API Request: ${config.method?.toUpperCase()} ${config.url}`);
        return config;
      },
      (error) => {
        console.error('API Request Error:', error);
        return Promise.reject(error);
      }
    );

    // Response interceptor
    this.client.interceptors.response.use(
      (response) => {
        console.log(`API Response: ${response.status} ${response.config.url}`);
        return response;
      },
      (error) => {
        console.error('API Response Error:', error.response?.data || error.message);
        return Promise.reject(error);
      }
    );
  }

  // Session Management
  async createSession(config) {
    try {
      const response = await this.client.post('/sessions', config);
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  async startEvolution(sessionId, parameters) {
    try {
      const response = await this.client.post(`/sessions/${sessionId}/start`, parameters);
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  async stopEvolution(sessionId) {
    try {
      const response = await this.client.post(`/sessions/${sessionId}/stop`);
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  async getSession(sessionId) {
    try {
      const response = await this.client.get(`/sessions/${sessionId}`);
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  async getAllSessions() {
    try {
      const response = await this.client.get('/sessions');
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  // System Metrics
  async getSystemMetrics() {
    try {
      const response = await this.client.get('/system/metrics');
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  // LLM Integration
  async generateWithLLM(prompt, type, context = {}) {
    try {
      const response = await this.client.post('/llm/generate', {
        prompt,
        type,
        context
      });
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  // Agent Operations
  async exportAgent(sessionId, agentId) {
    try {
      const response = await this.client.get(`/sessions/${sessionId}/agents/${agentId}/export`);
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  async importAgent(sessionId, agentData) {
    try {
      const response = await this.client.post(`/sessions/${sessionId}/agents/import`, agentData);
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  async cloneAgent(sessionId, agentId) {
    try {
      const response = await this.client.post(`/sessions/${sessionId}/agents/${agentId}/clone`);
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  // Evolution Analytics
  async getEvolutionAnalytics(sessionId, timeRange = '1h') {
    try {
      const response = await this.client.get(`/sessions/${sessionId}/analytics`, {
        params: { timeRange }
      });
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  async getPopulationDiversity(sessionId, generation) {
    try {
      const response = await this.client.get(`/sessions/${sessionId}/diversity`, {
        params: { generation }
      });
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  async getFitnessLandscape(sessionId) {
    try {
      const response = await this.client.get(`/sessions/${sessionId}/fitness-landscape`);
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  // Configuration Management
  async saveConfiguration(name, config) {
    try {
      const response = await this.client.post('/configurations', { name, config });
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  async loadConfiguration(name) {
    try {
      const response = await this.client.get(`/configurations/${name}`);
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  async getConfigurations() {
    try {
      const response = await this.client.get('/configurations');
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  // File Operations
  async uploadFile(file, type = 'agent') {
    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('type', type);

      const response = await this.client.post('/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  async downloadFile(fileId, filename) {
    try {
      const response = await this.client.get(`/download/${fileId}`, {
        responseType: 'blob'
      });
      
      // Create download link
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', filename);
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(url);
      
      return true;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  // Batch Operations
  async batchOperation(operation, items) {
    try {
      const response = await this.client.post('/batch', {
        operation,
        items
      });
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  // Real-time Data Streaming
  async getStreamingData(sessionId, dataType, callback) {
    try {
      const response = await this.client.get(`/sessions/${sessionId}/stream/${dataType}`, {
        responseType: 'stream'
      });
      
      response.data.on('data', (chunk) => {
        try {
          const data = JSON.parse(chunk.toString());
          callback(data);
        } catch (error) {
          console.error('Error parsing streaming data:', error);
        }
      });
      
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  // Advanced Analytics
  async getPerformanceProfile(sessionId) {
    try {
      const response = await this.client.get(`/sessions/${sessionId}/performance-profile`);
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  async getEvolutionPrediction(sessionId, generations = 10) {
    try {
      const response = await this.client.post(`/sessions/${sessionId}/predict`, {
        generations
      });
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  async getOptimizationSuggestions(sessionId) {
    try {
      const response = await this.client.get(`/sessions/${sessionId}/optimization-suggestions`);
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  // Error Handling
  handleError(error) {
    if (error.response) {
      // Server responded with error status
      const { status, data } = error.response;
      return {
        type: 'api_error',
        status,
        message: data.error || data.message || 'An API error occurred',
        details: data
      };
    } else if (error.request) {
      // Request was made but no response received
      return {
        type: 'network_error',
        message: 'Network error - please check your connection',
        details: error.request
      };
    } else {
      // Something else happened
      return {
        type: 'unknown_error',
        message: error.message || 'An unknown error occurred',
        details: error
      };
    }
  }

  // Utility Methods
  async healthCheck() {
    try {
      const response = await this.client.get('/health');
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  async getApiInfo() {
    try {
      const response = await this.client.get('/info');
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }
}

// Create singleton instance
const apiService = new ApiService();

export default apiService;