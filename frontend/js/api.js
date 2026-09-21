/**
 * Ziro Predict - Centralized FastAPI Service Client
 * Handles all HTTP communication, error normalization, timeouts, and request aborts.
 */

import { CONFIG, getApiBaseUrl } from './config.js';

class ApiClient {
  /**
   * Universal fetch wrapper with timeout and error handling
   */
  async request(endpoint, options = {}) {
    const baseUrl = getApiBaseUrl();
    const url = `${baseUrl}${endpoint.startsWith('/') ? endpoint : '/' + endpoint}`;
    
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), CONFIG.REQUEST_TIMEOUT_MS);

    const defaultHeaders = {
      'Accept': 'application/json',
      'Content-Type': 'application/json'
    };

    const config = {
      ...options,
      headers: {
        ...defaultHeaders,
        ...options.headers
      },
      signal: controller.signal
    };

    try {
      const response = await fetch(url, config);
      clearTimeout(timeoutId);

      // Handle non-2xx HTTP responses
      if (!response.ok) {
        let errorDetail = `HTTP Error ${response.status}: ${response.statusText}`;
        try {
          const errData = await response.json();
          if (errData && errData.detail) {
            // detail can be string or array of pydantic validation errors
            if (typeof errData.detail === 'string') {
              errorDetail = errData.detail;
            } else if (Array.isArray(errData.detail)) {
              errorDetail = errData.detail.map(d => `${d.loc ? d.loc.join('.') + ': ' : ''}${d.msg}`).join(', ');
            }
          }
        } catch (_) {
          // If response body is not valid JSON, use fallback text
        }
        
        const error = new Error(errorDetail);
        error.status = response.status;
        throw error;
      }

      return await response.json();
    } catch (err) {
      clearTimeout(timeoutId);
      
      if (err.name === 'AbortError') {
        const timeoutErr = new Error(`Request timed out after ${CONFIG.REQUEST_TIMEOUT_MS / 1000}s. Please verify that the FastAPI backend at '${baseUrl}' is running and responsive.`);
        timeoutErr.status = 408;
        throw timeoutErr;
      }
      
      if (err.message && err.message.includes('Failed to fetch')) {
        const connErr = new Error(`Unable to connect to FastAPI backend at '${baseUrl}'. Ensure the service is active and CORS is allowed.`);
        connErr.status = 503;
        throw connErr;
      }

      throw err;
    }
  }

  /**
   * System health check & model diagnostics
   * GET /health
   */
  async getHealth() {
    return this.request('/health', { method: 'GET' });
  }

  /**
   * Stateless live movement prediction
   * POST /predict
   */
  async predict(payload) {
    return this.request('/predict', {
      method: 'POST',
      body: JSON.stringify({
        symbol: payload.symbol,
        reference_timestamp: payload.reference_timestamp,
        target_timestamp: payload.target_timestamp,
        model_type: payload.model_type || 'pooled'
      })
    });
  }

  /**
   * Stateful paper prediction creation
   * POST /paper/predict
   */
  async createPaperPrediction(payload) {
    return this.request('/paper/predict', {
      method: 'POST',
      body: JSON.stringify({
        symbol: payload.symbol,
        reference_timestamp: payload.reference_timestamp,
        target_timestamp: payload.target_timestamp,
        model_type: payload.model_type || 'pooled'
      })
    });
  }

  /**
   * Get filtered paper prediction history
   * GET /paper/predictions
   */
  async getPredictions(filters = {}) {
    const params = new URLSearchParams();
    if (filters.symbol) params.append('symbol', filters.symbol);
    if (filters.status) params.append('status', filters.status);
    if (filters.horizon) params.append('horizon', filters.horizon);
    if (filters.predicted_class) params.append('predicted_class', filters.predicted_class);
    if (filters.limit) params.append('limit', filters.limit);

    const query = params.toString();
    return this.request(`/paper/predictions${query ? '?' + query : ''}`, { method: 'GET' });
  }

  /**
   * Get single prediction detail
   * GET /paper/predictions/{prediction_id}
   */
  async getPrediction(predictionId) {
    return this.request(`/paper/predictions/${encodeURIComponent(predictionId)}`, { method: 'GET' });
  }

  /**
   * Get paper trading evaluation and benchmark metrics
   * GET /paper/performance
   */
  async getPerformance() {
    return this.request('/paper/performance', { method: 'GET' });
  }

  /**
   * Resolve a specific paper prediction
   * POST /paper/resolve/{prediction_id}
   */
  async resolvePrediction(predictionId) {
    return this.request(`/paper/resolve/${encodeURIComponent(predictionId)}`, { method: 'POST' });
  }

  /**
   * Batch resolve all eligible paper predictions
   * POST /paper/resolve
   */
  async resolveAll() {
    return this.request('/paper/resolve', { method: 'POST' });
  }
}

export const api = new ApiClient();
