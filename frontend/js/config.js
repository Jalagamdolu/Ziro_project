/**
 * Ziro Predict - Configuration & Constants
 */

export const CONFIG = {
  // Default to local FastAPI server, or window override if provided
  DEFAULT_API_BASE_URL: window.API_BASE_URL || 'http://127.0.0.1:8000',
  
  // Storage key for user-configured API URL
  STORAGE_KEY_API_URL: 'ziro_api_base_url',

  // Request timeout in milliseconds
  REQUEST_TIMEOUT_MS: 10000,

  // Canonical NSE Liquid Universe Presets for Autocomplete
  POPULAR_SYMBOLS: [
    'RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK',
    '360ONE', 'ZOMATO', '20MICRONS', 'ZYDUSWELL', 'WIPRO',
    'SBIN', 'BHARTIARTL', 'KOTAKBANK', 'LT', 'AXISBANK'
  ],

  // Model Horizon Bounds
  MIN_HORIZON_SESSIONS: 1,
  MAX_HORIZON_SESSIONS: 7,

  // Market Hours
  MARKET_OPEN_TIME: '09:15',
  DEFAULT_REF_TIME: '10:30',

  // Terminal Dataset Boundaries
  MAX_HISTORICAL_DATE: '2026-09-10',
  MIN_HISTORICAL_DATE: '2026-08-10'
};

/**
 * Gets the active API Base URL from localStorage or default
 */
export function getApiBaseUrl() {
  return localStorage.getItem(CONFIG.STORAGE_KEY_API_URL) || CONFIG.DEFAULT_API_BASE_URL;
}

/**
 * Updates the active API Base URL
 */
export function setApiBaseUrl(url) {
  const cleaned = url.trim().replace(/\/+$/, '');
  localStorage.setItem(CONFIG.STORAGE_KEY_API_URL, cleaned);
  return cleaned;
}
