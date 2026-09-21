/**
 * Ziro Predict - Main Application Router & Controller
 */

import { api } from './api.js';
import { renderDashboard } from './views/dashboard.js';
import { renderLivePrediction } from './views/live-prediction.js';
import { renderHistory } from './views/history.js';
import { renderPerformance } from './views/performance.js';
import { renderModelInfo } from './views/model-info.js';
import { renderAbout } from './views/about.js';

class AppRouter {
  constructor() {
    this.currentView = null;
    this.container = document.getElementById('view-container');
    this.pageTitle = document.getElementById('page-title');
    this.pageCaption = document.getElementById('page-caption');
    this.healthIndicator = document.getElementById('sidebar-health-indicator');
    this.healthText = document.getElementById('sidebar-health-text');

    this.routes = {
      'dashboard': {
        title: 'Overview Dashboard',
        caption: 'NSE Intraday Movement Prediction & Paper Evaluation Platform',
        render: (params) => renderDashboard(this.container, this)
      },
      'predict': {
        title: 'Live Movement Prediction',
        caption: 'Stateless real-time directional movement classification',
        render: (params) => renderLivePrediction(this.container, params)
      },
      'history': {
        title: 'Prediction History',
        caption: 'Audit log of stateful paper predictions and forward settlements',
        render: (params) => renderHistory(this.container)
      },
      'performance': {
        title: 'Performance Dashboard',
        caption: 'Forward paper evaluation metrics vs Phase 5 historical test benchmark',
        render: (params) => renderPerformance(this.container)
      },
      'model-info': {
        title: 'Model Information & Governance',
        caption: 'Production model architecture, 26 features, and target definitions',
        render: (params) => renderModelInfo(this.container)
      },
      'about': {
        title: 'About Project & Settings',
        caption: 'System architecture, API reference, and backend URL configuration',
        render: (params) => renderAbout(this.container)
      }
    };

    this.init();
  }

  init() {
    // Navigation item click handlers
    document.querySelectorAll('[data-view]').forEach(link => {
      link.addEventListener('click', (e) => {
        e.preventDefault();
        const view = link.getAttribute('data-view');
        this.navigate(view);
      });
    });

    // Handle initial route based on hash or default to dashboard
    const initialHash = window.location.hash.replace(/^#\/?/, '').trim();
    const initialView = this.routes[initialHash] ? initialHash : 'dashboard';
    this.navigate(initialView);

    // Initial and periodic health check
    this.pollHealth();
    setInterval(() => this.pollHealth(), 30000);
  }

  navigate(viewName, params = {}) {
    const route = this.routes[viewName] || this.routes['dashboard'];
    this.currentView = viewName;

    // Update active state in sidebar
    document.querySelectorAll('.nav-item').forEach(item => {
      if (item.getAttribute('data-view') === viewName) {
        item.classList.add('active');
      } else {
        item.classList.remove('active');
      }
    });

    // Update top header
    if (this.pageTitle) this.pageTitle.textContent = route.title;
    if (this.pageCaption) this.pageCaption.textContent = route.caption;

    // Update URL hash
    window.location.hash = `#/${viewName}`;

    // Render view
    this.container.scrollTop = 0;
    route.render(params);
  }

  async pollHealth() {
    try {
      const res = await api.getHealth();
      if (res && res.status === 'HEALTHY') {
        if (this.healthIndicator) {
          this.healthIndicator.className = 'status-indicator';
        }
        if (this.healthText) {
          this.healthText.textContent = `FastAPI: HEALTHY (${res.model_version})`;
        }
      }
    } catch (_) {
      if (this.healthIndicator) {
        this.healthIndicator.className = 'status-indicator offline';
      }
      if (this.healthText) {
        this.healthText.textContent = 'FastAPI: Disconnected';
      }
    }
  }
}

// Bootstrap on DOM content loaded
document.addEventListener('DOMContentLoaded', () => {
  window.app = new AppRouter();
});
