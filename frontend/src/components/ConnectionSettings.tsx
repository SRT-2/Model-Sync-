import React, { useState } from 'react';
import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  Eye,
  EyeOff,
  Info,
  RefreshCw,
  Server,
  Shield,
  XCircle,
} from 'lucide-react';
import { useApp } from '../context/AppContext';

export const ConnectionSettings: React.FC = () => {
  const {
    config,
    updateConfig,
    health,
    connectionStatus,
    lastHealthCheckTime,
    availableModels,
    checkConnection,
  } = useApp();

  const [baseUrlInput, setBaseUrlInput] = useState(config.baseUrl);
  const [apiKeyInput, setApiKeyInput] = useState(config.routerApiKey);
  const [showToken, setShowToken] = useState(false);
  const [isTesting, setIsTesting] = useState(false);
  const [testResult, setTestResult] = useState<string | null>(null);

  const handleApply = async (e: React.FormEvent) => {
    e.preventDefault();
    updateConfig({
      baseUrl: baseUrlInput.trim(),
      routerApiKey: apiKeyInput.trim(),
    });
    setIsTesting(true);
    setTestResult(null);
    try {
      await checkConnection();
      setTestResult('Settings saved and connection test successful.');
    } catch (err: any) {
      setTestResult(`Connection test error: ${err.message}`);
    } finally {
      setIsTesting(false);
    }
  };

  const handleTestNow = async () => {
    setIsTesting(true);
    setTestResult(null);
    try {
      await checkConnection();
      setTestResult('Connection verified successfully.');
    } catch (err: any) {
      setTestResult(`Failed: ${err.message}`);
    } finally {
      setIsTesting(false);
    }
  };

  return (
    <div className="settings-container">
      {/* Page Header */}
      <div className="page-header">
        <div className="flex items-center gap-2">
          <Server className="text-blue-500" size={22} />
          <h1 className="page-title">Connection & Models Management</h1>
        </div>
        <p className="page-subtitle">
          Configure API connection endpoints, test router backend health, and view verified models.
        </p>
      </div>

      {/* Connection Form Card */}
      <section className="settings-card">
        <h2 className="card-heading flex items-center gap-2">
          <Shield size={17} className="text-blue-400" />
          <span>Router Backend Connection</span>
        </h2>

        <form onSubmit={handleApply} className="space-y-4">
          {/* Base URL */}
          <div>
            <label className="input-label" htmlFor="base-url-input">
              Backend Base URL
            </label>
            <input
              id="base-url-input"
              type="text"
              value={baseUrlInput}
              onChange={(e) => setBaseUrlInput(e.target.value)}
              placeholder="http://localhost:8000"
              className="input-field font-mono text-xs"
              required
            />
            <p className="input-helper">
              The address where the FastAPI router is listening (defaults to <code>http://localhost:8000</code>).
            </p>
          </div>

          {/* Router Inbound API Key (Bearer Token) */}
          <div>
            <div className="flex items-center justify-between mb-1">
              <label className="input-label mb-0" htmlFor="router-api-key-input">
                Router Inbound API Key (Optional)
              </label>
              <button
                type="button"
                onClick={() => setShowToken(!showToken)}
                className="action-btn text-xs"
                aria-label={showToken ? 'Hide API key' : 'Show API key'}
              >
                {showToken ? <EyeOff size={13} /> : <Eye size={13} />}
                <span>{showToken ? 'Hide' : 'Show'}</span>
              </button>
            </div>
            <div className="relative">
              <input
                id="router-api-key-input"
                type={showToken ? 'text' : 'password'}
                value={apiKeyInput}
                onChange={(e) => setApiKeyInput(e.target.value)}
                placeholder="Leave blank if ROUTER_API_KEY is not enabled on backend"
                className="input-field font-mono text-xs pr-10"
              />
            </div>

            {/* Security Explanation Alert */}
            <div className="security-notice mt-3">
              <Info size={16} className="text-blue-400 shrink-0 mt-0.5" />
              <div className="text-xs leading-relaxed">
                <strong>Security Notice: </strong>
                <span>
                  This field is for the <em>inbound Bearer token</em> required if your router backend has <code>ROUTER_API_KEY</code> set in <code>.env</code>.
                  <br />
                  <strong>Do NOT</strong> enter your Together AI or OpenRouter API key here! Upstream keys are safely kept on the backend server and never exposed to browser bundles or network traces.
                </span>
              </div>
            </div>
          </div>

          {/* Actions */}
          <div className="flex items-center gap-3 pt-2">
            <button
              type="submit"
              disabled={isTesting}
              className="btn-primary text-xs"
            >
              {isTesting ? <RefreshCw size={13} className="animate-spin" /> : <CheckCircle2 size={13} />}
              <span>Save & Connect</span>
            </button>

            <button
              type="button"
              onClick={handleTestNow}
              disabled={isTesting}
              className="btn-secondary text-xs"
            >
              <Activity size={13} />
              <span>Test /health Endpoint</span>
            </button>
          </div>

          {testResult && (
            <div className={`p-3 rounded text-xs mt-3 flex items-center gap-2 ${
              testResult.includes('error') || testResult.includes('Failed')
                ? 'error-banner'
                : 'bg-emerald-500/10 text-emerald-300 border border-emerald-500/30'
            }`}>
              {testResult.includes('error') || testResult.includes('Failed') ? (
                <AlertTriangle size={15} />
              ) : (
                <CheckCircle2 size={15} className="text-emerald-400" />
              )}
              <span className="font-medium">{testResult}</span>
            </div>
          )}
        </form>
      </section>

      {/* Backend Status & Health Diagnostic */}
      <section className="settings-card">
        <h2 className="card-heading flex items-center gap-2">
          <Activity size={17} className="text-emerald-400" />
          <span>Backend Health & Runtime Status</span>
        </h2>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <div className="diagnostic-panel">
            <span className="diagnostic-label">
              Connection State
            </span>
            <div className="flex items-center gap-2">
              {connectionStatus === 'online' && (
                <>
                  <CheckCircle2 size={18} className="text-emerald-400" />
                  <span className="diagnostic-value text-emerald-400">Online & Ready</span>
                </>
              )}
              {connectionStatus === 'unauthorized' && (
                <>
                  <AlertTriangle size={18} className="text-amber-400" />
                  <span className="diagnostic-value text-amber-400">401 Unauthorized (Key Required)</span>
                </>
              )}
              {connectionStatus === 'offline' && (
                <>
                  <XCircle size={18} className="text-rose-400" />
                  <span className="diagnostic-value text-rose-400">Offline (Unreachable)</span>
                </>
              )}
              {connectionStatus === 'checking' && (
                <>
                  <Activity size={18} className="animate-spin text-blue-400" />
                  <span className="diagnostic-value text-blue-400">Pinging server...</span>
                </>
              )}
            </div>
            {lastHealthCheckTime && (
              <span className="diagnostic-sub">
                Last checked: {new Date(lastHealthCheckTime).toLocaleTimeString()}
              </span>
            )}
          </div>

          <div className="diagnostic-panel">
            <span className="diagnostic-label">
              Upstream Provider Mode
            </span>
            <div className="diagnostic-value capitalize text-blue-400">
              {health?.provider || 'Unknown'}
            </div>
            <span className="diagnostic-sub">
              Server Version: {health?.version || 'N/A'} • Auth: {health?.auth_enabled ? 'Enabled' : 'Disabled'}
            </span>
          </div>
        </div>

        {health && (
          <div className="mt-4">
            <span className="text-xs font-bold text-zinc-200 block mb-2">
              Configured Downstream Models (reported by /health)
            </span>
            <div className="json-dump-box">
              <pre>{JSON.stringify(health.configured_models, null, 2)}</pre>
            </div>
          </div>
        )}
      </section>

      {/* Verified Models Table */}
      <section className="settings-card">
        <h2 className="card-heading flex items-center gap-2">
          <Server size={17} className="text-purple-400" />
          <span>Available Virtual Models (/v1/models)</span>
        </h2>

        {connectionStatus === 'offline' ? (
          <div className="error-banner">
            <AlertTriangle size={15} />
            <span>Backend is offline. Cannot query <code>/v1/models</code>. Start the backend with <code>python router.py</code>.</span>
          </div>
        ) : connectionStatus === 'unauthorized' ? (
          <div className="p-4 rounded bg-amber-500/10 border border-amber-500/30 text-amber-300 text-xs">
            401 Unauthorized. Please enter the valid Router API Key in the field above to query models.
          </div>
        ) : availableModels.length === 0 ? (
          <div className="text-xs text-zinc-400 py-3">
            No models returned from <code>/v1/models</code>.
          </div>
        ) : (
          <div className="table-responsive">
            <table className="insights-table">
              <thead>
                <tr>
                  <th>Model ID</th>
                  <th>Object</th>
                  <th>Owner</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {availableModels.map((model) => (
                  <tr key={model.id}>
                    <td className="font-mono text-xs font-bold text-blue-400">
                      {model.id}
                    </td>
                    <td className="text-xs text-zinc-300">{model.object}</td>
                    <td className="text-xs text-zinc-300">{model.owned_by}</td>
                    <td>
                      <span className="inline-flex items-center gap-1.5 text-xs text-emerald-400 font-semibold">
                        <CheckCircle2 size={13} />
                        <span>Verified Active</span>
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
};
