import React from 'react';
import {
  ArrowRight,
  CheckCircle2,
  Code2,
  Cpu,
  Layers,
  Route,
  Server,
  Sparkles,
  Zap,
} from 'lucide-react';
import { useApp } from '../context/AppContext';

export const RoutingInsights: React.FC = () => {
  const { health, sessionHistory } = useApp();

  // Calculate session category distribution from real completed requests
  const totalRequests = sessionHistory.length;
  const codingCount = sessionHistory.filter((s) => s.category === 'CODING').length;
  const testingCount = sessionHistory.filter((s) => s.category === 'UNIT_TESTING').length;
  const reasoningCount = sessionHistory.filter((s) => s.category === 'REASONING').length;
  const unclassifiedCount = sessionHistory.filter(
    (s) => !s.category || s.category === 'UNKNOWN'
  ).length;

  return (
    <div className="insights-container">
      {/* Page Title */}
      <div className="page-header">
        <div className="flex items-center gap-2">
          <Route className="text-blue-500" size={22} />
          <h1 className="page-title">Routing Insights & Architecture</h1>
        </div>
        <p className="page-subtitle">
          Dynamic classification pipeline and active downstream target models reported by the router backend.
        </p>
      </div>

      {/* Visual Pipeline Flow */}
      <section className="insights-card">
        <h2 className="card-heading flex items-center gap-2">
          <Zap size={17} className="text-amber-400" />
          <span>Real-time Routing Pipeline</span>
        </h2>
        <div className="pipeline-flow">
          <div className="pipeline-node">
            <div className="node-icon bg-blue-500/10 text-blue-400 border border-blue-500/30">
              <Layers size={20} />
            </div>
            <span className="node-title">Client Prompt</span>
            <span className="node-sub">POST /v1/chat/completions</span>
          </div>

          <ArrowRight className="pipeline-arrow text-zinc-400 hidden sm:block" size={20} />

          <div className="pipeline-node">
            <div className="node-icon bg-amber-500/10 text-amber-400 border border-amber-500/30">
              <Cpu size={20} />
            </div>
            <span className="node-title">Classifier Model</span>
            <span className="node-sub font-mono truncate max-w-[150px]" title={health?.configured_models.classifier}>
              {health?.configured_models.classifier || 'Configured via .env'}
            </span>
          </div>

          <ArrowRight className="pipeline-arrow text-zinc-400 hidden sm:block" size={20} />

          <div className="pipeline-node flex-1 min-w-[200px]">
            <div className="node-icon bg-purple-500/10 text-purple-400 border border-purple-500/30">
              <Route size={20} />
            </div>
            <span className="node-title">Dynamic Target Selection</span>
            <div className="grid grid-cols-3 gap-2 mt-2 w-full">
              <div className="target-pill-small border-blue-500/30 text-blue-300 bg-blue-500/10">
                <span>CODING</span>
              </div>
              <div className="target-pill-small border-emerald-500/30 text-emerald-300 bg-emerald-500/10">
                <span>TESTING</span>
              </div>
              <div className="target-pill-small border-purple-500/30 text-purple-300 bg-purple-500/10">
                <span>REASONING</span>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Active Downstream Target Table */}
      <section className="insights-card">
        <h2 className="card-heading flex items-center gap-2">
          <Server size={17} className="text-blue-400" />
          <span>Active Backend Routing Table</span>
        </h2>
        <div className="table-responsive">
          <table className="insights-table">
            <thead>
              <tr>
                <th>Category</th>
                <th>Trigger Criteria</th>
                <th>Configured Downstream Model</th>
                <th>Provider Mode</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>
                  <span className="routing-badge badge-coding">
                    <Code2 size={13} />
                    <span>CODING</span>
                  </span>
                </td>
                <td className="text-xs text-zinc-200">
                  Implementation, bug fixing, algorithm design, syntax questions
                </td>
                <td className="font-mono text-xs font-semibold text-blue-400">
                  {health?.configured_models.coding || 'Configured in backend'}
                </td>
                <td className="text-xs text-zinc-300 capitalize font-medium">
                  {health?.provider || 'OpenRouter / Together'}
                </td>
              </tr>
              <tr>
                <td>
                  <span className="routing-badge badge-testing">
                    <CheckCircle2 size={13} />
                    <span>UNIT_TESTING</span>
                  </span>
                </td>
                <td className="text-xs text-zinc-200">
                  Test generation, pytest, assertion writing, mock fixtures, edge cases
                </td>
                <td className="font-mono text-xs font-semibold text-emerald-400">
                  {health?.configured_models.unit_testing || 'Configured in backend'}
                </td>
                <td className="text-xs text-zinc-300 capitalize font-medium">
                  {health?.provider || 'OpenRouter / Together'}
                </td>
              </tr>
              <tr>
                <td>
                  <span className="routing-badge badge-reasoning">
                    <Cpu size={13} />
                    <span>REASONING</span>
                  </span>
                </td>
                <td className="text-xs text-zinc-200">
                  Architecture design, logical trade-offs, system architecture, open-ended analysis
                </td>
                <td className="font-mono text-xs font-semibold text-purple-400">
                  {health?.configured_models.reasoning || 'Configured in backend'}
                </td>
                <td className="text-xs text-zinc-300 capitalize font-medium">
                  {health?.provider || 'OpenRouter / Together'}
                </td>
              </tr>
              <tr>
                <td>
                  <span className="routing-badge bg-amber-500/10 text-amber-300 border border-amber-500/30">
                    <Sparkles size={13} />
                    <span>CLASSIFIER</span>
                  </span>
                </td>
                <td className="text-xs text-zinc-200">
                  Lightweight prompt analyzer that determines task classification
                </td>
                <td className="font-mono text-xs font-semibold text-amber-300">
                  {health?.configured_models.classifier || 'Configured in backend'}
                </td>
                <td className="text-xs text-zinc-300 capitalize font-medium">
                  {health?.provider || 'OpenRouter / Together'}
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>

      {/* Session Distribution Stats */}
      <section className="insights-card">
        <h2 className="card-heading flex items-center gap-2">
          <Layers size={17} className="text-purple-400" />
          <span>Current Session Route Distribution</span>
        </h2>

        {totalRequests === 0 ? (
          <div className="text-center py-8 text-zinc-400 text-xs">
            No requests have been executed in this session yet. Run a prompt in the Playground to see verified classification metrics.
          </div>
        ) : (
          <div className="space-y-4">
            <div className="grid grid-cols-1 sm:grid-cols-4 gap-3">
              <div className="stat-card">
                <span className="stat-label">Total Requests</span>
                <span className="stat-value">{totalRequests}</span>
              </div>
              <div className="stat-card border-blue-500/30">
                <span className="stat-label text-blue-400">Coding Tasks</span>
                <span className="stat-value text-blue-400">
                  {codingCount} ({((codingCount / totalRequests) * 100).toFixed(0)}%)
                </span>
              </div>
              <div className="stat-card border-emerald-500/30">
                <span className="stat-label text-emerald-400">Testing Tasks</span>
                <span className="stat-value text-emerald-400">
                  {testingCount} ({((testingCount / totalRequests) * 100).toFixed(0)}%)
                </span>
              </div>
              <div className="stat-card border-purple-500/30">
                <span className="stat-label text-purple-400">Reasoning Tasks</span>
                <span className="stat-value text-purple-400">
                  {reasoningCount} ({((reasoningCount / totalRequests) * 100).toFixed(0)}%)
                </span>
              </div>
            </div>

            {/* Distribution Bar */}
            <div className="distribution-bar-track">
              {codingCount > 0 && (
                <div
                  style={{ width: `${(codingCount / totalRequests) * 100}%` }}
                  className="bar-segment bg-blue-500"
                  title={`Coding: ${codingCount}`}
                />
              )}
              {testingCount > 0 && (
                <div
                  style={{ width: `${(testingCount / totalRequests) * 100}%` }}
                  className="bar-segment bg-emerald-500"
                  title={`Unit Testing: ${testingCount}`}
                />
              )}
              {reasoningCount > 0 && (
                <div
                  style={{ width: `${(reasoningCount / totalRequests) * 100}%` }}
                  className="bar-segment bg-purple-500"
                  title={`Reasoning: ${reasoningCount}`}
                />
              )}
              {unclassifiedCount > 0 && (
                <div
                  style={{ width: `${(unclassifiedCount / totalRequests) * 100}%` }}
                  className="bar-segment bg-zinc-600"
                  title={`Unclassified: ${unclassifiedCount}`}
                />
              )}
            </div>
          </div>
        )}
      </section>
    </div>
  );
};
