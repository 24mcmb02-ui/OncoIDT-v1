import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { api } from '@/api/client'

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface ClinicalRule {
  id: string
  name: string
  description: string
  enabled: boolean
  rule_type: 'hard' | 'soft'
  yaml_config: string
}

interface ModelVersion {
  version: string
  run_id: string
  status: 'staging' | 'production' | 'archived' | 'candidate'
  auroc: number | null
  auprc: number | null
  brier_score: number | null
  ece: number | null
  training_dataset_version: string
  trained_at: string
  approver: string | null
}

interface UserRecord {
  user_id: string
  username: string
  role: string
  ward_ids: string[]
  active: boolean
}

// ---------------------------------------------------------------------------
// YAML rule editor
// ---------------------------------------------------------------------------

function RuleEditor() {
  const qc = useQueryClient()
  const [editingId, setEditingId] = useState<string | null>(null)
  const [draftYaml, setDraftYaml] = useState('')
  const [yamlError, setYamlError] = useState<string | null>(null)

  const { data: rules = [], isLoading } = useQuery<ClinicalRule[]>({
    queryKey: ['admin-rules'],
    queryFn: () => api.get<ClinicalRule[]>('/admin/rules'),
  })

  const updateMutation = useMutation({
    mutationFn: ({ id, yaml_config }: { id: string; yaml_config: string }) =>
      api.put(`/admin/rules/${id}`, { yaml_config }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['admin-rules'] })
      setEditingId(null)
      setYamlError(null)
    },
    onError: (err: Error) => setYamlError(err.message),
  })

  function startEdit(rule: ClinicalRule) {
    setEditingId(rule.id)
    setDraftYaml(rule.yaml_config)
    setYamlError(null)
  }

  function validateYaml(yaml: string): boolean {
    // Basic client-side validation: check for required keys
    const hasName = /name\s*:/.test(yaml)
    const hasThreshold = /threshold\s*:/.test(yaml)
    if (!hasName || !hasThreshold) {
      setYamlError('YAML must contain "name" and "threshold" keys.')
      return false
    }
    setYamlError(null)
    return true
  }

  function handleSave(id: string) {
    if (!validateYaml(draftYaml)) return
    updateMutation.mutate({ id, yaml_config: draftYaml })
  }

  if (isLoading) {
    return <div className="space-y-2">{[...Array(3)].map((_, i) => <div key={i} className="h-12 animate-pulse rounded bg-gray-100" />)}</div>
  }

  return (
    <div className="space-y-3">
      {rules.map((rule) => (
        <div key={rule.id} className="rounded-lg border border-gray-200 bg-white p-3">
          <div className="flex items-start justify-between gap-2">
            <div>
              <div className="flex items-center gap-2">
                <span className="text-sm font-semibold text-gray-800">{rule.name}</span>
                <span className={`rounded-full px-1.5 py-0.5 text-xs font-medium ${rule.rule_type === 'hard' ? 'bg-red-100 text-red-700' : 'bg-blue-100 text-blue-700'}`}>
                  {rule.rule_type}
                </span>
                <span className={`rounded-full px-1.5 py-0.5 text-xs ${rule.enabled ? 'bg-green-100 text-green-700' : 'bg-gray-100 text-gray-500'}`}>
                  {rule.enabled ? 'enabled' : 'disabled'}
                </span>
              </div>
              <p className="mt-0.5 text-xs text-gray-500">{rule.description}</p>
            </div>
            {editingId !== rule.id && (
              <button
                onClick={() => startEdit(rule)}
                className="rounded border border-gray-300 px-2 py-1 text-xs text-gray-600 hover:bg-gray-50"
                disabled={rule.rule_type === 'hard'}
                title={rule.rule_type === 'hard' ? 'Hard rules cannot be modified' : 'Edit rule'}
              >
                Edit
              </button>
            )}
          </div>

          {editingId === rule.id && (
            <div className="mt-3 space-y-2">
              <textarea
                value={draftYaml}
                onChange={(e) => {
                  setDraftYaml(e.target.value)
                  setYamlError(null)
                }}
                rows={8}
                className="w-full rounded border border-gray-300 bg-gray-50 px-2 py-1.5 font-mono text-xs"
                aria-label="Rule YAML configuration"
                spellCheck={false}
              />
              {yamlError && (
                <p className="text-xs text-red-500">{yamlError}</p>
              )}
              <div className="flex gap-2">
                <button
                  onClick={() => handleSave(rule.id)}
                  disabled={updateMutation.isPending}
                  className="rounded bg-indigo-600 px-3 py-1 text-xs font-semibold text-white hover:bg-indigo-700 disabled:opacity-50"
                >
                  {updateMutation.isPending ? 'Saving…' : 'Save'}
                </button>
                <button
                  onClick={() => { setEditingId(null); setYamlError(null) }}
                  className="rounded border border-gray-300 px-3 py-1 text-xs text-gray-600 hover:bg-gray-50"
                >
                  Cancel
                </button>
              </div>
            </div>
          )}
        </div>
      ))}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Model registry
// ---------------------------------------------------------------------------

function ModelRegistry({ isAdmin }: { isAdmin: boolean }) {
  const qc = useQueryClient()

  const { data: models = [], isLoading } = useQuery<ModelVersion[]>({
    queryKey: ['admin-models'],
    queryFn: () => api.get<ModelVersion[]>('/training/models'),
  })

  const promoteMutation = useMutation({
    mutationFn: (version: string) =>
      api.post(`/admin/models/promote/${version}`, {}),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['admin-models'] }),
  })

  const statusColors: Record<string, string> = {
    production: 'bg-green-100 text-green-700',
    staging: 'bg-blue-100 text-blue-700',
    candidate: 'bg-yellow-100 text-yellow-700',
    archived: 'bg-gray-100 text-gray-500',
  }

  if (isLoading) {
    return <div className="h-32 animate-pulse rounded bg-gray-100" />
  }

  return (
    <div className="overflow-x-auto rounded-lg border border-gray-200">
      <table className="min-w-full text-xs">
        <thead className="bg-gray-50">
          <tr>
            {['Version', 'Status', 'AUROC', 'AUPRC', 'Brier', 'ECE', 'Dataset', 'Trained', 'Approver', ''].map((h) => (
              <th key={h} className="px-3 py-2 text-left font-semibold text-gray-600">{h}</th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y divide-gray-100 bg-white">
          {models.length === 0 && (
            <tr>
              <td colSpan={10} className="py-6 text-center text-gray-400">No model versions found.</td>
            </tr>
          )}
          {models.map((m) => (
            <tr key={m.version} className="hover:bg-gray-50">
              <td className="px-3 py-2 font-mono text-gray-800">{m.version}</td>
              <td className="px-3 py-2">
                <span className={`rounded-full px-1.5 py-0.5 font-medium ${statusColors[m.status] ?? ''}`}>
                  {m.status}
                </span>
              </td>
              <td className="px-3 py-2 text-gray-700">{m.auroc?.toFixed(3) ?? '—'}</td>
              <td className="px-3 py-2 text-gray-700">{m.auprc?.toFixed(3) ?? '—'}</td>
              <td className="px-3 py-2 text-gray-700">{m.brier_score?.toFixed(3) ?? '—'}</td>
              <td className="px-3 py-2 text-gray-700">{m.ece?.toFixed(3) ?? '—'}</td>
              <td className="px-3 py-2 text-gray-500">{m.training_dataset_version}</td>
              <td className="px-3 py-2 text-gray-500">{new Date(m.trained_at).toLocaleDateString()}</td>
              <td className="px-3 py-2 text-gray-500">{m.approver ?? '—'}</td>
              <td className="px-3 py-2">
                {isAdmin && m.status === 'candidate' && (
                  <button
                    onClick={() => promoteMutation.mutate(m.version)}
                    disabled={promoteMutation.isPending}
                    className="rounded bg-green-600 px-2 py-0.5 text-xs font-semibold text-white hover:bg-green-700 disabled:opacity-50"
                  >
                    Promote
                  </button>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

// ---------------------------------------------------------------------------
// User management
// ---------------------------------------------------------------------------

const ROLES = ['Clinician', 'Charge_Nurse', 'Infection_Control_Officer', 'Research_Analyst', 'System_Administrator', 'Audit_Reviewer']

function UserManagement() {
  const qc = useQueryClient()
  const [editingId, setEditingId] = useState<string | null>(null)
  const [draftRole, setDraftRole] = useState('')

  const { data: users = [], isLoading } = useQuery<UserRecord[]>({
    queryKey: ['admin-users'],
    queryFn: () => api.get<UserRecord[]>('/admin/users'),
  })

  const updateMutation = useMutation({
    mutationFn: ({ userId, role }: { userId: string; role: string }) =>
      api.put(`/admin/users/${userId}`, { role }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['admin-users'] })
      setEditingId(null)
    },
  })

  if (isLoading) {
    return <div className="h-32 animate-pulse rounded bg-gray-100" />
  }

  return (
    <div className="overflow-x-auto rounded-lg border border-gray-200">
      <table className="min-w-full text-xs">
        <thead className="bg-gray-50">
          <tr>
            {['Username', 'Role', 'Wards', 'Status', ''].map((h) => (
              <th key={h} className="px-3 py-2 text-left font-semibold text-gray-600">{h}</th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y divide-gray-100 bg-white">
          {users.length === 0 && (
            <tr>
              <td colSpan={5} className="py-6 text-center text-gray-400">No users found.</td>
            </tr>
          )}
          {users.map((u) => (
            <tr key={u.user_id} className="hover:bg-gray-50">
              <td className="px-3 py-2 font-medium text-gray-800">{u.username}</td>
              <td className="px-3 py-2">
                {editingId === u.user_id ? (
                  <select
                    value={draftRole}
                    onChange={(e) => setDraftRole(e.target.value)}
                    className="rounded border border-gray-300 px-1 py-0.5 text-xs"
                    aria-label="User role"
                  >
                    {ROLES.map((r) => <option key={r} value={r}>{r}</option>)}
                  </select>
                ) : (
                  <span className="rounded bg-gray-100 px-1.5 py-0.5 text-gray-700">{u.role}</span>
                )}
              </td>
              <td className="px-3 py-2 text-gray-500">{u.ward_ids.join(', ') || '—'}</td>
              <td className="px-3 py-2">
                <span className={`rounded-full px-1.5 py-0.5 font-medium ${u.active ? 'bg-green-100 text-green-700' : 'bg-gray-100 text-gray-500'}`}>
                  {u.active ? 'active' : 'inactive'}
                </span>
              </td>
              <td className="px-3 py-2">
                {editingId === u.user_id ? (
                  <div className="flex gap-1">
                    <button
                      onClick={() => updateMutation.mutate({ userId: u.user_id, role: draftRole })}
                      disabled={updateMutation.isPending}
                      className="rounded bg-indigo-600 px-2 py-0.5 text-xs text-white hover:bg-indigo-700 disabled:opacity-50"
                    >
                      Save
                    </button>
                    <button
                      onClick={() => setEditingId(null)}
                      className="rounded border border-gray-300 px-2 py-0.5 text-xs text-gray-600"
                    >
                      Cancel
                    </button>
                  </div>
                ) : (
                  <button
                    onClick={() => { setEditingId(u.user_id); setDraftRole(u.role) }}
                    className="rounded border border-gray-300 px-2 py-0.5 text-xs text-gray-600 hover:bg-gray-50"
                  >
                    Edit
                  </button>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Main page
// ---------------------------------------------------------------------------

type AdminTab = 'rules' | 'models' | 'users'

// In production this comes from the JWT payload in auth context.
// For now we read from localStorage as a placeholder.
function useIsAdmin(): boolean {
  try {
    const token = localStorage.getItem('onco_idt_token')
    if (!token) return false
    const payload = JSON.parse(atob(token.split('.')[1]))
    return payload?.role === 'System_Administrator'
  } catch {
    return false
  }
}

export function Admin() {
  const [tab, setTab] = useState<AdminTab>('rules')
  const isAdmin = useIsAdmin()

  const tabs: { key: AdminTab; label: string }[] = [
    { key: 'rules', label: 'Clinical Rules' },
    { key: 'models', label: 'Model Registry' },
    ...(isAdmin ? [{ key: 'users' as AdminTab, label: 'User Management' }] : []),
  ]

  return (
    <div className="min-h-screen bg-gray-50 p-4">
      <div className="mx-auto max-w-screen-xl space-y-4">

        <div>
          <h1 className="text-lg font-bold text-gray-900">Admin</h1>
          <p className="text-sm text-gray-500">
            {isAdmin ? 'System Administrator' : 'Limited access — some panels require System_Administrator role'}
          </p>
        </div>

        {/* Tab bar */}
        <div className="flex gap-1 rounded-lg bg-white p-1 shadow-sm border border-gray-200 w-fit">
          {tabs.map((t) => (
            <button
              key={t.key}
              onClick={() => setTab(t.key)}
              className={`rounded px-4 py-1.5 text-sm font-medium transition-colors ${
                tab === t.key
                  ? 'bg-indigo-600 text-white'
                  : 'text-gray-600 hover:text-gray-900'
              }`}
              aria-pressed={tab === t.key}
            >
              {t.label}
            </button>
          ))}
        </div>

        {/* Tab content */}
        {tab === 'rules' && (
          <div className="space-y-2">
            <p className="text-xs text-gray-500">
              Soft rules can be edited. Hard rules (SIRS, NEWS2, ANC+fever) are always enforced and cannot be disabled.
            </p>
            <RuleEditor />
          </div>
        )}

        {tab === 'models' && (
          <div className="space-y-2">
            <p className="text-xs text-gray-500">
              Promote a candidate model to production. Requires System_Administrator role.
            </p>
            <ModelRegistry isAdmin={isAdmin} />
          </div>
        )}

        {tab === 'users' && isAdmin && (
          <div className="space-y-2">
            <p className="text-xs text-gray-500">
              Manage user roles and ward access.
            </p>
            <UserManagement />
          </div>
        )}

        {tab === 'users' && !isAdmin && (
          <div className="flex h-32 items-center justify-center rounded-lg border border-gray-200 bg-white text-gray-400">
            System_Administrator role required.
          </div>
        )}

      </div>
    </div>
  )
}
