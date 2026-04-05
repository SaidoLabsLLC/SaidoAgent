/**
 * Settings page: team management, API key display, model routing config.
 */

import { type FormEvent, useCallback, useEffect, useState } from "react";
import { apiGet, apiPost, getStoredToken } from "../api/client";
import type { Team, TeamMember } from "../types";

export default function Settings() {
  const [teams, setTeams] = useState<Team[]>([]);
  const [members, setMembers] = useState<TeamMember[]>([]);
  const [newTeamName, setNewTeamName] = useState("");
  const [newMemberEmail, setNewMemberEmail] = useState("");
  const [newMemberRole, setNewMemberRole] = useState("viewer");
  const [selectedTeam, setSelectedTeam] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // -----------------------------------------------------------------------
  // Load teams
  // -----------------------------------------------------------------------

  useEffect(() => {
    apiGet<Team[]>("/v1/teams")
      .then(setTeams)
      .catch(() => {
        /* User may not have team access yet */
      });
  }, []);

  // -----------------------------------------------------------------------
  // Create team
  // -----------------------------------------------------------------------

  const handleCreateTeam = useCallback(
    async (e: FormEvent) => {
      e.preventDefault();
      if (!newTeamName.trim()) return;
      setLoading(true);
      setError(null);
      try {
        const team = await apiPost<Team>("/v1/teams", {
          name: newTeamName.trim(),
        });
        setTeams((prev) => [...prev, team]);
        setNewTeamName("");
      } catch (err) {
        setError((err as Error).message);
      } finally {
        setLoading(false);
      }
    },
    [newTeamName],
  );

  // -----------------------------------------------------------------------
  // Add member (placeholder -- requires user_id, so this is a simplified UI)
  // -----------------------------------------------------------------------

  const handleAddMember = useCallback(
    async (e: FormEvent) => {
      e.preventDefault();
      if (!selectedTeam || !newMemberEmail.trim()) return;
      setLoading(true);
      setError(null);
      try {
        const member = await apiPost<TeamMember>(
          `/v1/teams/${selectedTeam}/members`,
          {
            user_id: newMemberEmail.trim(), // Simplified: UI sends user_id
            role: newMemberRole,
          },
        );
        setMembers((prev) => [...prev, member]);
        setNewMemberEmail("");
      } catch (err) {
        setError((err as Error).message);
      } finally {
        setLoading(false);
      }
    },
    [selectedTeam, newMemberEmail, newMemberRole],
  );

  return (
    <div className="mx-auto max-w-3xl space-y-8">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Settings</h1>
        <p className="mt-1 text-sm text-gray-500">
          Manage your team, API keys, and configuration.
        </p>
      </div>

      {error && (
        <div className="rounded-md bg-red-50 p-3 text-sm text-red-700" role="alert">
          {error}
        </div>
      )}

      {/* API Key display */}
      <section className="card">
        <h2 className="text-lg font-semibold text-gray-900">
          Authentication Token
        </h2>
        <p className="mt-1 text-sm text-gray-500">
          Your current session JWT for API access.
        </p>
        <div className="mt-3">
          <code className="block overflow-x-auto rounded bg-gray-100 p-3 text-xs text-gray-600">
            {getStoredToken()
              ? `${getStoredToken()!.slice(0, 30)}...`
              : "No token stored"}
          </code>
        </div>
      </section>

      {/* Team management */}
      <section className="card">
        <h2 className="text-lg font-semibold text-gray-900">Teams</h2>

        {teams.length > 0 ? (
          <ul className="mt-3 divide-y divide-gray-100">
            {teams.map((team) => (
              <li
                key={team.id}
                className="flex items-center justify-between py-3"
              >
                <div>
                  <p className="text-sm font-medium text-gray-900">
                    {team.name}
                  </p>
                  <p className="text-xs text-gray-400">
                    Role: {team.role || "member"} -- ID: {team.id}
                  </p>
                </div>
                <button
                  onClick={() => setSelectedTeam(team.id)}
                  className={`badge cursor-pointer ${
                    selectedTeam === team.id
                      ? "bg-brand-100 text-brand-700"
                      : "bg-gray-100 text-gray-500"
                  }`}
                >
                  {selectedTeam === team.id ? "Selected" : "Select"}
                </button>
              </li>
            ))}
          </ul>
        ) : (
          <p className="mt-3 text-sm text-gray-400">
            No teams yet. Create one below.
          </p>
        )}

        <form onSubmit={handleCreateTeam} className="mt-4 flex gap-2">
          <input
            type="text"
            value={newTeamName}
            onChange={(e) => setNewTeamName(e.target.value)}
            placeholder="New team name"
            className="input flex-1"
            aria-label="New team name"
          />
          <button type="submit" disabled={loading} className="btn-primary">
            Create Team
          </button>
        </form>
      </section>

      {/* Add member */}
      {selectedTeam && (
        <section className="card">
          <h2 className="text-lg font-semibold text-gray-900">
            Add Team Member
          </h2>
          <form
            onSubmit={handleAddMember}
            className="mt-3 flex flex-col gap-3 sm:flex-row"
          >
            <input
              type="text"
              value={newMemberEmail}
              onChange={(e) => setNewMemberEmail(e.target.value)}
              placeholder="User ID"
              className="input flex-1"
              aria-label="User ID to add"
            />
            <select
              value={newMemberRole}
              onChange={(e) => setNewMemberRole(e.target.value)}
              className="input w-auto"
              aria-label="Role for new member"
            >
              <option value="viewer">Viewer</option>
              <option value="editor">Editor</option>
              <option value="admin">Admin</option>
            </select>
            <button type="submit" disabled={loading} className="btn-primary">
              Add Member
            </button>
          </form>

          {members.length > 0 && (
            <ul className="mt-4 divide-y divide-gray-100">
              {members.map((m) => (
                <li key={m.user_id} className="flex items-center justify-between py-2">
                  <div>
                    <p className="text-sm text-gray-900">
                      {m.name || m.email || m.user_id}
                    </p>
                  </div>
                  <span className="badge bg-gray-100 text-gray-600">
                    {m.role}
                  </span>
                </li>
              ))}
            </ul>
          )}
        </section>
      )}

      {/* Model routing config (placeholder) */}
      <section className="card">
        <h2 className="text-lg font-semibold text-gray-900">Model Routing</h2>
        <p className="mt-1 text-sm text-gray-500">
          Configure which LLM provider handles queries. Use the{" "}
          <code className="rounded bg-gray-100 px-1 text-xs">/cloud</code>{" "}
          toggle in Chat to route queries to cloud providers on demand.
        </p>
        <div className="mt-3 rounded-md bg-gray-50 p-4 text-sm text-gray-600">
          Model routing is configured server-side via environment variables.
          Contact your administrator to change the default provider.
        </div>
      </section>
    </div>
  );
}
