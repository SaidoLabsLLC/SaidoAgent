/**
 * Authentication page with email/password login and registration.
 */

import { type FormEvent, useState } from "react";
import { apiPost } from "../api/client";
import type { LoginRequest, RegisterRequest, RegisterResponse } from "../types";

interface LoginProps {
  onLogin: (credentials: LoginRequest) => Promise<void>;
}

export default function Login({ onLogin }: LoginProps) {
  const [mode, setMode] = useState<"login" | "register">("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [name, setName] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [success, setSuccess] = useState<string | null>(null);

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    setError(null);
    setSuccess(null);
    setLoading(true);

    try {
      if (mode === "register") {
        const body: RegisterRequest = { email, name, password };
        await apiPost<RegisterResponse>("/v1/auth/register", body);
        setSuccess("Account created. You can now sign in.");
        setMode("login");
      } else {
        await onLogin({ email, password });
      }
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="flex min-h-screen items-center justify-center bg-gray-50 px-4">
      <div className="w-full max-w-sm">
        <div className="mb-8 text-center">
          <h1 className="text-3xl font-bold text-brand-700">Saido Agent</h1>
          <p className="mt-2 text-sm text-gray-500">
            Knowledge-compounding AI agent
          </p>
        </div>

        <div className="card">
          <h2 className="mb-6 text-center text-lg font-semibold text-gray-900">
            {mode === "login" ? "Sign in to your account" : "Create an account"}
          </h2>

          {error && (
            <div
              className="mb-4 rounded-md bg-red-50 p-3 text-sm text-red-700"
              role="alert"
            >
              {error}
            </div>
          )}

          {success && (
            <div
              className="mb-4 rounded-md bg-green-50 p-3 text-sm text-green-700"
              role="status"
            >
              {success}
            </div>
          )}

          <form onSubmit={handleSubmit} className="space-y-4">
            {mode === "register" && (
              <div>
                <label
                  htmlFor="name"
                  className="mb-1 block text-sm font-medium text-gray-700"
                >
                  Name
                </label>
                <input
                  id="name"
                  type="text"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  className="input"
                  required
                  autoComplete="name"
                />
              </div>
            )}

            <div>
              <label
                htmlFor="email"
                className="mb-1 block text-sm font-medium text-gray-700"
              >
                Email
              </label>
              <input
                id="email"
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="input"
                required
                autoComplete="email"
                autoFocus
              />
            </div>

            <div>
              <label
                htmlFor="password"
                className="mb-1 block text-sm font-medium text-gray-700"
              >
                Password
              </label>
              <input
                id="password"
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="input"
                required
                autoComplete={
                  mode === "login" ? "current-password" : "new-password"
                }
                minLength={6}
              />
            </div>

            <button
              type="submit"
              disabled={loading}
              className="btn-primary w-full"
            >
              {loading
                ? "Please wait..."
                : mode === "login"
                  ? "Sign in"
                  : "Create account"}
            </button>
          </form>

          <div className="mt-4 text-center">
            <button
              onClick={() => {
                setMode(mode === "login" ? "register" : "login");
                setError(null);
                setSuccess(null);
              }}
              className="text-sm text-brand-600 hover:text-brand-700"
            >
              {mode === "login"
                ? "Need an account? Sign up"
                : "Already have an account? Sign in"}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
