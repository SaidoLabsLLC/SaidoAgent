/**
 * Auth state management hook.
 *
 * Stores JWT and user metadata in localStorage and provides login/logout
 * helpers. Components consuming this hook re-render when auth state changes.
 */

import { useCallback, useEffect, useState } from "react";
import {
  apiPost,
  clearStoredToken,
  getStoredToken,
  setStoredToken,
} from "../api/client";
import type { AuthState, LoginRequest, LoginResponse } from "../types";

const AUTH_META_KEY = "saido_auth_meta";

function loadAuthState(): AuthState {
  const token = getStoredToken();
  if (!token) {
    return {
      token: null,
      user_id: null,
      email: null,
      team_id: null,
      role: null,
      isAuthenticated: false,
    };
  }

  try {
    const meta = JSON.parse(localStorage.getItem(AUTH_META_KEY) || "{}");
    return {
      token,
      user_id: meta.user_id || null,
      email: meta.email || null,
      team_id: meta.team_id || null,
      role: meta.role || null,
      isAuthenticated: true,
    };
  } catch {
    return {
      token,
      user_id: null,
      email: null,
      team_id: null,
      role: null,
      isAuthenticated: true,
    };
  }
}

export function useAuth() {
  const [auth, setAuth] = useState<AuthState>(loadAuthState);

  // Re-check on storage events (e.g. other tabs)
  useEffect(() => {
    function handleStorage() {
      setAuth(loadAuthState());
    }
    window.addEventListener("storage", handleStorage);
    return () => window.removeEventListener("storage", handleStorage);
  }, []);

  const login = useCallback(
    async (credentials: LoginRequest): Promise<LoginResponse> => {
      const res = await apiPost<LoginResponse>(
        "/v1/auth/login",
        credentials,
      );
      setStoredToken(res.token);
      const meta = {
        user_id: res.user_id,
        email: res.email,
        team_id: res.team_id,
        role: res.role,
      };
      localStorage.setItem(AUTH_META_KEY, JSON.stringify(meta));
      setAuth({
        ...meta,
        token: res.token,
        isAuthenticated: true,
      });
      return res;
    },
    [],
  );

  const logout = useCallback(() => {
    clearStoredToken();
    localStorage.removeItem(AUTH_META_KEY);
    setAuth({
      token: null,
      user_id: null,
      email: null,
      team_id: null,
      role: null,
      isAuthenticated: false,
    });
  }, []);

  return { auth, login, logout };
}
