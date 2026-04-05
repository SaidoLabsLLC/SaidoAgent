/**
 * Root application component with route definitions and auth gating.
 */

import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import Layout from "./components/Layout";
import { useAuth } from "./hooks/useAuth";
import Chat from "./pages/Chat";
import Cost from "./pages/Cost";
import Dashboard from "./pages/Dashboard";
import DocumentDetail from "./pages/DocumentDetail";
import Documents from "./pages/Documents";
import Ingest from "./pages/Ingest";
import Login from "./pages/Login";
import Settings from "./pages/Settings";
import type { LoginRequest } from "./types";

export default function App() {
  const { auth, login, logout } = useAuth();

  const handleLogin = async (credentials: LoginRequest) => {
    await login(credentials);
  };

  if (!auth.isAuthenticated) {
    return <Login onLogin={handleLogin} />;
  }

  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout onLogout={logout} email={auth.email} />}>
          <Route path="/" element={<Dashboard />} />
          <Route path="/chat" element={<Chat />} />
          <Route path="/documents" element={<Documents />} />
          <Route path="/documents/:slug" element={<DocumentDetail />} />
          <Route path="/ingest" element={<Ingest />} />
          <Route path="/costs" element={<Cost />} />
          <Route path="/settings" element={<Settings />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
