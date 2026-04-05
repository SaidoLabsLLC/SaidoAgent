/* -----------------------------------------------------------------------
 * TypeScript types matching the Saido Agent API response models.
 * Keep in sync with saido_agent/api/routes.py Pydantic schemas.
 * ----------------------------------------------------------------------- */

// -- Auth -----------------------------------------------------------------

export interface LoginRequest {
  email: string;
  password: string;
  team_id?: string;
}

export interface LoginResponse {
  token: string;
  user_id: string;
  email: string;
  team_id: string;
  role: string;
  expires_in: number;
}

export interface RegisterRequest {
  email: string;
  name: string;
  password: string;
}

export interface RegisterResponse {
  user_id: string;
  email: string;
  name: string;
}

// -- Teams ----------------------------------------------------------------

export interface Team {
  id: string;
  name: string;
  owner_id: string;
  role?: string;
}

export interface TeamMember {
  user_id: string;
  email: string;
  name: string;
  role: string;
  team_id: string;
}

// -- Documents ------------------------------------------------------------

export interface DocumentSummary {
  slug: string;
  title: string;
  summary: string;
  categories: string[];
  updated: string | null;
}

export interface DocumentDetail {
  slug: string;
  title: string;
  body: string;
  summary: string;
  categories: string[];
  frontmatter: Record<string, unknown>;
}

// -- Search ---------------------------------------------------------------

export interface SearchResult {
  slug: string;
  title: string;
  summary: string;
  score: number;
  snippet: string;
}

// -- Query / Chat ---------------------------------------------------------

export interface Citation {
  slug: string;
  title: string;
  excerpt: string;
  verified: boolean;
}

export interface QueryRequest {
  question: string;
  context?: Record<string, unknown>;
  top_k?: number;
}

export interface QueryResponse {
  answer: string;
  citations: Citation[];
  confidence: string;
  retrieval_stats: Record<string, unknown>;
  tokens_used: number;
  provider: string;
}

// -- SSE streaming event types --------------------------------------------

export interface StreamTokenEvent {
  type: "token";
  content: string;
}

export interface StreamDoneEvent {
  type: "done";
  result: QueryResponse;
}

export type StreamEvent = StreamTokenEvent | StreamDoneEvent;

// -- Ingest ---------------------------------------------------------------

export interface IngestRequest {
  content: string;
  filename: string;
  metadata?: Record<string, unknown>;
}

export interface IngestResponse {
  slug: string;
  title: string;
  status: string;
  children: string[];
  error: string | null;
}

// -- Stats ----------------------------------------------------------------

export interface StatsResponse {
  document_count: number;
  category_count: number;
  concept_count: number;
  total_size_bytes: number;
}

// -- Chat UI state --------------------------------------------------------

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
  timestamp: number;
  streaming?: boolean;
}

// -- Auth state -----------------------------------------------------------

export interface AuthState {
  token: string | null;
  user_id: string | null;
  email: string | null;
  team_id: string | null;
  role: string | null;
  isAuthenticated: boolean;
}
