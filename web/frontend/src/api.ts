// 파이썬 서버(web/server.py)와 주고받는 모양과 부르는 함수.

export type PageInfo = { name: string; typeset: boolean; translated: boolean; version: number }

export type BookStatus = 'new' | 'partial' | 'needs_retry' | 'done' | 'missing'

export type Book = {
  id: string
  title: string
  input_dir: string
  output_dir: string
  status: BookStatus
  page_count: number
  typeset_count: number
  translated_count: number
  line_count: number
  failed_lines: number
  cover: string | null
  pages?: PageInfo[]
}

export type JobStatus = 'running' | 'done' | 'incomplete' | 'stopped' | 'error'
export type JobStage = 'preparing' | 'reading' | 'translating' | 'typesetting'

export type LineStatus = 'pending' | 'translated' | 'failed' | 'excluded'

/** 쪽 하나의 대사 — box는 원본 그림 픽셀 좌표 (x1, y1, x2, y2). */
export type Line = {
  id: string
  kind: 'bubble' | 'freeform'
  box: [number, number, number, number]
  original: string
  translation: string
  status: LineStatus
  note: string | null
}

export type PageLines = { page: string; size: [number, number]; version: string; has_page: boolean; lines: Line[] }

export type Job = {
  book_id: string
  title: string
  mode: RunMode
  pages: string[]
  status: JobStatus
  stopping: boolean
  stage: JobStage
  substage: string
  batch: number
  batches: number
  pages_done: number
  pages_total: number
  pages_skipped: number
  eta_sec: number | null
  elapsed_sec: number
  stage_elapsed_sec: number
  done_pages: { name: string; at: number }[]
  notices: { level: string; message: string }[]
  log: string[]
  message: string
  error_kind: string
  activity: string
  translation: { done: number; total: number } | null // 뒤에서 도는 번역 묶음 (읽는 동안에도)
  download: { label: string; done: number; total: number; index: number; count: number } | null // 처음 한 번 받는 모델 파일
  started_at: number
  version: number
  received_at?: number
}

export type RunMode = 'resume' | 'fresh' | 'retypeset'

// horizontal_scale (읽기 전용): 말풍선 안 보통 크기 대사를 그 글씨체로 그릴 때의 장평 — 미리보기가 따른다
export type FontChoice = { style: string; label: string; sample: string; file: string; horizontal_scale?: number }

export type Settings = {
  translation_backend: string // auto / antigravity / claude / codex
  translation_choices: Choice[]
  translation_active: string | null // 지금 실제로 쓰는 CLI
  translation_models: Record<string, string>
  translation_efforts: Record<string, string>
  translation_effort_default: string // 읽기 전용 — 권장 표시와 저장 값이 없을 때 쓰는 값 (config.DEFAULT_TRANSLATION_EFFORT)
  translation_backends: TranslationBackend[]
  sessions: number
  sessions_range: [number, number]
  sessions_default: number
  agy_path: string
  vertical_text: boolean
  low_vram: boolean
  narration_as_standard: boolean
  fit_translation: boolean // 좁은 말풍선만 짧은 번역을 한 번 더 받기 (기본 끔, config.ENABLE_FIT_TRANSLATION)
  slant_exclamations: boolean // 느낌표가 붙은 말풍선 대사를 가로쓰기일 때 기울여 쓰기 (기본 끔, config.SLANT_EXCLAMATIONS)
  fonts: FontChoice[]
  // 굵은 대사 (읽기 전용, 저절로 정해짐): synthetic = 굵은 파일이 없어 보통 글꼴을 합성 굵게,
  // emboldened = 굵은 파일이 있어도 획이 충분히 굵지 않아 합성 굵게를 더함 (synthetic이면 늘 true), horizontal_scale = 굵은 대사의 장평
  bold_font: { file: string; weight: number; synthetic: boolean; emboldened?: boolean; horizontal_scale?: number }
  font_files: string[]
  locked: boolean
}

export type System = {
  agy_found: boolean // agy(Antigravity CLI)만의 뜻 — 번역 연결 여부는 translation_found
  agy_path: string | null
  translation_found: boolean
  translation_backend: string | null
  translation_label: string | null
  translation_path: string | null
  gpu: string | null
  busy: boolean
}

export type Choice = { id: string; label: string }
export type TranslationBackend = {
  id: string
  label: string
  found: boolean
  path: string | null
  models: (Choice & { efforts?: Choice[] })[] // 첫 항목은 id ''(기본). efforts가 있으면 그 모델은 그 값만 받는다
  efforts: Choice[] // 비면 생각 깊이 칸을 숨긴다
}

export class ApiError extends Error {
  status: number
  constructor(message: string, status: number) {
    super(message)
    this.status = status
  }
}

export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response
  try {
    res = await fetch(path, { headers: { 'Content-Type': 'application/json' }, ...init })
  } catch {
    throw new ApiError('프로그램과 연결이 끊겼어요. 만화보고파를 실행한 창이 켜져 있는지 확인해 주세요.', 0)
  }
  const data = await res.json().catch(() => ({}))
  if (!res.ok) throw new ApiError(data.message || `요청이 실패했어요 (${res.status})`, res.status)
  return data as T
}

export const post = <T>(path: string, body: unknown = {}) =>
  api<T>(path, { method: 'POST', body: JSON.stringify(body) })

export const put = <T>(path: string, body: unknown) => api<T>(path, { method: 'PUT', body: JSON.stringify(body) })

export function pageUrl(bookId: string, name: string, kind: 'original' | 'translated', width = 0, version = 0) {
  const params = new URLSearchParams()
  if (width) params.set('w', String(width))
  if (version) params.set('v', String(version))
  const query = params.toString()
  return `/api/books/${bookId}/pages/${encodeURIComponent(name)}/${kind}${query ? `?${query}` : ''}`
}
