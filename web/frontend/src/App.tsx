import { createContext, useCallback, useContext, useEffect, useState } from 'react'
import { api, ApiError, post, type Book, type Job, type System } from './api'
import { go, setTitleProgress, useJob, useRoute } from './hooks'
import { Sidebar } from './Sidebar'
import { StatusBar, typesetPercent } from './StatusBar'
import { BookScreen } from './screens/BookScreen'
import { Home } from './screens/Home'
import { SettingsScreen } from './screens/SettingsScreen'
import { SetupScreen } from './screens/SetupScreen'

type Tone = 'info' | 'error'
type Toast = { id: number; text: string; tone: Tone }

type AppContext = {
  system: System | null
  refreshSystem: () => Promise<void>
  books: Book[]
  booksLoaded: boolean
  refreshBooks: () => Promise<void>
  job: Job | null
  toast: (text: string, tone?: Tone) => void
  openFolder: (path: string) => Promise<void>
  pickAndOpen: () => Promise<void>
}

const Ctx = createContext<AppContext | null>(null)
export const useApp = () => useContext(Ctx)!

export function App() {
  const route = useRoute()
  const [job, connected] = useJob()
  const [system, setSystem] = useState<System | null>(null)
  const [books, setBooks] = useState<Book[]>([])
  const [booksLoaded, setBooksLoaded] = useState(false) // 처음 읽기 전에 '책이 없어요'가 잠깐 뜨지 않게
  const [toasts, setToasts] = useState<Toast[]>([])
  const [skipSetup, setSkipSetup] = useState(false)

  const toast = useCallback((text: string, tone: Tone = 'info') => {
    const id = Date.now() + Math.random()
    setToasts((list) => [...list, { id, text, tone }])
    window.setTimeout(() => setToasts((list) => list.filter((t) => t.id !== id)), tone === 'error' ? 6500 : 2600)
  }, [])

  const refreshSystem = useCallback(async () => {
    try {
      setSystem(await api<System>('/api/system'))
    } catch (error) {
      toast((error as Error).message, 'error')
    }
  }, [toast])

  const refreshBooks = useCallback(async () => {
    try {
      setBooks((await api<{ books: Book[] }>('/api/books')).books)
      setBooksLoaded(true)
    } catch (error) {
      toast((error as Error).message, 'error')
    }
  }, [toast])

  useEffect(() => {
    refreshSystem()
    refreshBooks()
  }, [refreshSystem, refreshBooks])

  // 작업이 시작되거나 끝나면 책장 상태를 다시 읽는다 (쪽마다 다시 읽지는 않는다)
  const jobKey = job ? `${job.book_id}:${job.status}` : ''
  useEffect(() => {
    if (jobKey) refreshBooks()
  }, [jobKey, refreshBooks])

  // 창 제목에 진행을 붙인다 — 다른 창을 보고 있어도 작업 표시줄에서 보이게
  const [dismissed, setDismissed] = useState('')
  const jobId = job ? `${job.book_id}:${job.started_at}` : ''
  useEffect(() => {
    const percent = job ? typesetPercent(job) : null
    setTitleProgress(job?.status === 'running' ? `(${percent !== null ? `${percent}%` : '진행 중'}) ` : '')
  }, [job])

  const openFolder = useCallback(
    async (path: string) => {
      const book = await post<Book>('/api/books', { input_dir: path })
      await refreshBooks()
      go(`book/${book.id}`)
    },
    [refreshBooks],
  )

  const pickAndOpen = useCallback(async () => {
    try {
      const picked = await post<{ path: string | null }>('/api/dialog/folder', {})
      if (picked.path) await openFolder(picked.path)
    } catch (error) {
      if (error instanceof ApiError && error.status === 501) go('new')
      toast((error as Error).message, 'error')
    }
  }, [openFolder, toast])

  // 작업 막대는 그 책 화면 밖에서만 (책 화면에는 진행 칸이 있다)
  const showBar = !!job && dismissed !== jobId && !(route[0] === 'book' && route[1] === job.book_id)
  const needsSetup = system !== null && !system.translation_found && !skipSetup
  let screen
  if (needsSetup) screen = <SetupScreen onSkip={() => setSkipSetup(true)} />
  else if (route[0] === 'book' && route[1]) screen = <BookScreen key={route[1]} id={route[1]} />
  else if (route[0] === 'settings') screen = <SettingsScreen />
  // #/new(폴더 고르기 창을 못 띄울 때 가는 곳)로 오면 경로 칸이 열린 채로 새로 그린다
  else screen = <Home key={route[0] === 'new' ? 'new' : 'home'} manual={route[0] === 'new'} />

  return (
    <Ctx.Provider value={{ system, refreshSystem, books, booksLoaded, refreshBooks, job, toast, openFolder, pickAndOpen }}>
      <a
        className="skip-link"
        href="#main"
        onClick={(e) => {
          e.preventDefault() // 주소의 #/… 경로를 바꾸지 않고 초점만 옮긴다
          document.getElementById('main')?.focus()
        }}
      >
        본문으로 바로 가기
      </a>
      {!connected && (
        <div className="banner" role="alert">
          만화보고파 프로그램과 연결이 끊겼어요. 실행한 창(python main.py)이 켜져 있는지 확인해 주세요. 다시 켜면 저절로 이어져요.
        </div>
      )}
      <div className={`app${showBar ? ' with-bar' : ''}`}>
        <Sidebar route={route} />
        <main className="main" id="main" tabIndex={-1}>
          {screen}
        </main>
      </div>
      {job && showBar && <StatusBar job={job} onDismiss={() => setDismissed(jobId)} />}
      <div className="toasts" aria-live="polite">
        {toasts.map((t) => (
          <div key={t.id} className={`toast ${t.tone}`} role={t.tone === 'error' ? 'alert' : 'status'}>
            {t.text}
          </div>
        ))}
      </div>
    </Ctx.Provider>
  )
}
