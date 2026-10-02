import { useState, type FormEvent } from 'react'
import { useApp } from '../App'
import { useDocumentTitle } from '../hooks'
import { FolderIcon } from '../icons'
import { Cover, statusLine } from '../Sidebar'

export function Home({ manual }: { manual: boolean }) {
  const { books, booksLoaded, job, pickAndOpen, openFolder } = useApp()
  const [typing, setTyping] = useState(manual)
  const [path, setPath] = useState('')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  useDocumentTitle(books.length ? '책장' : '')

  const submit = async (event: FormEvent) => {
    event.preventDefault()
    if (!path.trim()) {
      setError('원고 폴더 경로를 적어 주세요.')
      return
    }
    setBusy(true)
    try {
      await openFolder(path.trim())
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  const pick = async () => {
    setBusy(true)
    await pickAndOpen()
    setBusy(false)
  }

  // 책 목록을 받기 전에는 그리지 않는다 — 첫 방문 인사말이 잠깐 떴다가 바뀌지 않게 (받는 데 보통 0.1초 안)
  if (!booksLoaded) return null

  return (
    <>
      <section className="welcome" aria-labelledby="welcome-title">
        <h1 id="welcome-title">{books.length ? '새 책 번역하기' : '만화 폴더를 고르면 번역을 시작해요'}</h1>
        <p>
          쪽 그림(jpg, png, webp)이 든 폴더를 골라 주세요. 번역한 쪽은 그 폴더 옆 ‘폴더 이름_번역’ 폴더에 같은 파일
          이름으로 저장돼요.
        </p>
        <div className="row">
          <button className="btn primary" onClick={pick} aria-disabled={busy}>
            {busy ? <span className="spinner" aria-hidden="true" /> : <FolderIcon />}
            폴더 고르기
          </button>
          {!typing && (
            <button className="link" onClick={() => setTyping(true)}>
              경로를 직접 적을게요
            </button>
          )}
        </div>
        {typing && (
          <form onSubmit={submit} className="field">
            <label htmlFor="folder-path">원고 폴더 경로</label>
            <div className="row">
              <input
                id="folder-path"
                name="folder-path"
                className="input"
                value={path}
                onChange={(e) => {
                  setPath(e.target.value)
                  setError('')
                }}
                placeholder="D:\만화\예시 만화 03권…"
                autoComplete="off"
                spellCheck={false}
                autoFocus
              />
              <button className="btn" type="submit" aria-disabled={busy}>
                열기
              </button>
            </div>
            {error && (
              <span className="error-text" role="alert">
                {error}
              </span>
            )}
          </form>
        )}
      </section>
      {books.length > 0 && (
        <section style={{ display: 'flex', flexDirection: 'column', gap: 14 }} aria-labelledby="shelf-title">
          <h2 id="shelf-title" className="section-title">
            최근 책
          </h2>
          <div className="shelf">
            {books.map((book) => (
              <a key={book.id} className="shelf-item" href={`#/book/${book.id}`}>
                <div className="sheet">
                  <Cover book={book} width={320} />
                </div>
                <b>{book.title}</b>
                <span className="num">{statusLine(book, job).text}</span>
              </a>
            ))}
          </div>
        </section>
      )}
    </>
  )
}
