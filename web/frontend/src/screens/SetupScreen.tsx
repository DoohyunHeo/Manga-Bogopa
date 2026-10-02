import { useState, type FormEvent } from 'react'
import { post, put } from '../api'
import { useApp } from '../App'
import { useDocumentTitle } from '../hooks'
import { CopyIcon } from '../icons'

const INSTALL = 'irm https://antigravity.google/cli/install.ps1 | iex'

/** 첫 실행 — 번역을 맡을 CLI(Antigravity CLI·Claude Code·Codex CLI)를 하나도 찾지 못했을 때. */
export function SetupScreen({ onSkip }: { onSkip: () => void }) {
  const { refreshSystem, system, toast } = useApp()
  const [copied, setCopied] = useState(false)
  const [checking, setChecking] = useState(false)
  const [path, setPath] = useState('')
  const [pathError, setPathError] = useState('')
  useDocumentTitle('번역 도구 연결')

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(INSTALL)
      setCopied(true)
      window.setTimeout(() => setCopied(false), 2000)
    } catch {
      toast('복사하지 못했어요. 명령을 직접 선택해서 복사해 주세요.', 'error')
    }
  }

  const recheck = async () => {
    setChecking(true)
    await refreshSystem()
    setChecking(false)
  }

  const savePath = async (event: FormEvent) => {
    event.preventDefault()
    if (!path.trim()) {
      setPathError('agy.exe 경로를 적어 주세요.')
      return
    }
    try {
      await put('/api/settings', { agy_path: path.trim() })
      await refreshSystem()
      const found = await post<{ ok: boolean; message: string }>('/api/system/login-check', { backend: 'antigravity' })
      if (!found.ok) setPathError(found.message)
    } catch (e) {
      setPathError((e as Error).message)
    }
  }

  return (
    <div className="welcome" style={{ gap: 24 }}>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
        <h1>번역을 맡을 도구를 연결해 주세요</h1>
        <p>
          번역은 구글의 Antigravity CLI가 맡아요. 한 번만 설치하고 로그인해 두면, 그다음부터는 폴더만 고르면 돼요. API 키는
          필요 없어요. Claude Code나 Codex CLI를 이미 쓰고 있다면 그걸로도 번역할 수 있어요. 로그인한 뒤 ‘다시 확인’을 눌러 주세요.
        </p>
      </div>
      <section className="panel" aria-label="연결 순서">
        <div className="setting-row" style={{ alignItems: 'flex-start' }}>
          <span className="status-icon ok num">1</span>
          <div className="text" style={{ gap: 8 }}>
            <b>PowerShell에서 설치 명령을 실행해요</b>
            <code className="cmd">{INSTALL}</code>
            <div>
              <button className="btn small" onClick={copy}>
                <CopyIcon />
                {copied ? '복사했어요' : '명령 복사'}
              </button>
            </div>
          </div>
        </div>
        <div className="divider" />
        <div className="setting-row" style={{ alignItems: 'flex-start' }}>
          <span className="status-icon ok num">2</span>
          <div className="text">
            <b>새 터미널에서 agy를 한 번 실행해 구글 계정으로 로그인해요</b>
            <span>설치 직후에는 터미널을 새로 열어야 agy를 찾을 수 있어요. 그래도 안 되면 컴퓨터를 한 번 다시 켜 주세요.</span>
          </div>
        </div>
        <div className="divider" />
        <div className="setting-row">
          <span className="status-icon ok num">3</span>
          <div className="text">
            <b>다 됐으면 다시 확인해요</b>
            <span>{system?.translation_found ? `${system.translation_label}를 찾았어요!` : '아직 찾지 못했어요.'}</span>
          </div>
          <button className="btn primary" onClick={recheck} aria-disabled={checking}>
            {checking && <span className="spinner" aria-hidden="true" />}
            다시 확인
          </button>
        </div>
      </section>
      <form className="field" onSubmit={savePath}>
        <label htmlFor="setup-path">다른 곳에 설치했다면 agy.exe 위치를 적어 주세요</label>
        <div className="row">
          <input id="setup-path" name="setup-path" className="input" value={path} onChange={(e) => { setPath(e.target.value); setPathError('') }} placeholder="C:\Users\…\AppData\Local\agy\bin\agy.exe" spellCheck={false} autoComplete="off" />
          <button className="btn" type="submit">저장</button>
        </div>
        {pathError && <span className="error-text" role="alert">{pathError}</span>}
      </form>
      <div>
        <button className="link" onClick={onSkip}>
          나중에 할게요 (이미 번역한 책은 볼 수 있어요)
        </button>
      </div>
    </div>
  )
}
