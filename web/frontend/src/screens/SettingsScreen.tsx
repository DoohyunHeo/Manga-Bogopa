import { Fragment, useEffect, useState, type FormEvent } from 'react'
import { api, post, put, type Settings } from '../api'
import { useApp } from '../App'
import { useDocumentTitle } from '../hooks'
import { AlertIcon, CheckIcon } from '../icons'

const loadedFonts = new Set<string>()

// 생각 깊이 이름 — 권장 표시는 서버가 알려 주는 기본값(translation_effort_default)에 붙인다
const EFFORT_LABELS: Record<string, string> = {
  low: '빠르게',
  medium: '보통',
  high: '꼼꼼하게',
  xhigh: '더 꼼꼼하게',
  max: '아주 꼼꼼하게',
  ultra: '가장 꼼꼼하게',
}

/** 목적격 조사 — 받침이 있으면 '을', 없으면 '를' (한글로 끝나지 않으면 '를'). */
function objectParticle(word: string) {
  const code = word.charCodeAt(word.length - 1) - 0xac00
  return code >= 0 && code < 11172 && code % 28 !== 0 ? '을' : '를'
}

function fontFamily(file: string) {
  let hash = 0
  for (const ch of file) hash = (hash * 31 + ch.charCodeAt(0)) | 0
  return `mb-font-${(hash >>> 0).toString(36)}`
}

/** 글꼴 미리보기 — 서버가 내주는 글꼴 파일을 그대로 읽는다. */
function useFontFace(file: string) {
  useEffect(() => {
    if (!file || loadedFonts.has(file)) return
    loadedFonts.add(file)
    // 한 파일이 모든 굵기를 맡는다고 알린다 — 브라우저가 굵은 파일을 한 번 더 가짜로 굵게 그리지 않고,
    // 가변 글꼴은 굵기 축을 따른다. 합성 굵게는 미리보기가 직접 그린다(FontPreview의 embolden)
    const face = new FontFace(fontFamily(file), `url(/api/fonts/${encodeURIComponent(file)})`, { weight: '1 1000' })
    face
      .load()
      .then((loaded) => document.fonts.add(loaded))
      .catch(() => loadedFonts.delete(file))
  }, [file])
}

function FontPreview({ file, sample, weight, embolden, scale }: { file: string; sample: string; weight?: number; embolden?: boolean; scale?: number }) {
  useFontFace(file)
  const narrowed = !!file && !!scale && scale !== 1
  return (
    <div
      className="preview"
      style={{
        fontFamily: file ? `'${fontFamily(file)}', var(--font)` : undefined,
        fontWeight: weight || undefined,
        // 합성 굵게를 더해 그리는 굵은 대사 — 미리보기도 획에 얇은 테를 둘러 그만큼 굵게 보인다
        WebkitTextStroke: embolden ? '0.04em currentColor' : undefined,
        // 장평 — 식자처럼 글자 폭만 좁혀 그린다 (서버가 주는 그 글씨체의 값, 왼쪽 기준)
        transform: narrowed ? `scaleX(${scale})` : undefined,
        transformOrigin: narrowed ? 'left center' : undefined,
      }}
    >
      {file ? sample : '글꼴을 골라 주세요'}
    </div>
  )
}

export function SettingsScreen() {
  const { toast, system, refreshSystem } = useApp()
  const [settings, setSettings] = useState<Settings | null>(null)
  const [login, setLogin] = useState<{ ok: boolean; message: string } | null>(null)
  const [checking, setChecking] = useState(false)
  const [allFonts, setAllFonts] = useState(false)
  const [editPath, setEditPath] = useState(false)
  const [path, setPath] = useState('')
  useDocumentTitle('설정')

  useEffect(() => {
    api<Settings>('/api/settings')
      .then(setSettings)
      .catch((e) => toast((e as Error).message, 'error'))
  }, [toast])

  if (!settings) return <p style={{ color: 'var(--tone)' }}>불러오는 중…</p>
  const locked = settings.locked

  const save = async (patch: Record<string, unknown>) => {
    try {
      setSettings(await put<Settings>('/api/settings', patch))
      toast('저장했어요. 다음 번역부터 적용돼요.')
    } catch (e) {
      toast((e as Error).message, 'error')
    }
  }

  const check = async () => {
    setChecking(true)
    try {
      await refreshSystem()
      const backend = settings && settings.translation_backend !== 'auto' ? { backend: settings.translation_backend } : {}
      setLogin(await post<{ ok: boolean; message: string }>('/api/system/login-check', backend))
    } catch (e) {
      toast((e as Error).message, 'error')
    } finally {
      setChecking(false)
    }
  }

  const savePath = async (event: FormEvent) => {
    event.preventDefault()
    await save({ agy_path: path })
    setEditPath(false)
    refreshSystem()
  }

  const fonts = allFonts ? settings.fonts : settings.fonts.slice(0, 4)
  const bold = settings.bold_font
  const found = !!system?.translation_found
  // 모델·생각 깊이를 보여 줄 CLI — 자동이면 지금 실제로 쓰는 것
  const targetId = settings.translation_backend === 'auto' ? settings.translation_active : settings.translation_backend
  const target = settings.translation_backends.find((b) => b.id === targetId) ?? null
  const modelId = target ? settings.translation_models[target.id] ?? '' : ''
  const model = target?.models.find((m) => m.id === modelId) ?? target?.models[0]
  const efforts = model?.efforts ?? target?.efforts ?? []
  const effortDefault = settings.translation_effort_default
  const effortId = target ? settings.translation_efforts[target.id] ?? effortDefault : effortDefault
  const effortName = (id: string) => EFFORT_LABELS[id] ?? efforts.find((e) => e.id === id)?.label ?? id
  const showAgyPath = settings.translation_backend === 'antigravity' || targetId === 'antigravity'

  return (
    <div className="settings">
      <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
        <h1>설정</h1>
        <p style={{ color: 'var(--tone)' }}>바꾸면 바로 저장되고, 다음 번역부터 적용돼요. 나머지는 여러 작품으로 시험해 가장 나았던 값으로 정해 두었어요.</p>
      </div>
      {locked && (
        <div className="notice info">
          <AlertIcon />
          <span className="grow">번역하는 동안에는 설정을 바꿀 수 없어요. 끝나거나 멈춘 뒤에 바꿔 주세요.</span>
        </div>
      )}

      <section className="panel" aria-labelledby="s-translate">
        <h2 id="s-translate">번역</h2>
        <div className="setting-row">
          <span className={`status-icon ${found && login?.ok !== false ? 'ok' : 'bad'}`}>
            {found && login?.ok !== false ? <CheckIcon /> : <AlertIcon />}
          </span>
          <div className="text">
            <b>{found ? `${system?.translation_label}로 번역해요` : '번역에 쓸 AI를 찾지 못했어요'}</b>
            <span>
              {login
                ? login.message
                : found
                  ? '로그인했는지는 ‘연결 확인’으로 볼 수 있어요.'
                  : 'Antigravity CLI, Claude Code, Codex CLI 가운데 하나를 설치하고 터미널에서 한 번 실행해 로그인해 주세요.'}
            </span>
          </div>
          <button className="btn small" onClick={check} aria-disabled={checking}>
            {checking && <span className="spinner" aria-hidden="true" />}
            연결 확인
          </button>
        </div>
        <div className="divider" />
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
          <label htmlFor="translation-backend" style={{ fontWeight: 700 }}>
            번역에 쓸 AI
          </label>
          <select
            id="translation-backend"
            name="translation-backend"
            className="select"
            style={{ alignSelf: 'flex-start', minWidth: 240 }}
            value={settings.translation_backend}
            disabled={locked}
            onChange={(e) => {
              setLogin(null)
              save({ translation_backend: e.target.value }).then(refreshSystem)
            }}
          >
            {settings.translation_choices.map((choice) => {
              const backend = settings.translation_backends.find((b) => b.id === choice.id)
              return (
                <option key={choice.id} value={choice.id}>
                  {choice.label}
                  {backend && !backend.found ? ' (설치 안 됨)' : ''}
                </option>
              )
            })}
          </select>
          {target && (
            <>
              <label htmlFor="translation-model" style={{ fontWeight: 700 }}>
                모델
              </label>
              <select
                id="translation-model"
                name="translation-model"
                className="select"
                style={{ alignSelf: 'flex-start', minWidth: 240 }}
                value={modelId}
                disabled={locked}
                onChange={(e) => save({ translation_models: { [target.id]: e.target.value } })}
              >
                {target.models.map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.id === '' ? '기본 (CLI가 정하는 모델)' : m.label}
                  </option>
                ))}
              </select>
            </>
          )}
          {target && efforts.length > 0 && (
            <>
              <label htmlFor="translation-effort" style={{ fontWeight: 700 }}>
                생각 깊이
              </label>
              <select
                id="translation-effort"
                name="translation-effort"
                className="select"
                style={{ alignSelf: 'flex-start', minWidth: 240 }}
                value={effortId}
                disabled={locked}
                onChange={(e) => save({ translation_efforts: { [target.id]: e.target.value } })}
              >
                {efforts.map((effort) => (
                  <option key={effort.id} value={effort.id}>
                    {effortName(effort.id)}
                    {effort.id === effortDefault ? ' (권장)' : ''}
                  </option>
                ))}
              </select>
              <span style={{ fontSize: 13, color: 'var(--tone)' }}>
                깊을수록 번역을 오래 고민해요. ‘{effortName(effortDefault)}’{objectParticle(effortName(effortDefault))} 권해요.
                {!['low', 'medium'].includes(effortDefault) && efforts.some((e) => e.id === 'medium') ? ' 빨리 끝내고 싶으면 ‘보통’으로 낮춰 보세요.' : ''}
              </span>
            </>
          )}
          {!target && settings.translation_backend === 'auto' && (
            <span style={{ fontSize: 13, color: 'var(--tone)' }}>설치된 AI를 찾으면 여기서 모델과 생각 깊이를 고를 수 있어요.</span>
          )}
        </div>
        {showAgyPath && (
        <div className="setting-row" style={{ alignItems: 'flex-start' }}>
          <div className="text">
            <b>Antigravity CLI 위치</b>
            <span style={{ overflowWrap: 'anywhere' }}>
              {settings.agy_path ? settings.agy_path : system?.agy_path ? `${system.agy_path} (자동으로 찾음)` : '자동으로 찾아요'}
            </span>
          </div>
          {!editPath && (
            <button className="link" onClick={() => { setPath(settings.agy_path); setEditPath(true) }} disabled={locked}>
              다른 위치 적기
            </button>
          )}
        </div>
        )}
        {showAgyPath && editPath && (
          <form className="field" onSubmit={savePath}>
            <label htmlFor="agy-path">agy.exe 전체 경로 (비우면 자동으로 찾아요)</label>
            <div className="row">
              <input id="agy-path" name="agy-path" className="input" value={path} onChange={(e) => setPath(e.target.value)} placeholder="C:\Users\…\AppData\Local\agy\bin\agy.exe" spellCheck={false} autoComplete="off" />
              <button className="btn" type="submit">저장</button>
              <button className="btn ghost" type="button" onClick={() => setEditPath(false)}>그만두기</button>
            </div>
          </form>
        )}
        <div className="divider" />
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
          <label htmlFor="sessions" style={{ fontWeight: 700 }}>
            번역 동시 진행
          </label>
          <select
            id="sessions"
            name="sessions"
            className="select"
            style={{ alignSelf: 'flex-start', minWidth: 180 }}
            value={settings.sessions}
            disabled={locked}
            onChange={(e) => save({ sessions: Number(e.target.value) })}
          >
            {Array.from({ length: settings.sessions_range[1] - settings.sessions_range[0] + 1 }, (_, i) => settings.sessions_range[0] + i).map((n) => (
              <option key={n} value={n}>
                {n}묶음{n === settings.sessions_default ? ' (권장)' : ''}
              </option>
            ))}
          </select>
          <span style={{ fontSize: 13, color: 'var(--tone)' }}>
            쪽을 읽는 동안 뒤에서 번역해요. 번역이 밀리면 여러 묶음을 함께 보내요. 많을수록 빠르지만 사용량 한도에 일찍 닿을 수 있어요.
          </span>
        </div>
        <div className="divider" />
        <div className="setting-row">
          <div className="text">
            <b id="fit-label">좁은 말풍선은 한 번 더 짧게 번역하기</b>
            <span>글자가 너무 작아지거나 낱말이 두 줄로 쪼개지는 말풍선만 번역기에 더 짧은 번역을 한 번 더 받아요. 그런 말풍선이 조금 줄지만, 20쪽짜리 한 화에 15초쯤 더 걸려요.</span>
          </div>
          <button className="switch" role="switch" aria-checked={settings.fit_translation} aria-labelledby="fit-label" disabled={locked} onClick={() => save({ fit_translation: !settings.fit_translation })}>
            <span />
          </button>
        </div>
      </section>

      <section className="panel" aria-labelledby="s-fonts">
        <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
          <h2 id="s-fonts">글꼴</h2>
          <p className="lead">원문 글씨의 생김새에 맞춰 이 글꼴로 써요. 글꼴 파일(.ttf, .otf)을 data/fonts 폴더에 넣으면 목록에 나와요.</p>
        </div>
        {fonts.map((font) => (
          <Fragment key={font.style}>
          <div className="font-row">
            <label htmlFor={`font-${font.style}`}>{font.label}</label>
            <select
              id={`font-${font.style}`}
              name={`font-${font.style}`}
              value={font.file}
              disabled={locked}
              onChange={(e) => save({ fonts: { [font.style]: e.target.value } })}
            >
              {!settings.font_files.includes(font.file) && <option value={font.file}>{font.file || '고르지 않음'}</option>}
              {settings.font_files.map((file) => (
                <option key={file} value={file}>
                  {file.replace(/\.(ttf|otf|ttc)$/i, '')}
                </option>
              ))}
            </select>
            <FontPreview file={font.file} sample={font.sample} scale={font.horizontal_scale} />
          </div>
          {font.style === 'standard' && (
            // 굵은 대사는 고르지 않는다 — 평범한 대사 글꼴과 같은 가족의 굵은 파일로 저절로 정해진다
            <div className="font-row">
              <span className="label">굵은 대사</span>
              <span className="bold-auto">
                {bold.synthetic || !bold.file
                  ? `${font.file.replace(/\.(ttf|otf|ttc)$/i, '')} (합성 굵게)`
                  : `${bold.file.replace(/\.(ttf|otf|ttc)$/i, '')} (자동)${bold.emboldened ? ' + 합성 굵게' : ''}`}
              </span>
              <FontPreview
                file={bold.synthetic || !bold.file ? font.file : bold.file}
                sample="잠깐만!!"
                weight={bold.weight || 700}
                embolden={bold.synthetic || !bold.file || !!bold.emboldened}
                scale={bold.horizontal_scale}
              />
            </div>
          )}
          </Fragment>
        ))}
        <button className="btn small" style={{ alignSelf: 'flex-start' }} onClick={() => setAllFonts((v) => !v)}>
          {allFonts ? '글꼴 4개만 보기' : `글꼴 ${settings.fonts.length - 4}개 더 보기`}
        </button>
      </section>

      <section className="panel" aria-labelledby="s-other">
        <h2 id="s-other">글자 쓰기와 컴퓨터</h2>
        <div className="setting-row">
          <div className="text">
            <b id="narration-label">해설도 평범한 대사 글꼴로 쓰기</b>
            <span>켜면 해설 글꼴로 쓰던 글자를 모두 평범한 대사 글꼴로 써요. 원문에서 가는 명조체로 쓴 글자, 말풍선 밖의 평범한 글자, 꼬리 없는 네모 해설 상자가 여기에 들어가요.</span>
          </div>
          <button className="switch" role="switch" aria-checked={settings.narration_as_standard} aria-labelledby="narration-label" disabled={locked} onClick={() => save({ narration_as_standard: !settings.narration_as_standard })}>
            <span />
          </button>
        </div>
        <div className="divider" />
        <div className="setting-row">
          <div className="text">
            <b id="vertical-label">좁고 긴 말풍선은 세로로 쓰기</b>
            <span>세로쓰기는 늘 한 줄로만 써요. 끄면 모두 가로로 써요.</span>
          </div>
          <button className="switch" role="switch" aria-checked={settings.vertical_text} aria-labelledby="vertical-label" disabled={locked} onClick={() => save({ vertical_text: !settings.vertical_text })}>
            <span />
          </button>
        </div>
        <div className="divider" />
        <div className="setting-row">
          <div className="text">
            <b id="slant-label">외침은 기울여 쓰기</b>
            <span>느낌표가 붙은 말풍선 대사를 오른쪽으로 조금 기울여 써요. 세로로 쓴 대사는 그대로예요. 정식판도 출판사마다 달라서 기본은 꺼 두었어요.</span>
          </div>
          <button className="switch" role="switch" aria-checked={settings.slant_exclamations} aria-labelledby="slant-label" disabled={locked} onClick={() => save({ slant_exclamations: !settings.slant_exclamations })}>
            <span />
          </button>
        </div>
        <div className="divider" />
        <div className="setting-row">
          <div className="text">
            <b id="vram-label">그래픽카드 메모리 아끼기</b>
            <span>메모리가 모자란다는 오류가 날 때만 켜 주세요. 조금 느려져요.{system?.gpu ? ` 지금 그래픽카드: ${system.gpu}` : ''}</span>
          </div>
          <button className="switch" role="switch" aria-checked={settings.low_vram} aria-labelledby="vram-label" disabled={locked} onClick={() => save({ low_vram: !settings.low_vram })}>
            <span />
          </button>
        </div>
      </section>
    </div>
  )
}
