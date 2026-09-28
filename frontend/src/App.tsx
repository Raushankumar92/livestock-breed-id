import { ChangeEvent, useEffect, useMemo, useState } from 'react'

type Prediction = { breed: string; confidence: number }
type Result = { filename: string; success: boolean; predictions: Prediction[]; animal_detected?: boolean; detection_confidence?: number; error?: string }
const API_URL = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

export default function App() {
  const [files, setFiles] = useState<File[]>([])
  const [results, setResults] = useState<Result[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const previews = useMemo(() => files.map(file => ({ file, url: URL.createObjectURL(file) })), [files])

  useEffect(() => () => previews.forEach(preview => URL.revokeObjectURL(preview.url)), [previews])

  function selectFiles(event: ChangeEvent<HTMLInputElement>) {
    const selected = Array.from(event.target.files ?? []).filter(file => file.type.startsWith('image/')).slice(0, 10)
    setFiles(selected)
    setResults([])
    setError('')
  }

  async function submit() {
    if (!files.length) return
    setLoading(true); setError(''); setResults([])
    try {
      const form = new FormData()
      files.forEach(file => form.append('files', file))
      const response = await fetch(`${API_URL}/api/v1/predict`, { method: 'POST', body: form })
      const body = await response.json()
      if (!response.ok) throw new Error(body.detail ?? 'The server could not classify the images.')
      setResults(body.results)
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Something went wrong.')
    } finally { setLoading(false) }
  }

  return <main>
    <section className="hero"><span>AI-powered livestock recognition</span><h1>Identify cattle breeds from photos.</h1><p>Upload up to 10 images. The system finds the animal, then returns its most likely breed and confidence.</p></section>
    <section className="panel">
      <label className="dropzone"><input type="file" accept="image/*" multiple onChange={selectFiles} /><strong>Choose livestock images</strong><small>JPG, PNG or WEBP · up to 10 images · 10 MB each</small></label>
      {previews.length > 0 && <div className="previews">{previews.map(({ file, url }) => <figure key={`${file.name}-${file.lastModified}`}><img src={url} alt="Selected upload" /><figcaption>{file.name}</figcaption></figure>)}</div>}
      <button disabled={!files.length || loading} onClick={submit}>{loading ? 'Analyzing images…' : `Analyze ${files.length} image${files.length === 1 ? '' : 's'}`}</button>
      {error && <p className="error">{error}</p>}
    </section>
    {results.length > 0 && <section className="results"><h2>Results</h2>{results.map(result => <article key={result.filename} className={!result.success ? 'failed' : ''}><div><strong>{result.filename}</strong>{result.success && <small>{result.animal_detected ? `Animal detected (${Math.round((result.detection_confidence ?? 0) * 100)}%)` : 'No livestock crop found — classified full image'}</small>}</div>{result.success ? <ol>{result.predictions.map(prediction => <li key={prediction.breed}><span>{prediction.breed}</span><b>{Math.round(prediction.confidence * 100)}%</b><i style={{ width: `${prediction.confidence * 100}%` }} /></li>)}</ol> : <span>{result.error}</span>}</article>)}</section>}
  </main>
}
