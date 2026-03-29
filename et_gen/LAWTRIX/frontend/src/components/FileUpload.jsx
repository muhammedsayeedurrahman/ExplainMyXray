import { useCallback, useState } from 'react'
import { useDropzone } from 'react-dropzone'
import { motion, AnimatePresence } from 'framer-motion'
import { API_BASE } from '../config'

export default function FileUpload({ onUploadComplete }) {
  const [uploading, setUploading] = useState(false)
  const [result, setResult] = useState(null)

  const onDrop = useCallback(async (files) => {
    if (!files.length) return
    setUploading(true)
    try {
      const formData = new FormData()
      formData.append('file', files[0])
      const res = await fetch(`${API_BASE}/upload`, {
        method: 'POST',
        body: formData,
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data.detail || 'Upload failed')
      setResult(data)
      onUploadComplete?.(data)
    } catch (err) {
      setResult({ error: err.message })
    } finally {
      setUploading(false)
    }
  }, [onUploadComplete])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'text/csv': ['.csv'],
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
    },
    maxFiles: 1,
  })

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      className="glass-card p-8"
    >
      <h2 className="text-lg font-semibold mb-4">Upload Invoice Data</h2>

      <div
        {...getRootProps()}
        className={`border-2 border-dashed rounded-xl p-12 text-center cursor-pointer transition-all
          ${isDragActive ? 'border-blue-500 bg-blue-500/10' : 'border-gray-700 hover:border-gray-600'}
          ${uploading ? 'opacity-50 pointer-events-none' : ''}`}
      >
        <input {...getInputProps()} />
        <div className="text-4xl mb-3">{uploading ? '...' : '+'}</div>
        <p className="text-gray-400">
          {isDragActive
            ? 'Drop your file here...'
            : uploading
            ? 'Analyzing invoices...'
            : 'Drag & drop CSV or Excel file, or click to browse'}
        </p>
        <p className="text-xs text-gray-600 mt-2">Supported: .csv, .xlsx</p>
      </div>

      <AnimatePresence>
        {result && !result.error && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="mt-4 p-4 rounded-xl bg-green-500/10 border border-green-500/30 text-green-400 text-sm"
          >
            {result.message} (Session: {result.session_id})
          </motion.div>
        )}
        {result?.error && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="mt-4 p-4 rounded-xl bg-red-500/10 border border-red-500/30 text-red-400 text-sm"
          >
            {result.error}
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  )
}
